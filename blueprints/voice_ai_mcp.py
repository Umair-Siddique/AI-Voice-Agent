import json
import base64
import asyncio
import threading
import io
import wave
try:
    import audioop
except ModuleNotFoundError:  # Python 3.13 removed audioop
    import audioop_lts as audioop
import uuid
from typing import Any, Dict, List, Optional

from flask import Blueprint, request, Response, jsonify
from twilio.twiml.voice_response import VoiceResponse, Connect

from config import Config
from utils import SYSTEM_MESSAGE
from blueprints import mcp_server

VOICE = getattr(Config, "VOICE_TTS_VOICE", "alloy")
STT_MODEL = getattr(Config, "VOICE_STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = getattr(Config, "VOICE_TTS_MODEL", "gpt-4o-mini-tts")
LLM_MODEL = getattr(Config, "VOICE_LLM_MODEL", getattr(mcp_server, "AGENT_MODEL", "gpt-5"))
SILENCE_RMS_THRESHOLD = getattr(Config, "VOICE_SILENCE_THRESHOLD", 700)
SILENCE_DURATION_MS = getattr(Config, "VOICE_SILENCE_DURATION_MS", 1200)
MAX_UTTERANCE_MS = getattr(Config, "VOICE_MAX_UTTERANCE_MS", 6000)
MIN_UTTERANCE_PCM_BYTES = getattr(Config, "VOICE_MIN_UTTERANCE_PCM_BYTES", 3200)
TWILIO_CHUNK_SAMPLES = 160  # 20ms of mono @ 8kHz in µ-law

voice_mcp_bp = Blueprint('voicemcp', __name__)
voice_conversations: Dict[str, List[Dict[str, str]]] = {}


def _ensure_conversation(session_id: str) -> List[Dict[str, str]]:
    if session_id not in voice_conversations:
        voice_conversations[session_id] = [{
            "role": "system",
            "content": SYSTEM_MESSAGE,
        }]
    return voice_conversations[session_id]


def _extract_text_from_response(response: Any) -> str:
    try:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
    except Exception:
        pass

    try:
        first_output = response.output[0]
        first_content = first_output.content[0]
        text_value = (getattr(first_content, "text", "") or "").strip()
        if text_value:
            return text_value
    except Exception:
        pass

    return str(response)


async def _transcribe_pcm(pcm_bytes: bytes) -> str:
    if not pcm_bytes or len(pcm_bytes) < MIN_UTTERANCE_PCM_BYTES:
        return ""

    def _call_openai():
        buffer_io = io.BytesIO()
        with wave.open(buffer_io, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(8000)
            wav_out.writeframes(pcm_bytes)
        buffer_io.seek(0)
        result = mcp_server.openai_client.audio.transcriptions.create(
            model=STT_MODEL,
            file=("speech.wav", buffer_io),
        )
        return getattr(result, "text", "").strip()

    return await asyncio.to_thread(_call_openai)


async def _run_llm_with_tools(session_id: str, user_text: str) -> str:
    def _call_openai():
        messages = _ensure_conversation(session_id)
        messages.append({"role": "user", "content": user_text})
        try:
            response = mcp_server.openai_client.responses.create(
                model=LLM_MODEL,
                input=messages,
                tools=mcp_server._build_mcp_tools_spec(),
            )
        except Exception as exc:
            raise exc

        final_text = _extract_text_from_response(response)
        messages.append({"role": "assistant", "content": final_text})
        return final_text

    return await asyncio.to_thread(_call_openai)


def _wav_bytes_to_ulaw_chunks(wav_bytes: bytes) -> List[str]:
    if not wav_bytes:
        return []

    wav_io = io.BytesIO(wav_bytes)
    with wave.open(wav_io, "rb") as wav_in:
        framerate = wav_in.getframerate()
        sampwidth = wav_in.getsampwidth()
        channels = wav_in.getnchannels()
        frames = wav_in.readframes(wav_in.getnframes())

    if channels > 1:
        frames = audioop.tomono(frames, sampwidth, 0.5, 0.5)
    if sampwidth != 2:
        frames = audioop.lin2lin(frames, sampwidth, 2)
    if framerate != 8000:
        frames, _ = audioop.ratecv(frames, 2, 1, framerate, 8000, None)

    ulaw_audio = audioop.lin2ulaw(frames, 2)

    chunks: List[str] = []
    for i in range(0, len(ulaw_audio), TWILIO_CHUNK_SAMPLES):
        chunk = ulaw_audio[i:i + TWILIO_CHUNK_SAMPLES]
        if not chunk:
            continue
        chunks.append(base64.b64encode(chunk).decode("utf-8"))
    return chunks


async def _synthesize_speech(text: str) -> List[str]:
    if not text:
        return []

    def _call_openai():
        with mcp_server.openai_client.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=VOICE,
            input=text,
            format="wav",
        ) as response:
            audio_bytes = response.read()
        return audio_bytes

    wav_bytes = await asyncio.to_thread(_call_openai)
    return _wav_bytes_to_ulaw_chunks(wav_bytes)


async def _stream_audio_chunks(ws, stream_sid: Optional[str], chunks: List[str]):
    if not stream_sid or not chunks:
        return

    loop = asyncio.get_event_loop()
    for payload in chunks:
        media_event = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        }
        await loop.run_in_executor(None, ws.send, json.dumps(media_event))
        await asyncio.sleep(0.02)  # pace to approximate realtime

    mark_event = {
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {"name": "assistantResponse"}
    }
    await loop.run_in_executor(None, ws.send, json.dumps(mark_event))


def _decode_media_payload(payload: str) -> Optional[Dict[str, Any]]:
    try:
        media_bytes = base64.b64decode(payload)
        pcm_chunk = audioop.ulaw2lin(media_bytes, 2)
        rms = audioop.rms(pcm_chunk, 2)
        return {"pcm": pcm_chunk, "rms": rms}
    except Exception as exc:
        print(f"Failed to decode media payload: {exc}")
        return None


@voice_mcp_bp.route("/", methods=["GET"])
def index_page():
    return jsonify({"message": "Twilio Media Stream Server is running!"})


@voice_mcp_bp.route("/incoming-call", methods=["GET", "POST"])
def handle_incoming_call():
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    response.say(
        "Please wait while we connect your call to the A. I. voice assistant, powered by Twilio and the Open A I Realtime API",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    response.pause(length=1)
    response.say(
        "O.K. you can start talking!",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    render_url = Config.RENDER_EXTERNAL_URL
    if render_url:
        host = render_url.replace('https://', '').replace('http://', '').rstrip('/')
    else:
        host = request.host
    ws_url = f'wss://{host}/voicemcp/media-stream'
    connect = Connect()
    connect.stream(url=ws_url)
    response.append(connect)
    return Response(str(response), mimetype="application/xml")


def register_websocket_routes(sock, app):
    """Register WebSocket routes with the sock instance."""

    @sock.route('/voicemcp/media-stream', endpoint='voicemcp_media_stream')
    def handle_media_stream(ws):
        print("Client connected")

        loop = None
        thread = None

        def run_async_handler():
            nonlocal loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(handle_media_stream_async(ws, app))
            except Exception as e:
                print(f"Error in handle_media_stream: {e}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    if loop:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                        loop.close()
                except Exception as e:
                    print(f"Error during cleanup: {e}")

        thread = threading.Thread(target=run_async_handler, daemon=False)
        thread.start()
        thread.join()


async def handle_media_stream_async(ws, app):
    OPENAI_API_KEY = app.config.get('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        print("Error: Missing OpenAI API key")
        return

    state: Dict[str, Any] = {
        "stream_sid": None,
        "session_id": str(uuid.uuid4()),
        "buffer": bytearray(),
        "speech_active": False,
        "last_voice_ts": 0,
        "last_chunk_ts": 0,
        "segment_start_ts": 0,
        "collecting": False,
        "closed": False,
    }

    segments_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    async def process_segments():
        while True:
            segment = await segments_queue.get()
            if segment is None:
                break
            await handle_segment(segment)

    async def handle_segment(segment: bytes):
        if len(segment) < MIN_UTTERANCE_PCM_BYTES:
            return

        try:
            transcript = await _transcribe_pcm(segment)
        except Exception as exc:
            print(f"[{state['session_id']}] STT error: {exc}")
            return

        if not transcript:
            print(f"[{state['session_id']}] Empty transcript, skipping")
            return

        print(f"[{state['session_id']}] User said: {transcript}")

        try:
            assistant_reply = await _run_llm_with_tools(state["session_id"], transcript)
        except Exception as exc:
            print(f"[{state['session_id']}] LLM error: {exc}")
            assistant_reply = "Sorry, I'm having trouble reaching our assistant right now."

        print(f"[{state['session_id']}] Assistant reply: {assistant_reply}")

        try:
            chunks = await _synthesize_speech(assistant_reply)
        except Exception as exc:
            print(f"[{state['session_id']}] TTS error: {exc}")
            return

        await _stream_audio_chunks(ws, state["stream_sid"], chunks)

    processor_task = asyncio.create_task(process_segments())

    async def flush_buffer(force: bool = False):
        if not state["buffer"]:
            return
        if not force and len(state["buffer"]) < MIN_UTTERANCE_PCM_BYTES:
            state["buffer"].clear()
            state["collecting"] = False
            state["segment_start_ts"] = 0
            return

        segment = bytes(state["buffer"])
        state["buffer"].clear()
        state["collecting"] = False
        state["segment_start_ts"] = 0
        try:
            await segments_queue.put(segment)
        except asyncio.QueueClosedError:
            pass

    try:
        while True:
            message = await loop.run_in_executor(None, ws.receive)
            if message is None:
                break
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue

            event = data.get("event")
            if event == "start":
                state["stream_sid"] = data["start"]["streamSid"]
                print(f"[{state['session_id']}] Incoming stream {state['stream_sid']} started")
            elif event == "media":
                decoded = _decode_media_payload(data["media"]["payload"])
                if not decoded:
                    continue
                timestamp = int(data["media"]["timestamp"])
                if not state["collecting"]:
                    state["collecting"] = True
                    state["segment_start_ts"] = timestamp
                state["buffer"].extend(decoded["pcm"])
                state["last_chunk_ts"] = timestamp
                if decoded["rms"] > SILENCE_RMS_THRESHOLD:
                    state["speech_active"] = True
                    state["last_voice_ts"] = timestamp

                silence_gap = timestamp - state["last_voice_ts"]
                segment_age = timestamp - state["segment_start_ts"]

                should_finalize = False
                if state["speech_active"] and silence_gap >= SILENCE_DURATION_MS:
                    should_finalize = True
                    state["speech_active"] = False
                elif state["collecting"] and segment_age >= MAX_UTTERANCE_MS:
                    should_finalize = True

                if should_finalize:
                    await flush_buffer()
            elif event == "stop":
                await flush_buffer(force=True)
                break
    except Exception as exc:
        print(f"[{state['session_id']}] Error in Twilio loop: {exc}")
    finally:
        state["closed"] = True
        await segments_queue.put(None)
        await processor_task
