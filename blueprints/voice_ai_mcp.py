import json
import base64
import asyncio
import threading
import audioop
import io
import wave
from typing import List, Optional

from flask import Blueprint, request, Response, jsonify, current_app
from twilio.twiml.voice_response import VoiceResponse, Connect
from openai import OpenAI

from config import Config
from utils import SYSTEM_MESSAGE
VOICE = 'alloy'
FRAME_MS = 20
SILENCE_FRAMES = 12  # ~240ms of silence before we close a chunk
RMS_THRESHOLD = 200  # basic VAD level, tuned for μ-law 8kHz
MAX_CHUNK_FRAMES = 400  # cap user turns to ~8 seconds

voice_mcp_bp = Blueprint('voicemcp', __name__)

@voice_mcp_bp.route("/", methods=["GET"])
def index_page():
    return jsonify({"message": "Twilio Media Stream Server is running!"})

@voice_mcp_bp.route("/incoming-call", methods=["GET", "POST"])
def handle_incoming_call():
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
    response.say(
        "Please wait while we connect your call to the A. I. voice assistant, powered by Twilio and the Open A I Realtime API",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    response.pause(length=1)
    response.say(   
        "O.K. you can start talking!",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    # Get the host - Render provides RENDER_EXTERNAL_URL, fallback to request.host
    render_url = Config.RENDER_EXTERNAL_URL
    if render_url:
        # Extract hostname from full URL (e.g., https://app.onrender.com -> app.onrender.com)
        host = render_url.replace('https://', '').replace('http://', '').rstrip('/')
    else:
        host = request.host
    # Ensure wss:// protocol for WebSocket
    ws_url = f'wss://{host}/voicemcp/media-stream'
    connect = Connect()
    connect.stream(url=ws_url)
    response.append(connect)
    return Response(str(response), mimetype="application/xml")


class SpeechChunker:
    """Simple energy based VAD to break audio into short user turns."""

    def __init__(self, rms_threshold: int = RMS_THRESHOLD,
                 silence_frames: int = SILENCE_FRAMES,
                 max_frames: int = MAX_CHUNK_FRAMES):
        self.rms_threshold = rms_threshold
        self.max_silence_frames = silence_frames
        self.max_frames = max_frames
        self.buffer = bytearray()
        self.silence_frames = 0
        self.active_frames = 0
        self.is_active = False

    def process_frame(self, pcm_frame: bytes) -> Optional[bytes]:
        if not pcm_frame:
            return None

        energy = audioop.rms(pcm_frame, 2)
        if energy > self.rms_threshold:
            self.is_active = True
            self.silence_frames = 0
        elif not self.is_active:
            return None
        else:
            self.silence_frames += 1

        self.buffer.extend(pcm_frame)
        self.active_frames += 1

        if self.silence_frames >= self.max_silence_frames:
            return self._finalize_chunk()

        if self.active_frames >= self.max_frames:
            return self._finalize_chunk()

        return None

    def flush(self) -> Optional[bytes]:
        if not self.buffer:
            return None
        return self._finalize_chunk(force=True)

    def reset(self):
        self.buffer.clear()
        self.silence_frames = 0
        self.active_frames = 0
        self.is_active = False

    def _finalize_chunk(self, force: bool = False) -> Optional[bytes]:
        if not self.buffer:
            return None
        chunk = bytes(self.buffer)
        self.reset()
        if not force and len(chunk) < 3200:  # skip extremely short blips (<200ms)
            return None
        return chunk


def ensure_openai_client(app) -> OpenAI:
    client = getattr(app, "openai_client", None)
    if client:
        return client
    api_key = app.config.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OpenAI API key")
    client = OpenAI(api_key=api_key)
    app.openai_client = client
    return client


def decode_twilio_payload(payload: str) -> bytes:
    """Convert Twilio μ-law base64 audio into PCM16 bytes."""
    mulaw = base64.b64decode(payload)
    return audioop.ulaw2lin(mulaw, 2)


def pcm16_to_wav_buffer(pcm_bytes: bytes, sample_rate: int = 8000) -> io.BytesIO:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    buffer.seek(0)
    buffer.name = "twilio-chunk.wav"
    return buffer


def transcribe_audio_chunk(client: OpenAI, pcm_chunk: bytes) -> str:
    wav_buffer = pcm16_to_wav_buffer(pcm_chunk)
    transcript = client.audio.transcriptions.create(
        model=Config.STT_MODEL,
        file=wav_buffer,
        response_format="json"
    )
    text = getattr(transcript, "text", "")
    return text.strip()


def call_llm(client: OpenAI, conversation_history: List[dict]) -> str:
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, *conversation_history]
    result = client.chat.completions.create(
        model=Config.GPT_MODEL,
        temperature=Config.TEMPERATURE,
        messages=messages
    )
    message_content = result.choices[0].message.content
    if isinstance(message_content, list):
        text = " ".join(
            part.get("text", "")
            for part in message_content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = message_content or ""
    return text.strip()


def synthesize_speech_mulaw(client: OpenAI, text: str) -> bytes:
    if not text:
        return b""
    speech = client.audio.speech.create(
        model=Config.TTS_MODEL,
        voice=VOICE,
        input=text,
        format="wav",
        sample_rate=8000
    )
    wav_bytes = speech.read()
    return wav_bytes_to_mulaw(wav_bytes)


def wav_bytes_to_mulaw(wav_bytes: bytes) -> bytes:
    if not wav_bytes:
        return b""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        framerate = wav_file.getframerate()

    if channels > 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2
    if framerate != 8000:
        frames, _ = audioop.ratecv(frames, sample_width, 1, framerate, 8000, None)

    return audioop.lin2ulaw(frames, 2)


async def stream_audio_to_twilio(ws, stream_sid: Optional[str], mulaw_bytes: bytes):
    if not stream_sid or not mulaw_bytes:
        return

    frame_size = 160  # 20ms of μ-law audio at 8kHz
    loop = asyncio.get_event_loop()

    for idx in range(0, len(mulaw_bytes), frame_size):
        frame = mulaw_bytes[idx:idx + frame_size]
        if not frame:
            continue
        payload = base64.b64encode(frame).decode("utf-8")
        media_message = {
            "event": "media",
            "streamSid": stream_sid,
            "media": {"payload": payload}
        }
        try:
            await loop.run_in_executor(None, ws.send, json.dumps(media_message))
        except Exception as exc:
            print(f"Error sending audio frame to Twilio: {exc}")
            break
        await asyncio.sleep(FRAME_MS / 1000)

    mark_event = {
        "event": "mark",
        "streamSid": stream_sid,
        "mark": {"name": "assistantComplete"}
    }
    try:
        await loop.run_in_executor(None, ws.send, json.dumps(mark_event))
    except Exception as exc:
        print(f"Error sending mark event: {exc}")

def register_websocket_routes(sock, app):
    """Register WebSocket routes with the sock instance."""

    @sock.route('/voicemcp/media-stream', endpoint='voicemcp_media_stream')
    def handle_media_stream(ws):
        """Bridge a Twilio media stream to STT -> GPT -> TTS."""
        print("Client connected to /voicemcp/media-stream")

        loop = None

        def run_async_handler():
            nonlocal loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(handle_media_stream_async(ws, app))
            except Exception as exc:
                print(f"Error in handle_media_stream_async: {exc}")
                import traceback
                traceback.print_exc()
            finally:
                try:
                    if loop:
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()
                        if pending:
                            loop.run_until_complete(
                                asyncio.gather(*pending, return_exceptions=True)
                            )
                        loop.close()
                except Exception as cleanup_exc:
                    print(f"Error during websocket cleanup: {cleanup_exc}")

        thread = threading.Thread(target=run_async_handler, daemon=False)
        thread.start()
        thread.join()


async def handle_media_stream_async(ws, app):
    """Async bridge that performs STT -> GPT -> TTS for each user turn."""
    try:
        client = ensure_openai_client(app)
    except Exception as exc:
        print(f"Unable to initialize OpenAI client: {exc}")
        return

    stream_sid = None
    loop = asyncio.get_event_loop()
    chunker = SpeechChunker()
    audio_queue: asyncio.Queue = asyncio.Queue()
    conversation_history: List[dict] = []

    async def receive_from_twilio():
        nonlocal stream_sid
        try:
            while True:
                message = await loop.run_in_executor(None, ws.receive)
                if message is None:
                    break
                data = json.loads(message)
                event_type = data.get("event")

                if event_type == "start":
                    stream_sid = data["start"]["streamSid"]
                    chunker.reset()
                    continue

                if event_type == "media":
                    payload = data.get("media", {}).get("payload")
                    if not payload:
                        continue
                    pcm_frame = decode_twilio_payload(payload)
                    chunk = chunker.process_frame(pcm_frame)
                    if chunk:
                        await audio_queue.put(chunk)
                    continue

                if event_type == "stop":
                    flush_chunk = chunker.flush()
                    if flush_chunk:
                        await audio_queue.put(flush_chunk)
                    break
        except Exception as exc:
            print(f"Error receiving stream from Twilio: {exc}")
        finally:
            await audio_queue.put(None)

    async def process_turns():
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                break

            try:
                transcript = await loop.run_in_executor(
                    None, transcribe_audio_chunk, client, chunk
                )
            except Exception as exc:
                print(f"Transcription failed: {exc}")
                continue

            if not transcript:
                continue

            conversation_history.append({"role": "user", "content": transcript})

            try:
                assistant_text = await loop.run_in_executor(
                    None, call_llm, client, conversation_history
                )
            except Exception as exc:
                print(f"LLM call failed: {exc}")
                conversation_history.pop()  # remove user turn to retry later
                continue

            if not assistant_text:
                continue

            conversation_history.append({"role": "assistant", "content": assistant_text})

            try:
                mulaw_bytes = await loop.run_in_executor(
                    None, synthesize_speech_mulaw, client, assistant_text
                )
            except Exception as exc:
                print(f"TTS synthesis failed: {exc}")
                continue

            await stream_audio_to_twilio(ws, stream_sid, mulaw_bytes)

    await asyncio.gather(receive_from_twilio(), process_turns())

