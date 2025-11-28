import json
import base64
import asyncio
import threading
import io
import wave
import struct
import time
from typing import List, Optional

import numpy as np
import requests
from urllib.parse import urlparse

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
STT_ENDPOINT = "https://api.openai.com/v1/audio/transcriptions"
STT_MAX_RETRIES = 3

voice_mcp_bp = Blueprint('voicemcp', __name__)

@voice_mcp_bp.route("/", methods=["GET"])
def index_page():
    return jsonify({"message": "Twilio Media Stream Server is running!"})

@voice_mcp_bp.route("/incoming-call", methods=["GET", "POST"])
def handle_incoming_call():
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    def resolve_public_host() -> str:
        # Highest priority: explicit config
        if Config.RENDER_EXTERNAL_URL:
            parsed = urlparse(Config.RENDER_EXTERNAL_URL)
            if parsed.hostname:
                return parsed.hostname
            return Config.RENDER_EXTERNAL_URL.replace('https://', '').replace('http://', '').rstrip('/')
        # Next: forwarded host set by Render's proxy
        xf_host = request.headers.get('X-Forwarded-Host')
        if xf_host:
            # X-Forwarded-Host can be a CSV; take first
            return xf_host.split(',')[0].strip().split(':')[0]
        # Fallback: Host header or request.host (strip port)
        host = request.headers.get('Host', request.host or '')
        return host.split(':')[0]

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
    # Compute public host for Twilio to connect back via WSS
    host = resolve_public_host()
    ws_url = f'wss://{host}/voicemcp/media-stream'
    print(f"[voicemcp] WS URL for Twilio Connect: {ws_url} | XFH={request.headers.get('X-Forwarded-Host')} Host={request.headers.get('Host')}")
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

        energy = frame_rms(pcm_frame)
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


def frame_rms(pcm_frame: bytes) -> float:
    if not pcm_frame:
        return 0.0
    samples = np.frombuffer(pcm_frame, dtype=np.int16)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


_MULAW_BIAS = 0x84
_MULAW_CLIP = 32635


def _mulaw_byte_to_pcm(byte_value: int) -> int:
    byte_value = ~byte_value & 0xFF
    sign = byte_value & 0x80
    exponent = (byte_value >> 4) & 0x07
    mantissa = byte_value & 0x0F
    sample = ((mantissa << 3) + _MULAW_BIAS) << exponent
    sample -= _MULAW_BIAS
    return -sample if sign else sample


def mulaw_to_linear_bytes(mulaw_bytes: bytes) -> bytes:
    if not mulaw_bytes:
        return b""
    pcm = bytearray(len(mulaw_bytes) * 2)
    for idx, byte_value in enumerate(mulaw_bytes):
        sample = _mulaw_byte_to_pcm(byte_value)
        struct.pack_into("<h", pcm, idx * 2, sample)
    return bytes(pcm)


def _linear_sample_to_mulaw(sample: int) -> int:
    sample = max(-32768, min(32767, sample))
    sign = 0 if sample >= 0 else 0x80
    if sign:
        sample = -sample
    sample += _MULAW_BIAS
    if sample > _MULAW_CLIP:
        sample = _MULAW_CLIP

    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (sample & mask):
        mask >>= 1
        exponent -= 1

    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulaw_byte


def linear_pcm_to_mulaw_bytes(samples: np.ndarray) -> bytes:
    if samples.size == 0:
        return b""
    samples = samples.astype(np.int16, copy=False)
    encoded = bytearray(samples.size)
    for idx, sample in enumerate(samples):
        encoded[idx] = _linear_sample_to_mulaw(int(sample))
    return bytes(encoded)


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
    return mulaw_to_linear_bytes(mulaw)


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


def transcribe_audio_chunk(api_key: str, pcm_chunk: bytes) -> str:
    wav_buffer = pcm16_to_wav_buffer(pcm_chunk)
    wav_bytes = wav_buffer.getvalue()
    if not wav_bytes:
        return ""

    files = {"file": ("twilio-chunk.wav", wav_bytes, "audio/wav")}
    data = {"model": Config.STT_MODEL, "response_format": "json"}
    headers = {"Authorization": f"Bearer {api_key}"}
    last_error = None

    for attempt in range(1, STT_MAX_RETRIES + 1):
        try:
            response = requests.post(
                STT_ENDPOINT,
                headers=headers,
                data=data,
                files=files,
                timeout=60
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("text", "").strip()
        except requests.exceptions.RequestException as exc:
            status = getattr(exc.response, "status_code", "no-status")
            body = exc.response.text[:500] if getattr(exc, "response", None) else ""
            print(
                f"STT request attempt {attempt}/{STT_MAX_RETRIES} failed "
                f"status={status}: {exc}. body='{body}'"
            )
            last_error = exc
            if attempt < STT_MAX_RETRIES:
                time.sleep(0.5 * attempt)
            else:
                raise

    raise RuntimeError(f"STT failed after {STT_MAX_RETRIES} retries: {last_error}")


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
    try:
        # Request raw PCM at 8k to avoid WAV header parsing and resampling
        speech = client.audio.speech.create(
            model=Config.TTS_MODEL,
            voice=VOICE,
            input=text,
            format="pcm",
            sample_rate=8000
        )
        pcm_bytes = speech.read()
        samples = np.frombuffer(pcm_bytes, dtype=np.int16)
        return linear_pcm_to_mulaw_bytes(samples)
    except Exception as exc:
        print(f"TTS (pcm) failed, falling back to wav: {exc}")
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

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sample_width)
    if dtype is None:
        raise ValueError("Unsupported sample width returned from TTS response.")

    samples = np.frombuffer(frames, dtype=dtype)

    if sample_width == 1:
        samples = (samples.astype(np.int16) - 128) << 8
    elif sample_width == 2:
        samples = samples.astype(np.int16, copy=False)
    else:  # 32-bit
        samples = (samples >> 16).astype(np.int16)

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)

    if framerate != 8000 and samples.size > 0:
        duration = samples.size / framerate
        target_len = max(1, int(duration * 8000))
        x_old = np.linspace(0, duration, samples.size, endpoint=False)
        x_new = np.linspace(0, duration, target_len, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.int16)

    return linear_pcm_to_mulaw_bytes(samples)


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
    api_key = app.config.get("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in app config.")
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
                    print(f"[voicemcp:{stream_sid}] stream started")
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
                    print(f"[voicemcp:{stream_sid}] stream stop received, flushing")
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
                    None, transcribe_audio_chunk, api_key, chunk
                )
            except Exception as exc:
                print(f"STT error: {exc}")
                continue

            if not transcript:
                # Quiet chunk, ignore
                continue

            conversation_history.append({"role": "user", "content": transcript})
            print(f"[voicemcp:{stream_sid}] user: {transcript}")

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
            print(f"[voicemcp:{stream_sid}] assistant: {assistant_text}")

            try:
                mulaw_bytes = await loop.run_in_executor(
                    None, synthesize_speech_mulaw, client, assistant_text
                )
            except Exception as exc:
                print(f"TTS synthesis failed: {exc}")
                continue

            await stream_audio_to_twilio(ws, stream_sid, mulaw_bytes)

    await asyncio.gather(receive_from_twilio(), process_turns())

