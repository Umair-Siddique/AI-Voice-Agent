import json
import base64
import asyncio
import threading
import websockets
from websockets.exceptions import ConnectionClosed
from flask import Blueprint, request, Response, jsonify, current_app
from twilio.twiml.voice_response import VoiceResponse, Connect
from config import Config
from utils import SYSTEM_MESSAGE
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created', 'session.updated'
]
SHOW_TIMING_MATH = False

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

def register_websocket_routes(sock, app):
    """Register WebSocket routes with the sock instance."""
    
    @sock.route('/voicemcp/media-stream')
    def handle_media_stream(ws):
        """Handle WebSocket connections between Twilio and OpenAI."""
        print("Client connected")
        
        # Create a new event loop in a separate thread for asyncio operations
        # This avoids conflicts with gevent's event loop
        loop = None
        thread = None
        
        def run_async_handler():
            nonlocal loop
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(handle_media_stream_async(ws, app))
            except Exception as e:
                print(f"Error in handle_media_stream: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up
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
        
        # Run the async handler in a separate thread
        thread = threading.Thread(target=run_async_handler, daemon=False)
        thread.start()
        thread.join()  # Wait for the connection to complete

async def handle_media_stream_async(ws, app):
    """Async handler for WebSocket connections between Twilio and OpenAI."""
    
    # Get OpenAI API key from app config
    OPENAI_API_KEY = app.config.get('OPENAI_API_KEY')
    
    if not OPENAI_API_KEY:
        print("Error: Missing OpenAI API key")
        return
    
    async with websockets.connect(
        f"wss://api.openai.com/v1/realtime?model=gpt-realtime&temperature={Config.TEMPERATURE}",
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
    ) as openai_ws:
        await initialize_session(openai_ws)

        # Connection specific state
        stream_sid = None
        latest_media_timestamp = 0
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None
        
        async def receive_from_twilio():
            """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
            nonlocal stream_sid, latest_media_timestamp, response_start_timestamp_twilio, last_assistant_item
            try:
                loop = asyncio.get_event_loop()
                while True:
                    # Run blocking ws.receive() in thread pool
                    try:
                        message = await loop.run_in_executor(None, ws.receive)
                        if message is None:
                            break
                    except Exception as e:
                        print(f"Error receiving message from Twilio: {e}")
                        break
                    
                    try:
                        data = json.loads(message)
                        if data['event'] == 'media' and openai_ws.state.name == 'OPEN':
                            latest_media_timestamp = int(data['media']['timestamp'])
                            audio_append = {
                                "type": "input_audio_buffer.append",
                                "audio": data['media']['payload']
                            }
                            await openai_ws.send(json.dumps(audio_append))
                        elif data['event'] == 'start':
                            stream_sid = data['start']['streamSid']
                            print(f"Incoming stream has started {stream_sid}")
                            response_start_timestamp_twilio = None
                            latest_media_timestamp = 0
                            last_assistant_item = None
                        elif data['event'] == 'mark':
                            if mark_queue:
                                mark_queue.pop(0)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON from Twilio: {e}")
                    except Exception as e:
                        print(f"Error processing Twilio message: {e}")
            except Exception as e:
                print(f"Client disconnected: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up OpenAI connection
                try:
                    if openai_ws.state.name == 'OPEN':
                        await openai_ws.close()
                except Exception as e:
                    print(f"Error closing OpenAI connection: {e}")

        async def send_to_twilio():
            """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
            nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio, latest_media_timestamp
            try:
                loop = asyncio.get_event_loop()
                async for openai_message in openai_ws:
                    try:
                        response = json.loads(openai_message)
                        if response['type'] in LOG_EVENT_TYPES:
                            print(f"Received event: {response['type']}", response)

                        if response.get('type') == 'response.output_audio.delta' and 'delta' in response:
                            audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
                            audio_delta = {
                                "event": "media",
                                "streamSid": stream_sid,
                                "media": {
                                    "payload": audio_payload
                                }
                            }
                            # Run blocking ws.send() in thread pool
                            try:
                                await loop.run_in_executor(None, ws.send, json.dumps(audio_delta))
                            except Exception as e:
                                print(f"Error sending audio to Twilio: {e}")
                                break

                            if response.get("item_id") and response["item_id"] != last_assistant_item:
                                response_start_timestamp_twilio = latest_media_timestamp
                                last_assistant_item = response["item_id"]
                                if SHOW_TIMING_MATH:
                                    print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

                            await send_mark(ws, stream_sid)

                        # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
                        if response.get('type') == 'input_audio_buffer.speech_started':
                            print("Speech started detected.")
                            if last_assistant_item:
                                print(f"Interrupting response with id: {last_assistant_item}")
                                await handle_speech_started_event()
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON from OpenAI: {e}")
                    except Exception as e:
                        print(f"Error processing OpenAI message: {e}")
            except ConnectionClosed as e:
                print(f"OpenAI WebSocket connection closed: {e.code} - {e.reason}")
            except Exception as e:
                print(f"Error in send_to_twilio: {e}")
                import traceback
                traceback.print_exc()

        async def handle_speech_started_event():
            """Handle interruption when the caller's speech starts."""
            nonlocal response_start_timestamp_twilio, last_assistant_item, latest_media_timestamp, stream_sid
            print("Handling speech started event.")
            try:
                if mark_queue and response_start_timestamp_twilio is not None:
                    elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
                    if SHOW_TIMING_MATH:
                        print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

                    if last_assistant_item:
                        if SHOW_TIMING_MATH:
                            print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

                        truncate_event = {
                            "type": "conversation.item.truncate",
                            "item_id": last_assistant_item,
                            "content_index": 0,
                            "audio_end_ms": elapsed_time
                        }
                        if openai_ws.state.name == 'OPEN':
                            await openai_ws.send(json.dumps(truncate_event))

                    loop = asyncio.get_event_loop()
                    try:
                        await loop.run_in_executor(None, ws.send, json.dumps({
                            "event": "clear",
                            "streamSid": stream_sid
                        }))
                    except Exception as e:
                        print(f"Error sending clear event: {e}")

                    mark_queue.clear()
                    last_assistant_item = None
                    response_start_timestamp_twilio = None
            except Exception as e:
                print(f"Error in handle_speech_started_event: {e}")
                import traceback
                traceback.print_exc()

        async def send_mark(connection, stream_sid):
            if stream_sid:
                try:
                    mark_event = {
                        "event": "mark",
                        "streamSid": stream_sid,
                        "mark": {"name": "responsePart"}
                    }
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, connection.send, json.dumps(mark_event))
                    mark_queue.append('responsePart')
                except Exception as e:
                    print(f"Error sending mark event: {e}")

        await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))


async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad"}
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE
                }
            },
            "instructions": SYSTEM_MESSAGE,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

