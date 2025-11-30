import os
import sys
import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime

# Ensure project root is importable when run as a script
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, jsonify, request, Response
from openai import OpenAI
from config import Config
from utils import SYSTEM_MESSAGE

voice_mcp_bp = Blueprint("voice_mcp", __name__)

# Session storage for conversation history
conversation_sessions: Dict[str, List[Dict[str, str]]] = {}


@voice_mcp_bp.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "success": True,
            "message": "Twilio ConversationRelay Voice Assistant is ready. Configure your Twilio number to use /voicemcp/twiml for voice calls.",
            "endpoints": {
                "twiml": "/voicemcp/twiml",
                "websocket": "/voicemcp/websocket",
                "connect_status": "/voicemcp/connect-status"
            },
            "active_sessions": len(conversation_sessions)
        }
    )


@voice_mcp_bp.route("/test", methods=["GET"])
def test_endpoint():
    """
    Test endpoint to verify the service is working
    """
    import platform
    return jsonify({
        "status": "ok",
        "message": "Voice AI MCP endpoint is working",
        "openai_configured": bool(Config.OPENAI_API_KEY),
        "python_version": platform.python_version(),
        "active_sessions": len(conversation_sessions)
    })


@voice_mcp_bp.route("/twiml", methods=["GET", "POST"])
def twiml_handler():
    """
    TwiML endpoint that Twilio calls when receiving a phone call.
    Returns TwiML with ConversationRelay configuration.
    Reference: https://www.twilio.com/docs/voice/conversationrelay/conversationrelay-noun
    """
    # Get the base URL for WebSocket connection
    # ngrok provides HTTPS, so use wss:// for WebSocket Secure
    scheme = "wss" if request.is_secure else "ws"
    host = request.host
    
    # For ngrok, always use wss even if request.is_secure is False
    # ngrok forwards HTTPS to HTTP locally
    if "ngrok" in host or "ngrok-free.app" in host:
        scheme = "wss"
    
    # WebSocket URL matches the blueprint prefix + route
    ws_url = f"{scheme}://{host}/voicemcp/websocket"
    
    print(f"[TwiML] Incoming call, WebSocket URL: {ws_url}")
    
    # Build TwiML response per Twilio docs
    # Using Google voices which are more widely compatible
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect action="/voicemcp/connect-status">
        <ConversationRelay url="{ws_url}" language="en-US" voice="Google.en-US-Neural2-A" welcomeGreeting="Hello! I'm your AI assistant. How can I help you today?" />
    </Connect>
</Response>'''
    
    print(f"[TwiML] Returned TwiML with voice: Google.en-US-Neural2-A")
    
    return Response(twiml, mimetype='text/xml')


@voice_mcp_bp.route("/connect-status", methods=["POST"])
def connect_status():
    """
    Callback endpoint for ConversationRelay connection status.
    Called when the ConversationRelay session ends.
    """
    status = request.form.get("ConversationRelayStatus", "unknown")
    call_sid = request.form.get("CallSid", "unknown")
    handoff_data = request.form.get("HandoffData", "")
    
    print(f"[Twilio] ConversationRelay ended for {call_sid}")
    print(f"[Twilio] Status: {status}")
    if handoff_data:
        print(f"[Twilio] Handoff data: {handoff_data}")
    
    # Log all form data for debugging
    print(f"[Twilio] All form data: {dict(request.form)}")
    
    # Return empty TwiML to end the call gracefully
    return Response('<?xml version="1.0" encoding="UTF-8"?><Response></Response>', mimetype='text/xml')


def register_websocket_routes(sock, app):
    """
    Register WebSocket routes for Twilio ConversationRelay.
    Twilio will connect to this WebSocket and send/receive conversation events.
    Reference: https://www.twilio.com/docs/voice/conversationrelay/websocket-messages
    """
    
    @sock.route('/voicemcp/websocket')
    def voice_websocket(ws):
        """
        WebSocket handler for Twilio ConversationRelay.
        Receives transcribed user speech and sends back AI responses.
        """
        session_id = None
        call_sid = None
        
        print("[WebSocket] New connection established")
        
        try:
            while True:
                try:
                    message = ws.receive()
                    if message is None:
                        print("[WebSocket] Received None, connection closing")
                        break
                except Exception as recv_error:
                    # Check if this is a normal close
                    if "Connection closed" in str(recv_error):
                        print(f"[WebSocket] Connection closed by Twilio: {recv_error}")
                        break
                    else:
                        print(f"[WebSocket] Receive error: {recv_error}")
                        break
                
                try:
                    data = json.loads(message)
                    event_type = data.get("type")
                    
                    print(f"[WebSocket] Received event: {event_type}")
                    print(f"[WebSocket] Full message: {json.dumps(data, indent=2)}")
                    
                    if event_type == "setup":
                        # Initial setup message from Twilio
                        # Format: https://www.twilio.com/docs/voice/conversationrelay/websocket-messages#setup-message
                        session_id = data.get("sessionId")
                        call_sid = data.get("callSid")
                        account_sid = data.get("accountSid")
                        from_number = data.get("from")
                        to_number = data.get("to")
                        direction = data.get("direction")
                        
                        print(f"[WebSocket] Setup - CallSid: {call_sid}, SessionId: {session_id}")
                        print(f"[WebSocket] Call: {from_number} -> {to_number} ({direction})")
                        
                        # Initialize conversation session
                        conversation_sessions[session_id] = [
                            {"role": "system", "content": SYSTEM_MESSAGE},
                            {"role": "system", "content": "Be concise and conversational. Responses should be natural for voice interaction. Keep answers brief (2-3 sentences max)."}
                        ]
                        
                        print(f"[WebSocket] Session initialized for {session_id}")
                        
                        # Twilio will play the welcomeGreeting automatically
                        # We don't need to send anything here, just keep the connection open
                        print(f"[WebSocket] Ready to receive prompts")
                    
                    elif event_type == "prompt":
                        # User speech has been transcribed
                        # Format: https://www.twilio.com/docs/voice/conversationrelay/websocket-messages#prompt-message
                        user_text = data.get("voicePrompt", "")
                        lang = data.get("lang", "en-US")
                        is_last = data.get("last", True)
                        
                        if not user_text:
                            print("[WebSocket] Empty voicePrompt, skipping")
                            continue
                        
                        print(f"[WebSocket] User said ({lang}): {user_text}")
                        
                        # Get or create session
                        if not session_id or session_id not in conversation_sessions:
                            session_id = call_sid or f"session-{datetime.now().timestamp()}"
                            conversation_sessions[session_id] = [
                                {"role": "system", "content": SYSTEM_MESSAGE},
                                {"role": "system", "content": "Be concise and conversational. Keep answers brief (2-3 sentences max)."}
                            ]
                            print(f"[WebSocket] Created new session: {session_id}")
                        
                        # Generate AI response
                        ai_response = generate_voice_response(session_id, user_text)
                        
                        print(f"[WebSocket] AI response: {ai_response}")
                        
                        # Send response back to Twilio for TTS
                        # Format: https://www.twilio.com/docs/voice/conversationrelay/websocket-messages#text-tokens-message
                        response_message = {
                            "type": "text",
                            "token": ai_response,
                            "last": True,
                            "interruptible": True
                        }
                        
                        ws.send(json.dumps(response_message))
                        print(f"[WebSocket] Sent text response")
                    
                    elif event_type == "dtmf":
                        # DTMF digit pressed
                        digit = data.get("digit")
                        print(f"[WebSocket] DTMF digit pressed: {digit}")
                    
                    elif event_type == "interrupt":
                        # User interrupted the AI
                        utterance = data.get("utteranceUntilInterrupt", "")
                        duration = data.get("durationUntilInterruptMs", 0)
                        print(f"[WebSocket] User interrupted after {duration}ms: '{utterance}'")
                        # Twilio handles this automatically, just log it
                    
                    elif event_type == "error":
                        # Error message from Twilio
                        error_desc = data.get("description", "Unknown error")
                        print(f"[WebSocket] Error from Twilio: {error_desc}")
                    
                    else:
                        print(f"[WebSocket] Unknown event type: {event_type}")
                        print(f"[WebSocket] Full data: {data}")
                
                except json.JSONDecodeError as e:
                    print(f"[WebSocket] JSON decode error: {e}")
                    print(f"[WebSocket] Raw message: {message}")
                    continue
                except Exception as e:
                    print(f"[WebSocket] Error processing message: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        except Exception as e:
            print(f"[WebSocket] Connection error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up session on disconnect
            if session_id and session_id in conversation_sessions:
                del conversation_sessions[session_id]
                print(f"[WebSocket] Cleaned up session {session_id}")
            print("[WebSocket] Connection closed")
    
    print("✅ WebSocket routes registered for Twilio ConversationRelay")


def generate_voice_response(session_id: str, user_text: str) -> str:
    """
    Generate an AI response for voice conversation using OpenAI.
    No MCP tools - simple chat completion optimized for voice.
    """
    if not Config.OPENAI_API_KEY:
        return "Configuration error: OpenAI API key is not set."
    
    # Get or create session messages
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "system", "content": "Be concise and conversational. Keep answers brief (2-3 sentences max)."}
        ]
    
    messages = conversation_sessions[session_id]
    
    # Add user message
    messages.append({"role": "user", "content": user_text})
    
    # Prune history to keep it manageable (keep system + last 4 turns)
    system_msgs = [m for m in messages if m.get("role") == "system"]
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    if len(convo) > 8:  # 4 user + 4 assistant
        convo = convo[-8:]
    messages = system_msgs + convo
    conversation_sessions[session_id] = messages
    
    try:
        # Use OpenAI chat completion (no tools)
        client = OpenAI(api_key=Config.OPENAI_API_KEY)
        # Use a valid model - gpt-4o-mini for fast voice responses, fallback to gpt-4o
        model = os.getenv("GPT_FAST_MODEL") or os.getenv("VOICE_MODEL") or "gpt-4o-mini"
        
        print(f"[OpenAI] Using model: {model}")
        
        # Build parameters - minimal params for compatibility
        params = {
            "model": model,
            "messages": messages,
        }
        
        # Try max_completion_tokens first (newer models), fallback to max_tokens
        try:
            params["max_completion_tokens"] = 150
            response = client.chat.completions.create(**params)
        except Exception as e:
            error_str = str(e)
            if "max_completion_tokens" in error_str or "unsupported_parameter" in error_str:
                # Older model, use max_tokens instead
                del params["max_completion_tokens"]
                params["max_tokens"] = 150
                response = client.chat.completions.create(**params)
            else:
                raise
        
        print(f"[OpenAI] Response received, choices: {len(response.choices)}")
        
        # Extract content safely
        if not response.choices:
            print("[OpenAI] No choices in response")
            return "I'm having trouble generating a response. Please try again."
        
        choice = response.choices[0]
        print(f"[OpenAI] Choice finish_reason: {choice.finish_reason}")
        
        if not choice.message:
            print("[OpenAI] No message in choice")
            return "I'm having trouble generating a response. Please try again."
        
        assistant_text = choice.message.content
        
        if not assistant_text:
            print(f"[OpenAI] Empty content. Full response: {response}")
            return "I'm having trouble generating a response. Please try again."
        
        assistant_text = assistant_text.strip()
        print(f"[OpenAI] Generated response length: {len(assistant_text)} chars")
        
        # Add assistant response to history
        messages.append({"role": "assistant", "content": assistant_text})
        
        return assistant_text
    
    except Exception as exc:
        print(f"[OpenAI error] {exc}")
        import traceback
        traceback.print_exc()
        return "I'm sorry, I encountered an error. Could you please repeat that?"


def _print_banner() -> None:
    print("")
    print("=== Twilio ConversationRelay Voice Assistant ===")
    print("Configure your Twilio number with the /voice/twiml endpoint.")
    print("The assistant will handle phone calls using Twilio's voice AI.")
    print("")


if __name__ == "__main__":
    # Print setup instructions when run directly
    _print_banner()
    print("Setup Instructions:")
    print("1. Ensure your Flask app is running with flask-sock enabled")
    print("2. Configure your Twilio phone number:")
    print("   - Go to Twilio Console > Phone Numbers")
    print("   - Select your number")
    print("   - Under 'Voice & Fax', set 'A CALL COMES IN' to:")
    print("     Webhook: https://your-domain.com/voice/twiml")
    print("     HTTP POST")
    print("3. Make sure your server is publicly accessible (use ngrok for testing)")
    print("")
    print("Example ngrok command: ngrok http 5000")
    print("")

