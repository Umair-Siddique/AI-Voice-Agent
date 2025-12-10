import os
import sys
import json
import asyncio
import re
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

# Ensure project root is importable when run as a script
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Blueprint, jsonify, request, Response
from openai import OpenAI
from fastmcp import Client as FastMCPClient
from config import Config
from utils import SYSTEM_MESSAGE

voice_mcp_bp = Blueprint("voice_mcp", __name__)

# MCP Server Configuration
SIMPLYBOOK_MCP_URL = os.getenv("SIMPLYBOOK_MCP_URL", "https://simplybook-mcp-server.onrender.com/sse")
SIMPLYBOOK_MCP_LABEL = os.getenv("SIMPLYBOOK_MCP_LABEL", "simplybook")
SIMPLYBOOK_MCP_HEADERS_JSON = os.getenv("SIMPLYBOOK_MCP_HEADERS_JSON", "").strip()
# For voice assistant, we want automatic tool calling without approval prompts
# Set to "never" (default) for automatic execution, or "always" to require manual approval
SIMPLYBOOK_MCP_REQUIRE_APPROVAL = os.getenv("SIMPLYBOOK_MCP_REQUIRE_APPROVAL", "never").strip()
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5")  # Using gpt-5 for better accuracy
# Max output tokens needs to be high enough for the agentic loop:
# - Tool discovery (mcp_list_tools)
# - Reasoning tokens (internal thinking)
# - Multiple tool calls
# - Final response generation
# 600 is too low for complex workflows, increasing to 2000
VOICE_MAX_OUTPUT_TOKENS = int(os.getenv("VOICE_MAX_OUTPUT_TOKENS", "2000"))
# Max tool calls to prevent infinite loops
VOICE_MAX_TOOL_CALLS = int(os.getenv("VOICE_MAX_TOOL_CALLS", "15"))
# Cooldown after sending a final answer to avoid immediately processing
# trailing transcript fragments as new prompts (seconds)
VOICE_FINAL_COOLDOWN_SECONDS = float(os.getenv("VOICE_FINAL_COOLDOWN_SECONDS", "1.0"))

# Twilio ConversationRelay Configuration for better speech recognition
TWILIO_LANGUAGE = os.getenv("TWILIO_LANGUAGE", "en-US")
TWILIO_VOICE = os.getenv("TWILIO_VOICE", "Google.en-US-Neural2-A")
TWILIO_TRANSCRIPTION_PROVIDER = os.getenv("TWILIO_TRANSCRIPTION_PROVIDER", "deepgram")  # deepgram or google
TWILIO_DTMF_DETECTION = os.getenv("TWILIO_DTMF_DETECTION", "true").lower()
TWILIO_INTERRUPTIBLE = os.getenv("TWILIO_INTERRUPTIBLE", "true").lower()
TWILIO_PROFANITY_FILTER = os.getenv("TWILIO_PROFANITY_FILTER", "false").lower()
TWILIO_WELCOME_GREETING = os.getenv("TWILIO_WELCOME_GREETING", "Hello! I'm your AI assistant. How can I help you today?")


def _build_mcp_tools_spec() -> List[Dict[str, Any]]:
    """
    Build the Responses API 'tools' specification for a remote MCP server.

    This follows the official OpenAI documentation:
    https://platform.openai.com/docs/guides/tools-connectors-mcp
    
    When passed to client.responses.create(tools=...), the OpenAI API will:
    1. Automatically discover tools from the MCP server
    2. Decide which tools to call based on the conversation
    3. Execute the tool calls on the MCP server
    4. Process results and generate a final response
    
    No manual tool calling loop is needed - the API handles everything automatically.
    """
    tool: Dict[str, Any] = {
        "type": "mcp",
        "server_label": SIMPLYBOOK_MCP_LABEL,
        "server_url": SIMPLYBOOK_MCP_URL,
    }

    # Optional headers (parsed from JSON) if provided
    if SIMPLYBOOK_MCP_HEADERS_JSON:
        try:
            headers = json.loads(SIMPLYBOOK_MCP_HEADERS_JSON)
            if isinstance(headers, dict) and headers:
                tool["headers"] = headers
        except json.JSONDecodeError:
            print("⚠️  WARNING: SIMPLYBOOK_MCP_HEADERS_JSON is not valid JSON; ignoring.")

    # Optional approval policy
    # For voice assistant, we auto-approve tools to avoid interrupting the conversation
    # Set to "never" to allow automatic tool execution
    # Set to "always" only if you want to manually approve each tool call
    if SIMPLYBOOK_MCP_REQUIRE_APPROVAL and SIMPLYBOOK_MCP_REQUIRE_APPROVAL != "never":
        tool["require_approval"] = SIMPLYBOOK_MCP_REQUIRE_APPROVAL
    else:
        # Default to "never" for voice to enable automatic tool calling
        tool["require_approval"] = "never"

    return [tool]


# Session storage for conversation history
# Using OpenAI Responses API with automatic MCP tool calling (no manual ReAct loop needed)
conversation_sessions: Dict[str, List[Dict[str, str]]] = {}
# Per-session prompt buffers to avoid sending partial transcripts
prompt_buffers: Dict[str, Dict[str, Any]] = {}


# ============================================================================
# MCP Helper Functions (for debugging and compatibility)
# Note: These are no longer needed for the main voice response flow since
# OpenAI's Responses API handles MCP tool calling automatically. 
# Kept here for backwards compatibility and debugging purposes.
# ============================================================================

def _safe_to_string(value: Any) -> str:
    """Convert any value to a printable string."""
    try:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    except Exception:
        try:
            return repr(value)
        except Exception:
            return "<unserializable>"


async def _mcp_call(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Call a tool on the FastMCP server and return a printable string result.
    """
    mcp_url = SIMPLYBOOK_MCP_URL.rstrip("/")
    
    try:
        client = FastMCPClient(mcp_url)
        async with client:
            # Wait a moment for SSE connection to establish
            await asyncio.sleep(0.5)
            result = await client.call_tool(tool_name, arguments or {})
            # FastMCP result typically exposes .data; fall back to dict if needed
            data = getattr(result, "data", result)
            return _safe_to_string(data)
    except Exception as e:
        raise Exception(f"MCP call failed for '{tool_name}': {str(e)}")


async def _mcp_tools_brief(limit: int = 24) -> str:
    """
    Fetch tools from MCP and return a brief markdown list with names and summaries.
    """
    mcp_url = SIMPLYBOOK_MCP_URL.rstrip("/")
    client = FastMCPClient(mcp_url)
    lines: List[str] = []
    async with client:
        await asyncio.sleep(0.5)
        tools = await client.list_tools()
        for idx, t in enumerate(tools):
            if idx >= limit:
                lines.append(f"- ... and {len(tools) - limit} more")
                break
            name = getattr(t, "name", "unknown_tool")
            desc = getattr(t, "description", "") or ""
            lines.append(f"- {name}: {desc}")
    return "\n".join(lines) if lines else "- (no tools discovered)"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from text and parse it.
    Accepts plain JSON or fenced ```json blocks.
    """
    if not text:
        return None

    stripped = text.strip()

    # Try direct parse first
    try:
        return json.loads(stripped)
    except Exception:
        pass

    # Look for fenced code block ```json ... ```
    fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # Try to locate the first balanced JSON object within the text
    candidate = _find_first_json_block(stripped)
    if candidate:
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Fallback: find first {...} greedily (legacy behavior)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


def _is_continuation(prev: str, current: str) -> bool:
    """
    Heuristic to detect if `current` is just a continuation/duplication of the
    previous final user text (e.g., trailing fragment after we already sent a response).
    """
    if not prev or not current:
        return False

    prev_norm = prev.strip().lower()
    curr_norm = current.strip().lower()

    # Direct prefix/suffix overlap
    if curr_norm.startswith(prev_norm) or prev_norm.startswith(curr_norm):
        return True

    prev_tokens = prev_norm.split()
    curr_tokens = curr_norm.split()
    if not prev_tokens or not curr_tokens:
        return False

    overlap = len(set(prev_tokens) & set(curr_tokens)) / min(len(prev_tokens), len(curr_tokens))
    return overlap >= 0.7


def _find_first_json_block(text: str) -> Optional[str]:
    """
    Scan text and return the first substring that forms a balanced JSON object.
    Helpful when the model emits multiple JSON objects back-to-back.
    """
    start_idx: Optional[int] = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(text):
        if start_idx is None:
            if ch == "{":
                start_idx = idx
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0 and start_idx is not None:
                return text[start_idx : idx + 1]

    return None


async def _fetch_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the input schema for a given tool name by querying the MCP server.
    """
    mcp_url = SIMPLYBOOK_MCP_URL.rstrip("/")
    client = FastMCPClient(mcp_url)
    async with client:
        await asyncio.sleep(0.3)
        tools = await client.list_tools()
        for t in tools:
            if getattr(t, "name", None) == tool_name:
                # Prefer inputSchema, else schema
                schema = getattr(t, "inputSchema", None) or getattr(t, "schema", None)
                return schema if isinstance(schema, dict) else None
    return None


# ============================================================================
# Flask Routes
# ============================================================================

@voice_mcp_bp.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "success": True,
            "message": "Twilio ConversationRelay Voice Assistant with OpenAI MCP integration is ready. Configure your Twilio number to use /voicemcp/twiml for voice calls.",
            "endpoints": {
                "twiml": "/voicemcp/twiml",
                "websocket": "/voicemcp/websocket",
                "connect_status": "/voicemcp/connect-status"
            },
            "features": {
                "mcp_integration": "OpenAI Responses API with automatic tool calling",
                "mcp_server": SIMPLYBOOK_MCP_URL,
                "model": AGENT_MODEL,
                "documentation": "https://platform.openai.com/docs/guides/tools-connectors-mcp"
            },
            "active_sessions": len(conversation_sessions)
        }
    )


@voice_mcp_bp.route("/test", methods=["GET"])
def test_endpoint():
    """
    Test endpoint to verify the service is working with OpenAI MCP integration
    """
    import platform
    return jsonify({
        "status": "ok",
        "message": "Voice AI with OpenAI MCP integration is working",
        "openai_configured": bool(Config.OPENAI_API_KEY),
        "mcp_configured": bool(SIMPLYBOOK_MCP_URL),
        "mcp_server_url": SIMPLYBOOK_MCP_URL,
        "model": AGENT_MODEL,
        "max_output_tokens": VOICE_MAX_OUTPUT_TOKENS,
        "max_tool_calls": VOICE_MAX_TOOL_CALLS,
        "require_approval": SIMPLYBOOK_MCP_REQUIRE_APPROVAL,
        "implementation": "OpenAI Responses API with automatic MCP tool calling + incomplete response handling",
        "twilio_config": {
            "language": TWILIO_LANGUAGE,
            "voice": TWILIO_VOICE,
            "transcription_provider": TWILIO_TRANSCRIPTION_PROVIDER,
            "dtmf_detection": TWILIO_DTMF_DETECTION,
            "interruptible": TWILIO_INTERRUPTIBLE,
            "profanity_filter": TWILIO_PROFANITY_FILTER
        },
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
    
    # Build TwiML response per Twilio docs with enhanced speech recognition settings
    # Enhanced configuration for better speech recognition:
    # - transcriptionProvider: Deepgram provides better accuracy for conversational speech
    # - dtmfDetection: Enable DTMF digit detection
    # - interruptible: Allow caller to interrupt AI responses
    # - profanityFilter: false to avoid censoring legitimate words
    twiml = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect action="/voicemcp/connect-status">
        <ConversationRelay 
            url="{ws_url}" 
            language="{TWILIO_LANGUAGE}" 
            voice="{TWILIO_VOICE}" 
            transcriptionProvider="{TWILIO_TRANSCRIPTION_PROVIDER}"
            dtmfDetection="{TWILIO_DTMF_DETECTION}"
            interruptible="{TWILIO_INTERRUPTIBLE}"
            profanityFilter="{TWILIO_PROFANITY_FILTER}"
            welcomeGreeting="{TWILIO_WELCOME_GREETING}" />
    </Connect>
</Response>'''
    
    print(f"[TwiML] Returned TwiML with voice: {TWILIO_VOICE}, transcription: {TWILIO_TRANSCRIPTION_PROVIDER}")
    
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
                        # Initialize prompt buffer state
                        prompt_buffers[session_id] = {"buffer": "", "last_final": "", "last_received": time.time()}
                        
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
                        
                        if not user_text or not user_text.strip():
                            print("[WebSocket] Empty voicePrompt, skipping")
                            continue
                        
                        # Log transcription details for debugging
                        print(f"[WebSocket] Transcribed speech ({lang}): '{user_text}'")
                        print(f"[WebSocket] Is final utterance: {is_last}")
                        
                        # Get or create session
                        if not session_id or session_id not in conversation_sessions:
                            session_id = call_sid or f"session-{datetime.now().timestamp()}"
                            conversation_sessions[session_id] = [
                                {"role": "system", "content": SYSTEM_MESSAGE},
                                {"role": "system", "content": "Be concise and conversational. Keep answers brief (2-3 sentences max)."}
                            ]
                            prompt_buffers[session_id] = {"buffer": "", "last_final": "", "last_received": time.time()}
                            print(f"[WebSocket] Created new session: {session_id}")
                        
                        # Strip and clean the user text before processing
                        user_text = user_text.strip()

                        # Prepare session buffer
                        buf_state = prompt_buffers.setdefault(
                            session_id,
                            {"buffer": "", "last_final": "", "last_received": 0.0},
                        )
                        buf_state["last_received"] = time.time()

                        # Merge incremental transcripts; prefer longest to avoid duplicates
                        current_buf = buf_state.get("buffer", "")
                        if current_buf and user_text.startswith(current_buf):
                            merged_text = user_text
                        elif current_buf and current_buf.startswith(user_text):
                            merged_text = current_buf
                        else:
                            merged_text = f"{current_buf} {user_text}".strip() if current_buf else user_text

                        buf_state["buffer"] = merged_text

                        # Only process when Twilio marks the utterance as complete
                        if not is_last:
                            print("[WebSocket] Partial transcript stored, waiting for end of utterance...")
                            continue

                        final_text = buf_state["buffer"].strip() or user_text
                        if not final_text:
                            print("[WebSocket] Final transcript empty after buffering, skipping")
                            buf_state["buffer"] = ""
                            continue

                        now = time.time()
                        last_final_time = buf_state.get("last_final_time", 0.0)
                        if last_final_time and (now - last_final_time) < VOICE_FINAL_COOLDOWN_SECONDS:
                            print("[WebSocket] Within cooldown window after previous response; skipping to avoid trailing fragments.")
                            buf_state["buffer"] = ""
                            continue

                        # If Twilio resumes the same utterance after we already answered, drop it
                        last_final = buf_state.get("last_final", "")
                        if last_final and _is_continuation(last_final, final_text):
                            print("[WebSocket] Detected continuation/duplicate of previous utterance; ignoring to avoid duplicate sends.")
                            buf_state["buffer"] = ""
                            continue

                        print(f"[WebSocket] Processing complete question: '{final_text}'")
                        
                        # Mark the time we are processing to avoid immediate follow-up fragments
                        buf_state["last_final_time"] = now

                        # Check if tools are needed before generating response
                        # Get current conversation history (without the new user message yet)
                        current_history = conversation_sessions.get(session_id, [])
                        needs_tools = _check_if_tools_needed(final_text, current_history)
                        
                        # If tools are needed, send immediate acknowledgment
                        if needs_tools:
                            acknowledgment = "Please give me a few minutes, let me check."
                            print(f"[WebSocket] Tools needed - sending immediate acknowledgment: {acknowledgment}")
                            
                            # Send acknowledgment immediately
                            ack_message = {
                                "type": "text",
                                "token": acknowledgment,
                                "last": True,
                                "interruptible": True
                            }
                            ws.send(json.dumps(ack_message))
                            print(f"[WebSocket] Sent immediate acknowledgment to Twilio")

                        # Generate AI response (this will include tool calls if needed)
                        ai_response = generate_voice_response(session_id, final_text)
                        buf_state["last_final"] = final_text
                        buf_state["last_final_time"] = time.time()
                        buf_state["buffer"] = ""
                        
                        print(f"[WebSocket] Generated AI response: {ai_response}")
                        
                        # Send final response back to Twilio for TTS
                        # Format: https://www.twilio.com/docs/voice/conversationrelay/websocket-messages#text-tokens-message
                        response_message = {
                            "type": "text",
                            "token": ai_response,
                            "last": True,
                            "interruptible": True
                        }
                        
                        ws.send(json.dumps(response_message))
                        print(f"[WebSocket] Sent final text response to Twilio for TTS")
                    
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
            if session_id and session_id in prompt_buffers:
                del prompt_buffers[session_id]
            print("[WebSocket] Connection closed")
    
    print("✅ WebSocket routes registered for Twilio ConversationRelay")


def _check_if_tools_needed(user_text: str, conversation_history: List[Dict[str, str]]) -> bool:
    """
    Analyze the user query using OpenAI to determine if SimplyBook MCP tools are needed.
    Uses recent conversation context to avoid asking for time when the user is just chatting.
    Returns True if tools are likely needed, False otherwise.
    """
    if not Config.OPENAI_API_KEY:
        return False
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # Include a short slice of recent conversation so the model can understand context.
    # Limit total characters to keep the prompt compact for the fast model.
    recent_msgs = [
        m for m in conversation_history[-8:]  # last ~4 turns (user + assistant)
        if m.get("role") in ("user", "assistant")
    ]
    history_lines: List[str] = []
    total_chars = 0
    for msg in recent_msgs:
        role = msg.get("role", "unknown")
        content = (msg.get("content", "") or "").strip()
        if not content:
            continue
        line = f"{role}: {content}"
        total_chars += len(line)
        history_lines.append(line)
        if total_chars >= 1500:  # soft cap to avoid long prompts
            break
    history_block = "\n".join(history_lines) if history_lines else "(no prior turns)"
    
    # Build a focused analysis prompt
    analysis_prompt = (
        "Analyze the following user query and determine if it requires accessing SimplyBook booking system tools.\n\n"
        "Conversation so far:\n"
        f"{history_block}\n\n"
        "SimplyBook tools are needed for:\n"
        "- Getting services list (get_services)\n"
        "- Checking availability (get_available_slots)\n"
        "- Getting providers (get_providers)\n"
        "- Creating bookings (create_booking)\n"
        "- Managing clients (get_clients_list, create_client)\n"
        "- Updating existing appointments (rescheduling) or cancelling bookings\n"
        "- Any query about appointments, bookings, schedules, availability, services, or providers\n\n"
        "Simple conversational queries that don't need booking system data do NOT require tools.\n\n"
        "User query: " + user_text + "\n\n"
        "Respond with ONLY 'YES' if tools are needed, or 'NO' if not needed. No other text."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use a fast, cheap model for analysis
            messages=[
                {"role": "system", "content": "You are a tool usage analyzer. Respond with only YES or NO."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=10,
            temperature=0
        )
        
        answer = response.choices[0].message.content.strip().upper()
        needs_tools = answer == "YES"
        
        print(f"🔍 [Voice] Tool analysis: Query needs tools = {needs_tools}")
        return needs_tools
        
    except Exception as e:
        print(f"⚠️  [Voice] Error analyzing tool need: {e}")
        # On error, assume tools might be needed to be safe
        return True


def generate_voice_response(session_id: str, user_text: str) -> str:
    """
    Generate an AI response for voice conversation using OpenAI's official MCP implementation.
    Following: https://platform.openai.com/docs/guides/tools-connectors-mcp
    The Responses API handles MCP tool calling automatically when tools are provided.
    Optimized for voice: brief responses, conversational tone.
    """
    if not Config.OPENAI_API_KEY:
        return "Configuration error: OpenAI API key is not set."
    
    # Initialize conversation if new session
    if session_id not in conversation_sessions:
        print(f"")
        print(f"{'='*80}")
        print(f"🆕 [Voice] Initializing new session: {session_id}")
        print(f"{'='*80}")
        
        # Voice-optimized system message
        system_prompt = (
            "You are a helpful AI assistant for a booking system with access to SimplyBook MCP tools. "
            "Be conversational and keep responses brief (2-3 sentences max) for voice interaction.\n\n"
            "IMPORTANT - Tool Usage:\n"
            "- ALWAYS use tools to get real-time data - NEVER make assumptions or guess\n"
            "- When asked about services: use get_services tool\n"
            "- When asked about availability: use get_available_slots tool with service_id, provider_id, and date\n"
            "- When booking: use get_providers, then get_available_slots to confirm time exists, then create_booking\n"
            "- For client info: use get_clients_list or create_client if needed\n\n"
            "Scheduling safety:\n"
            "- ALWAYS verify availability with get_available_slots before creating bookings\n"
            "- Never say 'no availability' without actually checking the tool\n"
            "- If multiple providers exist, ask user to choose or pick the first one\n"
            "- Confirm all booking details (date, time, service, client) before creating\n\n"
            "Remember: You have tools - use them! Don't guess or make up information."
        )
        conversation_sessions[session_id] = [{"role": "system", "content": system_prompt}]
    
    # Append user message
    print(f"")
    print(f"{'='*80}")
    print(f"👤 [Voice] User: {user_text}")
    print(f"{'='*80}")
    print(f"")
    
    conversation_sessions[session_id].append({"role": "user", "content": user_text})
    messages = conversation_sessions[session_id]
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # Build MCP tools spec following OpenAI documentation
    # This tells the Responses API to automatically discover and use MCP server tools
    mcp_tools = _build_mcp_tools_spec()
    
    try:
        print(f"🔧 [Voice] Calling Responses API with MCP tools...")
        
        # Use OpenAI's official Responses API with MCP tools
        # The API will automatically:
        # 1. Discover tools from the MCP server
        # 2. Decide which tools to call based on the conversation
        # 3. Execute the tools
        # 4. Process the results
        # 5. Generate a final response
        response = client.responses.create(
            model=AGENT_MODEL,
            input=messages,
            tools=mcp_tools,  # This enables automatic MCP tool calling
            max_output_tokens=VOICE_MAX_OUTPUT_TOKENS,  # Increased to 2000 for complex workflows
            max_tool_calls=VOICE_MAX_TOOL_CALLS,  # Limit tool calls to prevent infinite loops
        )
        
        # Extract the final response text
        final_text = ""
        approval_needed = False
        response_status = getattr(response, 'status', 'unknown')
        
        # Helper function to count tool calls
        def count_tool_calls(response_obj):
            """Count the number of tool calls in a response."""
            count = 0
            outputs = getattr(response_obj, 'output', [])
            for output in outputs:
                output_type = getattr(output, 'type', None)
                if output_type in ("mcp_call_tool", "mcp_call"):
                    count += 1
            return count
        
        # Log initial tool calls
        initial_tool_calls = count_tool_calls(response)
        if initial_tool_calls > 0:
            print(f"📊 [Voice] Initial response tool calls: {initial_tool_calls}/{VOICE_MAX_TOOL_CALLS}")
        
        try:
            # Check if there are any approval requests blocking the response
            outputs = getattr(response, "output", [])
            for output in outputs:
                output_type = getattr(output, "type", None)
                
                # Check for approval request
                if output_type == "mcp_approval_request":
                    approval_needed = True
                    tool_name = getattr(output, "name", "unknown")
                    print(f"⚠️  [Voice] Approval required for tool: {tool_name}")
                    print(f"⚠️  [Voice] This should not happen in voice mode. Check SIMPLYBOOK_MCP_REQUIRE_APPROVAL config.")
                    final_text = "I need to access some information, but I'm waiting for approval. Please check your configuration."
                    break
            
            # If no approval needed, extract the actual message text
            if not approval_needed:
                # Try to get output_text first (GPT-4 style)
                output_text = getattr(response, "output_text", None)
                if isinstance(output_text, str) and output_text.strip():
                    final_text = output_text.strip()
                    print(f"✅ [Voice] Extracted from output_text")
                
                # If output_text not available, try to extract from output array
                if not final_text:
                    for output in outputs:
                        output_type = getattr(output, "type", None)
                        
                        # Look for message type output (the final assistant message)
                        if output_type == "message":
                            content = getattr(output, "content", [])
                            for content_item in content:
                                if hasattr(content_item, "type") and content_item.type == "text":
                                    text = getattr(content_item, "text", "")
                                    if text and text.strip():
                                        final_text = text.strip()
                                        print(f"✅ [Voice] Extracted from message.content.text")
                                        break
                            if final_text:
                                break
                        
                        # Check for text in tool call results (could be mcp_call_tool or mcp_call)
                        elif output_type in ("mcp_call_tool", "mcp_call"):
                            if not final_text:  # Only use if no message found yet
                                content = getattr(output, "content", [])
                                for content_item in content:
                                    if hasattr(content_item, "type") and content_item.type == "text":
                                        text = getattr(content_item, "text", "")
                                        if text and text.strip():
                                            # This is tool output, not final response
                                            # Don't use it as final text
                                            break
                        
                        # Also check for text content directly in output
                        elif hasattr(output, "text"):
                            text = getattr(output, "text", "")
                            if text and text.strip():
                                final_text = text.strip()
                                print(f"✅ [Voice] Extracted from output.text")
                                break
        
        except Exception as e:
            print(f"❌ [Voice] Error extracting text from response: {e}")
            import traceback
            traceback.print_exc()
        
        # Track the final response object for tool call counting
        final_response_obj = response
        
        # Handle incomplete responses (response_status already set above)
        if not final_text:
            print(f"⚠️  [Voice] No text response extracted. Response status: {response_status}")
            output_types = [getattr(o, 'type', 'unknown') for o in getattr(response, 'output', [])]
            print(f"⚠️  [Voice] Response outputs: {output_types}")
            
            # If response is incomplete, try to continue it (may need multiple continuations for complex workflows)
            if response_status == 'incomplete':
                print(f"🔄 [Voice] Response incomplete, attempting to continue...")
                max_continuations = 3  # Allow up to 3 continuations for complex multi-tool workflows
                current_response = response
                
                # Count tool calls made so far
                current_tool_calls = count_tool_calls(response)
                print(f"📊 [Voice] Tool calls made so far: {current_tool_calls}/{VOICE_MAX_TOOL_CALLS}")
                
                for continuation_attempt in range(max_continuations):
                    try:
                        response_id = getattr(current_response, 'id', None)
                        if not response_id:
                            print(f"⚠️  [Voice] No response ID available for continuation")
                            break
                        
                        # Count tool calls in current response
                        current_tool_calls = count_tool_calls(current_response)
                        remaining_tool_calls = max(0, VOICE_MAX_TOOL_CALLS - current_tool_calls)
                        
                        if remaining_tool_calls <= 0:
                            print(f"⚠️  [Voice] Tool call limit reached ({current_tool_calls}/{VOICE_MAX_TOOL_CALLS}). Cannot continue.")
                            break
                        
                        print(f"🔄 [Voice] Continuation attempt {continuation_attempt + 1}/{max_continuations}...")
                        print(f"📊 [Voice] Remaining tool calls: {remaining_tool_calls}")
                        
                        # When using previous_response_id, max_tool_calls should be the TOTAL limit
                        # OpenAI API handles this automatically, but we pass it for clarity
                        continued_response = client.responses.create(
                            model=AGENT_MODEL,
                            input=messages,
                            tools=mcp_tools,
                            max_output_tokens=VOICE_MAX_OUTPUT_TOKENS,
                            max_tool_calls=VOICE_MAX_TOOL_CALLS,  # Total limit across all responses
                            previous_response_id=response_id,  # Continue from previous response
                        )
                        
                        continued_status = getattr(continued_response, 'status', 'unknown')
                        print(f"🔄 [Voice] Continued response status: {continued_status}")
                        
                        # Log tool calls in continued response for debugging
                        continued_outputs = getattr(continued_response, 'output', [])
                        continued_output_types = [getattr(o, 'type', 'unknown') for o in continued_outputs]
                        print(f"🔄 [Voice] Continued response outputs: {continued_output_types}")
                        
                        # Log any tool calls made in continuation
                        continuation_tool_calls = 0
                        for output in continued_outputs:
                            output_type = getattr(output, 'type', None)
                            if output_type in ("mcp_call_tool", "mcp_call"):
                                continuation_tool_calls += 1
                                tool_name = getattr(output, "name", "unknown")
                                tool_args = getattr(output, "arguments", {})
                                print(f"🔧 [Voice] Continued response called tool: {tool_name}")
                                print(f"   Args: {json.dumps(tool_args, indent=2) if isinstance(tool_args, dict) else tool_args}")
                        
                        # Count total tool calls across all responses
                        total_tool_calls = count_tool_calls(continued_response)
                        print(f"📊 [Voice] Total tool calls: {total_tool_calls}/{VOICE_MAX_TOOL_CALLS}")
                        
                        # Check if we've reached the tool call limit
                        if total_tool_calls >= VOICE_MAX_TOOL_CALLS:
                            print(f"⚠️  [Voice] Tool call limit reached ({total_tool_calls}/{VOICE_MAX_TOOL_CALLS})")
                            if not final_text and continued_status == 'incomplete':
                                final_text = "I've reached the maximum number of steps for this request. Please try asking a simpler question or break it into smaller parts."
                            break
                        
                        # Try to extract text from continued response
                        # First check output_text
                        output_text = getattr(continued_response, "output_text", None)
                        if isinstance(output_text, str) and output_text.strip():
                            final_text = output_text.strip()
                            print(f"✅ [Voice] Extracted from continued response output_text")
                            break
                        
                        # Then check message outputs
                        outputs = getattr(continued_response, 'output', [])
                        for output in outputs:
                            output_type = getattr(output, 'type', None)
                            if output_type == "message":
                                content = getattr(output, "content", [])
                                for content_item in content:
                                    if hasattr(content_item, "type") and content_item.type == "text":
                                        text = getattr(content_item, "text", "")
                                        if text and text.strip():
                                            final_text = text.strip()
                                            print(f"✅ [Voice] Extracted text from continued response message")
                                            break
                                if final_text:
                                    break
                        
                        # If we got a final response, break
                        if final_text:
                            final_response_obj = continued_response  # Track final response for tool call counting
                            break
                        
                        # Update current response for next iteration
                        current_response = continued_response
                        final_response_obj = continued_response  # Track final response for tool call counting
                        
                        # If still incomplete and we haven't reached max attempts or tool call limit, continue
                        if continued_status == 'incomplete' and continuation_attempt < max_continuations - 1:
                            if total_tool_calls < VOICE_MAX_TOOL_CALLS:
                                print(f"🔄 [Voice] Continued response also incomplete, will try again...")
                                continue
                            else:
                                print(f"⚠️  [Voice] Cannot continue: tool call limit reached")
                                if not final_text:
                                    final_text = "I've reached the maximum number of steps for this request. Please try asking a simpler question or break it into smaller parts."
                                break
                        elif continued_status == 'completed':
                            # Response completed but no text extracted - might be an issue
                            print(f"⚠️  [Voice] Continued response completed but no text found")
                            break
                        else:
                            # Unknown status or max attempts reached
                            break
                            
                    except Exception as e:
                        print(f"❌ [Voice] Error continuing incomplete response (attempt {continuation_attempt + 1}): {e}")
                        import traceback
                        traceback.print_exc()
                        break
            
            # If still no text, provide a helpful fallback
            if not final_text:
                final_text = "I'm having trouble completing that request. Could you try rephrasing your question?"
        
        response_status = getattr(response, 'status', 'unknown')
        print(f"🤖 [Voice] Response received (status: {response_status})")
        print(f"")
        
        # Log ALL outputs for debugging
        try:
            outputs = getattr(response, "output", [])
            print(f"📊 [Voice] Response has {len(outputs)} output items")
            
            # Debug: dump output types and basic info
            for idx, output in enumerate(outputs):
                output_type = getattr(output, "type", None)
                # Also log the actual attribute names to debug
                attrs = [attr for attr in dir(output) if not attr.startswith('_')]
                print(f"   [{idx}] Type: {output_type}, Attributes: {attrs[:5]}")  # First 5 attrs
            
            print(f"")
            print(f"📋 [Voice] Detailed output breakdown:")
            
            for idx, output in enumerate(outputs):
                output_type = getattr(output, "type", None)
                print(f"   [{idx}] Type: {output_type}")
                
                # Log when model requests to list tools
                if output_type == "mcp_list_tools":
                    server_label = getattr(output, "server_label", "unknown")
                    tools = getattr(output, "tools", [])
                    print(f"🔍 [Voice] Model requested to list tools from MCP server: {server_label}")
                    print(f"   Found {len(tools)} tools")
                
                # Log when model calls an MCP tool (could be mcp_call_tool or mcp_call)
                elif output_type in ("mcp_call_tool", "mcp_call"):
                    tool_name = getattr(output, "name", "unknown")
                    tool_args = getattr(output, "arguments", {})
                    print(f"🔧 [Voice] Model called MCP tool: {tool_name}")
                    print(f"   Args: {json.dumps(tool_args, indent=2)}")
                    
                    # Check if there was an error
                    error = getattr(output, "error", None)
                    if error:
                        print(f"❌ [Voice] Tool call error: {error}")
                    else:
                        # Get the result
                        content = getattr(output, "content", [])
                        if content:
                            # Content is usually a list of content items
                            for content_item in content:
                                if hasattr(content_item, "type") and content_item.type == "text":
                                    text = getattr(content_item, "text", "")
                                    print(f"✅ [Voice] Tool result (text): {text[:300]}...")
                                else:
                                    print(f"✅ [Voice] Tool result: {str(content_item)[:300]}...")
                
                # Log approval requests (if require_approval is set)
                elif output_type == "mcp_approval_request":
                    tool_name = getattr(output, "name", "unknown")
                    args = getattr(output, "arguments", {})
                    print(f"⚠️  [Voice] MCP tool approval requested for: {tool_name}")
                    print(f"   Args: {json.dumps(args, indent=2)}")
                
                # Log reasoning
                elif output_type == "reasoning":
                    summary = getattr(output, "summary", [])
                    if summary:
                        print(f"💭 [Voice] Reasoning: {summary[:100]}...")
                
                # Log message outputs
                elif output_type == "message":
                    print(f"💬 [Voice] Message output detected")
        except Exception as e:
            print(f"⚠️  [Voice] Error logging outputs: {e}")
            import traceback
            traceback.print_exc()
        
        # Count total tool calls used (from final response)
        total_tool_calls_used = count_tool_calls(final_response_obj)
        if total_tool_calls_used > 0:
            print(f"📊 [Voice] Total tool calls used: {total_tool_calls_used}/{VOICE_MAX_TOOL_CALLS}")
        
        print(f"")
        print(f"{'='*80}")
        print(f"💬 [Voice] Final Response: {final_text}")
        print(f"{'='*80}")
        print(f"")
        
        # Update conversation history with assistant's response
        conversation_sessions[session_id].append({"role": "assistant", "content": final_text})
        
        # Prune history to keep conversation manageable
        system_msgs = [m for m in conversation_sessions[session_id] if m.get("role") == "system"]
        convo = [m for m in conversation_sessions[session_id] if m.get("role") in ("user", "assistant")]
        if len(convo) > 12:  # Keep last 6 exchanges (6 user + 6 assistant)
            convo = convo[-12:]
        conversation_sessions[session_id] = system_msgs + convo
        
        return final_text
        
    except Exception as exc:
        print(f"❌ [Voice] OpenAI API error: {exc}")
        import traceback
        traceback.print_exc()
        return "I'm sorry, I encountered an error. Could you please repeat that?"


def _print_banner() -> None:
    print("")
    print("=== Twilio ConversationRelay Voice Assistant with OpenAI MCP ===")
    print("Configure your Twilio number with the /voicemcp/twiml endpoint.")
    print("Using OpenAI's Responses API with automatic MCP tool calling.")
    print("Documentation: https://platform.openai.com/docs/guides/tools-connectors-mcp")
    print("")




