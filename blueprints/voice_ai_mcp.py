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
from db_helpers import (
    create_conversation_async, 
    end_conversation_async, 
    save_message_async,
    get_conversation_id,
    set_conversation_metadata,
    increment_message_count,
    cleanup_session_metadata,
    get_user_conversations,
    get_conversation_messages
)

voice_mcp_bp = Blueprint("voice_mcp", __name__)

# MCP Server Configuration
SIMPLYBOOK_MCP_URL = os.getenv("SIMPLYBOOK_MCP_URL", "https://simplybook-mcp-server-q8v5.onrender.com/sse")
SIMPLYBOOK_MCP_LABEL = os.getenv("SIMPLYBOOK_MCP_LABEL", "simplybook")
SIMPLYBOOK_MCP_HEADERS_JSON = os.getenv("SIMPLYBOOK_MCP_HEADERS_JSON", "").strip()
# For voice assistant, we want automatic tool calling without approval prompts
# Set to "never" (default) for automatic execution, or "always" to require manual approval
SIMPLYBOOK_MCP_REQUIRE_APPROVAL = os.getenv("SIMPLYBOOK_MCP_REQUIRE_APPROVAL", "never").strip()
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5.2")  # Using gpt-5.2 for better accuracy
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


def _sanitize_for_voice(text: str) -> str:
    """
    Remove special characters / Markdown artifacts that sound bad in TTS,
    while preserving the underlying text as much as possible.
    """
    if not text:
        return ""

    s = str(text)

    # Remove ONLY the fence marker lines, keep code content.
    # e.g. ```json ... ``` -> ... (no backticks spoken)
    s = re.sub(r"(?m)^\s*```[a-zA-Z0-9_-]*\s*$", "", s)

    # Replace markdown links: [label](url) -> label
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)

    # Replace bare autolinks: <https://...> -> https://...
    s = re.sub(r"<(https?://[^>]+)>", r"\1", s)

    # Strip headings: ### Title -> Title
    s = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", s)

    # Strip blockquote markers
    s = re.sub(r"(?m)^\s{0,3}>\s?", "", s)

    # Strip common list markers at line starts: "- ", "* ", "1. "
    s = re.sub(r"(?m)^\s*[-*+]\s+", "", s)
    s = re.sub(r"(?m)^\s*\d+\.\s+", "", s)

    # Remove emphasis/strikethrough markers and inline code ticks.
    s = s.replace("**", "")
    s = s.replace("__", "")
    s = s.replace("~~", "")
    s = s.replace("`", "")
    s = s.replace("*", "")
    # Keep underscores (can be meaningful in emails/usernames); remove other common table separators.
    s = s.replace("|", " ")

    # Collapse whitespace/newlines
    s = re.sub(r"\s+", " ", s).strip()
    return s


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


@voice_mcp_bp.route("/conversations/<phone_number>", methods=["GET"])
def get_user_conversation_history(phone_number: str):
    """
    Get conversation history for a specific phone number.
    Optional query parameters:
    - limit: number of conversations to return (default: 10)
    """
    try:
        limit = min(int(request.args.get("limit", 10)), 50)  # Cap at 50
        conversations = get_user_conversations(phone_number, limit)
        
        return jsonify({
            "success": True,
            "phone_number": phone_number,
            "conversations": conversations,
            "count": len(conversations)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@voice_mcp_bp.route("/conversations/<conversation_id>/messages", methods=["GET"])
def get_conversation_message_history(conversation_id: str):
    """
    Get all messages for a specific conversation.
    """
    try:
        messages = get_conversation_messages(conversation_id)
        
        return jsonify({
            "success": True,
            "conversation_id": conversation_id,
            "messages": messages,
            "count": len(messages)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
                        
                        # Create conversation record in database (async, non-blocking)
                        # This will automatically set the conversation metadata when complete
                        create_conversation_async(
                            session_id=session_id,
                            call_sid=call_sid,
                            user_phone=from_number or "unknown",
                            twilio_number=to_number or "unknown",
                            call_direction=direction or "inbound"
                        )
                        
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
                            
                            # Create conversation record for fallback session (async, non-blocking)
                            # This will automatically set the conversation metadata when complete
                            create_conversation_async(
                                session_id=session_id,
                                call_sid=call_sid or "unknown",
                                user_phone="unknown",
                                twilio_number="unknown",
                                call_direction="inbound"
                            )
                            
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
                        
                        # Save user message to database (async, non-blocking)
                        conversation_id = get_conversation_id(session_id)
                        if conversation_id:
                            message_order = increment_message_count(session_id)
                            save_message_async(
                                conversation_id=conversation_id,
                                role="user",
                                content=final_text,
                                message_order=message_order
                            )
                        
                        # Mark the time we are processing to avoid immediate follow-up fragments
                        buf_state["last_final_time"] = now

                        # Fetch conversation context (current session + last 5 previous conversations)
                        # This includes BOTH the current conversation messages AND previous call history
                        conversation_context = []
                        try:
                            # Get current conversation ID
                            conversation_id = get_conversation_id(session_id)
                            print(f"🔍 [WebSocket] Current conversation_id: {conversation_id}")
                            
                            if conversation_id:
                                # FIRST: Get messages from CURRENT conversation (already saved to DB)
                                print(f"📖 [WebSocket] Fetching current conversation messages from database")
                                current_conv_messages = get_conversation_messages(conversation_id)
                                print(f"🔍 [WebSocket] Raw messages from DB: {len(current_conv_messages) if current_conv_messages else 0}")
                                
                                current_messages = []
                                if current_conv_messages:
                                    for msg in current_conv_messages:
                                        role = msg.get("role")
                                        content = msg.get("content", "")
                                        if role in ("user", "assistant") and content:
                                            current_messages.append({
                                                "role": role,
                                                "content": content
                                            })
                                            print(f"   ✓ Loaded {role}: {content[:60]}...")
                                
                                if current_messages:
                                    print(f"✅ [WebSocket] Loaded {len(current_messages)} messages from current conversation")
                                else:
                                    print(f"ℹ️  [WebSocket] No messages in current conversation yet")
                                
                                # SECOND: Get previous conversations for context
                                user_phone = None
                                all_conversations = get_user_conversations("unknown", limit=50)
                                
                                # Find current conversation to get phone number
                                for conv in all_conversations:
                                    if conv.get("conversation_id") == conversation_id:
                                        user_phone = conv.get("user_phone")
                                        print(f"📞 [WebSocket] Found user_phone: {user_phone}")
                                        break
                                
                                previous_conversations = []
                                if user_phone and user_phone != "unknown":
                                    print(f"📞 [WebSocket] Fetching previous conversations for {user_phone}")
                                    
                                    # Get user's recent conversations (excluding current one)
                                    user_conversations = get_user_conversations(user_phone, limit=10)
                                    print(f"   Found {len(user_conversations)} total conversations for user")
                                    
                                    # Collect messages from previous conversations (last 10 messages)
                                    message_count = 0
                                    for conv in user_conversations:
                                        conv_id = conv.get("conversation_id")
                                        if conv_id == conversation_id:
                                            print(f"   Skipping current conversation {conv_id}")
                                            continue  # Skip current conversation
                                        
                                        print(f"   Loading messages from conversation {conv_id}")
                                        conv_messages = get_conversation_messages(conv_id or "")
                                        for msg in reversed(conv_messages):  # Get most recent first
                                            if message_count >= 10:
                                                break
                                            if msg.get("role") in ("user", "assistant"):
                                                previous_conversations.insert(0, {
                                                    "role": msg.get("role"),
                                                    "content": msg.get("content", "")
                                                })
                                                message_count += 1
                                        
                                        if message_count >= 10:
                                            break
                                    
                                    if previous_conversations:
                                        print(f"✅ [WebSocket] Loaded {len(previous_conversations)} messages from previous conversations")
                                    else:
                                        print(f"ℹ️  [WebSocket] No previous conversations found")
                                else:
                                    print(f"ℹ️  [WebSocket] No valid user_phone, skipping previous conversations")
                                
                                # Combine: previous conversations + current conversation messages
                                conversation_context = previous_conversations + current_messages
                                print(f"📚 [WebSocket] Total context: {len(conversation_context)} messages ({len(previous_conversations)} previous + {len(current_messages)} current)")
                            else:
                                print(f"⚠️  [WebSocket] No conversation_id found for session {session_id}")
                                
                        except Exception as e:
                            print(f"⚠️  [WebSocket] Error loading conversation context: {e}")
                            import traceback
                            traceback.print_exc()
                            conversation_context = []

                        # Check if tools are needed before generating response
                        # Get current conversation history (without the new user message yet)
                        current_history = conversation_sessions.get(session_id, [])
                        needs_tools = _check_if_tools_needed(final_text, current_history, conversation_context)
                        
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

                        # Generate AI response with full conversation context (previous + current)
                        ai_response = generate_voice_response(session_id, final_text, conversation_context)
                        buf_state["last_final"] = final_text
                        buf_state["last_final_time"] = time.time()
                        buf_state["buffer"] = ""
                        
                        print(f"[WebSocket] Generated AI response: {ai_response}")
                        
                        # Save AI response to database (async, non-blocking)
                        conversation_id = get_conversation_id(session_id)
                        if conversation_id:
                            message_order = increment_message_count(session_id)
                            save_message_async(
                                conversation_id=conversation_id,
                                role="assistant",
                                content=ai_response,
                                message_order=message_order
                            )
                        
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
            if session_id:
                # End conversation in database (async, non-blocking)
                end_conversation_async(session_id)
                
                # Clean up session data
                if session_id in conversation_sessions:
                    del conversation_sessions[session_id]
                    print(f"[WebSocket] Cleaned up session {session_id}")
                if session_id in prompt_buffers:
                    del prompt_buffers[session_id]
                
                # Clean up conversation metadata
                cleanup_session_metadata(session_id)
                
            print("[WebSocket] Connection closed")
    
    print("✅ WebSocket routes registered for Twilio ConversationRelay")


def _check_if_tools_needed(user_text: str, conversation_history: List[Dict[str, str]], conversation_context: List[Dict[str, str]] = None) -> bool:
    """
    Analyze the user query using OpenAI to determine if SimplyBook MCP tools are needed.
    Uses FULL conversation context (previous calls + current session) to better understand user intent.
    Returns True if tools are likely needed, False otherwise.
    
    Args:
        user_text: The current user query
        conversation_history: Current session conversation history (in-memory)
        conversation_context: Full conversation context from DB (previous calls + current session messages)
    """
    if not Config.OPENAI_API_KEY:
        return False
    
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # Build conversation context - use full context from DB if available
    history_lines: List[str] = []
    total_chars = 0
    
    # Use the full conversation context from database (includes previous calls + current session)
    if conversation_context:
        for msg in conversation_context:
            role = msg.get("role", "unknown")
            content = (msg.get("content", "") or "").strip()
            if not content or role not in ("user", "assistant"):
                continue
            line = f"{role}: {content}"
            total_chars += len(line)
            history_lines.append(line)
            if total_chars >= 2000:  # Increased limit to include more context
                break
    
    # Fallback to in-memory conversation history if no DB context
    if not history_lines:
        recent_msgs = [
            m for m in conversation_history[-8:]
            if m.get("role") in ("user", "assistant")
        ]
        for msg in recent_msgs:
            role = msg.get("role", "unknown")
            content = (msg.get("content", "") or "").strip()
            if not content:
                continue
            line = f"{role}: {content}"
            total_chars += len(line)
            history_lines.append(line)
            if total_chars >= 1500:
                break
    
    history_block = "\n".join(history_lines) if history_lines else "(no prior turns)"
    
    # Build a focused analysis prompt with full conversation context
    analysis_prompt = (
        "Analyze the following user query and determine if it requires accessing SimplyBook booking system tools.\n\n"
        "Use the conversation history (both previous and current) to understand context and user intent.\n\n"
        "Conversation history:\n"
        f"{history_block}\n\n"
        "Available SimplyBook MCP tools:\n"
        "- get_services: Get list of available services\n"
        "- get_available_slots: Check availability for booking\n"
        "- get_providers: Get list of service providers\n"
        "- create_booking: Create new appointments\n"
        "- edit_booking: Update/reschedule appointments\n"
        "- cancel_booking: Cancel appointments\n"
        "- get_booking_list: Get user's bookings\n"
        "- get_clients_list: Search/manage clients\n"
        "- create_client: Create new client\n"
        "- get_additional_fields: Get required booking fields\n\n"
        "Tools are needed when the user wants to:\n"
        "- Book, reschedule, or cancel appointments\n"
        "- Check availability or service information\n"
        "- View or manage their bookings\n"
        "- Any action requiring real-time booking system data\n\n"
        "Tools are NOT needed for:\n"
        "- General conversation or greetings\n"
        "- Questions about the business (use system knowledge)\n"
        "- Follow-up responses that don't require new data\n\n"
        "Current user query: " + user_text + "\n\n"
        "Based on the full conversation context and current query, respond with ONLY 'YES' if tools are needed, or 'NO' if not needed. No other text."
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


def generate_voice_response(session_id: str, user_text: str, conversation_context: List[Dict[str, str]] = None) -> str:
    """
    Generate an AI response for voice conversation using OpenAI's official MCP implementation.
    Following: https://platform.openai.com/docs/guides/tools-connectors-mcp
    The Responses API handles MCP tool calling automatically when tools are provided.
    Optimized for voice: brief responses, conversational tone.
    
    Args:
        session_id: Current session ID
        user_text: User's input text
        conversation_context: Full conversation context from DB (previous calls + current session messages)
    """
    if not Config.OPENAI_API_KEY:
        return "Configuration error: OpenAI API key is not set."
    
    # Build system prompt (always, for every turn)
    print(f"")
    print(f"{'='*80}")
    print(f"👤 [Voice] User: {user_text}")
    print(f"{'='*80}")
    print(f"")
    
    system_prompt = (
        "You are the Flexbody Solution assistant, helping clients with assisted stretching sessions. "
        "Flexbody Solution specializes in improving mobility, flexibility, and overall physical well-being through "
        "personalized assisted stretching. We serve athletes, fitness enthusiasts, office workers, injured individuals, "
        "and seniors. Our services include one-on-one assisted stretch sessions and corporate wellness programs.\n\n"
        "You can call tools via the MCP server to manage bookings, check availability, and handle appointments. "
        "Plan briefly, use the available tools to fetch real data, and reply in plain text.\n\n"

        "About Flexbody Solution:\n"
        "- Contact: info@flexbodysolution.com\n"
        "- Website: https://flexbodysolution.com/\n"
        "- Services: Assisted stretching therapy to improve mobility, reduce pain, prevent injuries, and enhance performance\n"
        "- Benefits: Improved flexibility, better posture, injury prevention, enhanced athletic performance, and relaxation\n\n"
        
        # ========== CRITICAL BOOKING WORKFLOW ==========
        "BOOKING WORKFLOW - FOLLOW THIS EXACTLY:\n"
        "When a user wants to book an appointment:\n"
        "1. First, identify the service_id they want to book\n"
        "2. IMMEDIATELY call get_additional_fields with that service_id\n"
        "3. Review the response - each field has these properties:\n"
        "   - 'name': A hash string like '185de2378348fc2103d16fa11f9ccf02' (THIS is what you use in 'field')\n"
        "   - 'id': A numeric ID like 1, 2, 3 (DO NOT use this in 'field')\n"
        "   - 'field_name': Human-readable name like 'Do you currently have any injuries?'\n"
        "   - 'optional': true/false (if false, MUST collect from user)\n"
        "4. For each field where 'optional' is false:\n"
        "   - Ask the user using the 'field_name' in a friendly, conversational way\n"
        "   - For select fields, present the 'field_options' naturally\n"
        "   - Store the user's answer to use in step 6\n"
        "5. Once you have ALL required information, present complete booking details and ask for confirmation\n"
        "6. CRITICAL - When calling create_booking, format additional_fields as a LIST like this:\n"
        "   additional_fields: [\n"
        "     {\"field\": \"<the 'name' hash from get_additional_fields>\", \"value\": \"<user's answer>\"},\n"
        "     {\"field\": \"<another 'name' hash>\", \"value\": \"<another answer>\"}\n"
        "   ]\n"
        "   DO NOT use the numeric 'id' - ONLY use the 'name' hash string!\n"
        "   DO NOT use a dict/object - it MUST be a LIST!\n"
        "7. NEVER create a booking without first calling get_additional_fields\n"
        "8. NEVER skip required fields (where optional=false)\n\n"
        
        "CONCRETE EXAMPLE - Booking:\n"
        "get_additional_fields returns:\n"
        "{\n"
        "  \"data\": [\n"
        "    {\"id\": 1, \"name\": \"185de2378348fc2103d16fa11f9ccf02\", \"field_name\": \"Do you have injuries?\", \"optional\": false},\n"
        "    {\"id\": 3, \"name\": \"efe4d8742c016d370a4eb2936b98fd2e\", \"field_name\": \"Area of focus\", \"optional\": false}\n"
        "  ]\n"
        "}\n"
        "You call create_booking with:\n"
        "{\n"
        "  \"service_id\": \"3\",\n"
        "  \"additional_fields\": [\n"
        "    {\"field\": \"185de2378348fc2103d16fa11f9ccf02\", \"value\": \"No\"},\n"
        "    {\"field\": \"efe4d8742c016d370a4eb2936b98fd2e\", \"value\": \"Core and back\"}\n"
        "  ]\n"
        "}\n"
        "Notice: Use 'name' (the hash), NOT 'id' (the number)! Use LIST format, NOT dict!\n\n"
        # ========== END BOOKING WORKFLOW ==========
        
        # ========== UNDERSTANDING BOOKING IDs vs CODES ==========
        "CRITICAL - Understanding Booking IDs vs Booking Codes:\n"
        "- Booking ID: Numeric ID like '123' or '456' - found in the 'id' field when you call get_booking_list\n"
        "- Booking Code (Ticket): String like '15m7hpawm' - this is ONLY for check-in, NOT for cancellation/rescheduling\n"
        "- To cancel or reschedule, you MUST use the booking ID from get_booking_list, NOT the booking code\n"
        "- NEVER try to cancel/reschedule using a booking code - it will fail with 404\n\n"
        
        "HOW TO FIND BOOKINGS:\n"
        "To find a customer's bookings, use get_booking_list with ONE of these approaches:\n"
        "1. Search by email: get_booking_list(search='customer@email.com')\n"
        "2. Search by phone: get_booking_list(search='+31612345678')\n"
        "3. Search by client_id: get_booking_list(client_id='316')\n"
        "4. Search by name: get_booking_list(search='John Smith')\n"
        "The result will have a 'data' array with booking objects. Each booking has:\n"
        "- 'id': The booking ID (use THIS for cancel/edit)\n"
        "- 'code': The booking code/ticket (ignore this for cancel/reschedule)\n"
        "- 'start_date_time': When the appointment is\n"
        "- 'service_name', 'provider_name', 'client_name', etc.\n\n"
        # ========== END BOOKING IDs EXPLANATION ==========
        
# ========== RESCHEDULING WORKFLOW ==========
"RESCHEDULING WORKFLOW - CRITICAL STEPS:\n"
"1. Call get_booking_list(search='<user email or phone or name>')\n"
"2. Look at the 'data' array and extract the booking's 'id' field (NOT 'code')\n"
"3. If multiple bookings, ask user to confirm which one to reschedule\n"
"4. CRITICAL: Call get_booking_details(booking_id='<the id from step 2>') to get:\n"
"   - service_id\n"
"   - provider_id\n"
"   - duration (to calculate end_datetime)\n"
"   - additional_fields (MUST include these when editing!)\n"
"5. Ask user for new date preference\n"
"6. Call get_available_slots with the service_id, provider_id, and new date\n"
"7. Show available times to user and get their choice\n"
"8. Calculate end_datetime = new_start_datetime + original booking duration\n"
"9. CRITICAL - Call edit_booking with ALL required fields INCLUDING additional_fields:\n"
"   {\n"
"     \"booking_id\": \"<id from step 2>\",\n"
"     \"booking_data\": {\n"
"       \"service_id\": \"<from step 4>\",\n"
"       \"provider_id\": \"<from step 4>\",\n"
"       \"start_datetime\": \"YYYY-MM-DD HH:mm:ss\",\n"
"       \"end_datetime\": \"YYYY-MM-DD HH:mm:ss\",\n"
"       \"additional_fields\": <copy array from step 4's additional_fields>\n"
"     }\n"
"   }\n"
"   IMPORTANT: Transform additional_fields from get_booking_details format to create_booking format:\n"
"   - get_booking_details returns: [{\"field\": \"hash\", \"value\": \"answer\", \"id\": 1, ...}]\n"
"   - edit_booking needs: [{\"field\": \"hash\", \"value\": \"answer\"}] (only field and value)\n"
"10. Confirm the change to the user\n\n"

"Example reschedule flow:\n"
"User: 'Reschedule my appointment'\n"
"AI: [calls get_booking_list(search='user@email.com')]\n"
"AI: [gets {'id': '640', 'start_date_time': '2026-01-13 13:15:00', ...}]\n"
"AI: [calls get_booking_details(booking_id='640')]\n"
"AI: [gets {\n"
"  'service_id': 3,\n"
"  'provider_id': 2,\n"
"  'duration': 75,\n"
"  'additional_fields': [\n"
"    {'field': '185de2378348fc2103d16fa11f9ccf02', 'value': 'No', 'id': 1, ...},\n"
"    {'field': 'efe4d8742c016d370a4eb2936b98fd2e', 'value': 'Hips', 'id': 3, ...}\n"
"  ]\n"
"}]\n"
"AI: 'Found your booking for Jan 13 at 1:15pm. What new date works?'\n"
"User: 'January 13 at 2:30pm'\n"
"AI: [calls get_available_slots(service_id='3', provider_id='2', date='2026-01-13')]\n"
"AI: [sees 14:30:00 is available]\n"
"AI: [calculates: start='2026-01-13 14:30:00', end='2026-01-13 15:45:00' (14:30 + 75 min)]\n"
"AI: [calls edit_booking(booking_id='640', booking_data={\n"
"  'service_id': '3',\n"
"  'provider_id': '2',\n"
"  'start_datetime': '2026-01-13 14:30:00',\n"
"  'end_datetime': '2026-01-13 15:45:00',\n"
"  'additional_fields': [\n"
"    {'field': '185de2378348fc2103d16fa11f9ccf02', 'value': 'No'},\n"
"    {'field': 'efe4d8742c016d370a4eb2936b98fd2e', 'value': 'Hips'}\n"
"  ]\n"
"})]\n"
"AI: 'Done! Rescheduled to Jan 13 at 2:30pm.'\n\n"
# ========== END RESCHEDULING ==========
        
        # ========== CANCELLATION WORKFLOW ==========
        "CANCELLATION WORKFLOW:\n"
        "1. Call get_booking_list(search='<user email or phone or name>')\n"
        "2. Look at the 'data' array in the result\n"
        "3. Find the relevant booking and extract its 'id' field (NOT 'code')\n"
        "4. If multiple bookings, ask user to confirm which one to cancel\n"
        "5. Show booking details and ask for final confirmation\n"
        "6. Call cancel_booking(booking_id='<the id from step 3>')\n"
        "7. Confirm the cancellation\n\n"
        
        "Example cancellation:\n"
        "User: 'Cancel my booking, code is 15m7hpawm'\n"
        "AI: [calls get_booking_list(search='user@email.com')]\n"
        "AI: [gets result with booking {'id': '789', 'code': '15m7hpawm', 'start_date_time': '2026-01-12 09:15:00', ...}]\n"
        "AI: 'I found your booking for Jan 12 at 9:15am. Confirm cancellation?'\n"
        "User: 'Yes'\n"
        "AI: [calls cancel_booking(booking_id='789')] ← Use 'id' NOT 'code'!\n"
        "AI: 'Your booking has been cancelled.'\n\n"
        # ========== END CANCELLATION ==========
        
        "Scheduling safety:\n"
        "- ALWAYS confirm personal details (name, email, or phone number) BEFORE creating, rescheduling, or canceling any booking\n"
        "- Never assume which appointment the user wants to change\n"
        "- If multiple bookings match, list options and ask user to choose\n"
        "- Always confirm booking details before making changes\n"
        "- Verify availability with get_available_slots before confirming new times\n\n"
        
        "CRITICAL RULES:\n"
        "- We operate ONLY in Rotterdam - there is no Den Haag or other location\n"
        "- ALWAYS use Netherlands timezone for all bookings\n"
        "- ALWAYS assign any available therapist automatically - NEVER ask customers to choose or mention therapist names\n"
        "- ALWAYS check tool results to know if an action succeeded or failed\n"
        "- If a booking tool returns success, DO NOT ask for confirmation again - the booking is complete\n"
        "- If get_additional_fields fails, proceed without additional fields (some services may not require them)\n\n"
        
        "Response style:\n"
        "- Keep answers short and simple sentences\n"
        "- Plain text only (no markdown or code blocks)\n"
        "- Be concise, friendly, and action-oriented\n"
        "- Focus on helping clients improve their mobility and well-being\n"
        "- When asking for information, be natural and conversational, not robotic"
    )
    
    # ALWAYS rebuild conversation from scratch using database context
    # This ensures we have the complete history on every turn
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation context if available (includes both previous calls AND current session from DB)
    if conversation_context and len(conversation_context) > 0:
        print(f"📚 [Voice] Rebuilding conversation with {len(conversation_context)} messages from database")
        
        # Add all previous messages from database
        for msg in conversation_context:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({
                    "role": role,
                    "content": content
                })
        
        print(f"✅ [Voice] Loaded full conversation history")
    else:
        print(f"ℹ️  [Voice] No previous conversation context available")
    
    # Append current user message
    messages.append({"role": "user", "content": user_text})
    
    # Update the session storage for this turn (will be used for fallback/debugging)
    conversation_sessions[session_id] = messages
    
    # Debug: Print the conversation structure
    print(f"📊 [Voice] Conversation has {len(messages)} messages:")
    for idx, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:100]  # First 100 chars
        print(f"   [{idx}] {role}: {content}...")
    
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
        
        # Final cleanup for TTS: remove special chars/markdown artifacts without changing meaning
        sanitized_text = _sanitize_for_voice(final_text)
        if sanitized_text:
            final_text = sanitized_text

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
        
        # Update in-memory conversation history with assistant's response
        # Note: This is less important now since we reload from database on each turn
        # The database is the source of truth, not the in-memory session
        if session_id in conversation_sessions:
            conversation_sessions[session_id].append({"role": "assistant", "content": final_text})
            
            # Light pruning only to prevent extreme memory usage in very long sessions
            # (We reload from DB anyway, so this is just for memory management)
            system_msgs = [m for m in conversation_sessions[session_id] if m.get("role") == "system"]
            convo = [m for m in conversation_sessions[session_id] if m.get("role") in ("user", "assistant")]
            if len(convo) > 50:  # Only prune if conversation gets very long
                convo = convo[-50:]  # Keep last 25 exchanges
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




