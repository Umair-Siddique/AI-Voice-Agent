import os
import sys
import json
import asyncio
import re
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
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))  # Lower for voice to keep responses fast
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5")  # Using gpt-5 for better accuracy
VOICE_MAX_OUTPUT_TOKENS = int(os.getenv("VOICE_MAX_OUTPUT_TOKENS", "600"))
VOICE_MAX_TOOL_OBSERVATION_CHARS = int(os.getenv("VOICE_MAX_TOOL_OBSERVATION_CHARS", "1200"))

# Session storage for conversation history
conversation_sessions: Dict[str, List[Dict[str, str]]] = {}
# Agent conversation storage (for ReAct loop)
agent_conversations: Dict[str, List[Dict[str, Any]]] = {}


# ============================================================================
# MCP Helper Functions (adapted from mcp_server.py)
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
            "message": "Twilio ConversationRelay Voice Assistant with MCP tool calling is ready. Configure your Twilio number to use /voicemcp/twiml for voice calls.",
            "endpoints": {
                "twiml": "/voicemcp/twiml",
                "websocket": "/voicemcp/websocket",
                "connect_status": "/voicemcp/connect-status"
            },
            "features": {
                "mcp_tools": True,
                "react_agent": True,
                "mcp_server": SIMPLYBOOK_MCP_URL,
                "agent_model": AGENT_MODEL,
                "max_steps": AGENT_MAX_STEPS
            },
            "active_sessions": len(conversation_sessions),
            "active_agent_sessions": len(agent_conversations)
        }
    )


@voice_mcp_bp.route("/test", methods=["GET"])
def test_endpoint():
    """
    Test endpoint to verify the service is working with MCP tools
    """
    import platform
    return jsonify({
        "status": "ok",
        "message": "Voice AI MCP endpoint is working",
        "openai_configured": bool(Config.OPENAI_API_KEY),
        "mcp_configured": bool(SIMPLYBOOK_MCP_URL),
        "mcp_server_url": SIMPLYBOOK_MCP_URL,
        "agent_model": AGENT_MODEL,
        "agent_max_steps": AGENT_MAX_STEPS,
        "python_version": platform.python_version(),
        "active_sessions": len(conversation_sessions),
        "active_agent_sessions": len(agent_conversations)
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
    Generate an AI response for voice conversation using OpenAI with ReAct-style MCP tool calling.
    Optimized for voice: brief responses, low max_steps, fast model.
    """
    if not Config.OPENAI_API_KEY:
        return "Configuration error: OpenAI API key is not set."
    
    # Initialize agent conversation with ReAct protocol if new session
    if session_id not in agent_conversations:
        # Discover tools list for better guidance
        try:
            print(f"🔍 [Voice] Discovering tools from MCP server: {SIMPLYBOOK_MCP_URL}")
            tools_list = asyncio.run(_mcp_tools_brief())
            print(f"✅ [Voice] Discovered tools successfully")
        except Exception as exc:
            print(f"❌ [Voice] Failed to discover tools: {exc}")
            tools_list = f"- (failed to load tools: {exc})"

        system_prompt = (
            "You are a helpful AI voice assistant that can achieve goals by using tools. "
            "Keep responses VERY brief and conversational (1-2 sentences max for voice).\n\n"
            "Available tools:\n"
            f"{tools_list}\n\n"
            "Interaction protocol (STRICT):\n"
            "1) When you need to use a tool, respond with ONLY a single JSON object:\n"
            '{\n'
            '  "thought": "brief reason",\n'
            '  "action": "<tool_name>",\n'
            '  "action_input": { /* JSON arguments */ }\n'
            "}\n"
            "2) When ready to answer, respond with ONLY:\n"
            '{\n'
            '  "final": "<brief natural answer>"\n'
            "}\n"
            "Rules:\n"
            "- Keep responses SHORT and conversational for voice.\n"
            "- Never include text outside JSON.\n"
            "- For appointments: always confirm details before changes.\n"
        )
        agent_conversations[session_id] = [{"role": "system", "content": system_prompt}]
    
    # Append user message
    agent_conversations[session_id].append({"role": "user", "content": user_text})
    messages = agent_conversations[session_id]
    
    final_text: Optional[str] = None
    client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # ReAct loop with lower max_steps for voice (faster responses)
    for step in range(AGENT_MAX_STEPS):
        print(f"[Voice] ReAct step {step + 1}/{AGENT_MAX_STEPS}")
        
        try:
            # Use Responses API for better GPT-5 compatibility
            response = client.responses.create(
                model=AGENT_MODEL,
                input=messages,
                max_output_tokens=VOICE_MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:
            print(f"❌ [Voice] OpenAI error: {exc}")
            return "I'm sorry, I encountered an error. Could you please repeat that?"
        
        # Extract assistant text similar to mcp_server.py
        assistant_text = ""
        try:
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text.strip():
                assistant_text = output_text.strip()
        except Exception:
            pass
        if not assistant_text:
            try:
                first_output = response.output[0]
                first_content = first_output.content[0]
                assistant_text = (getattr(first_content, "text", "") or "").strip()
            except Exception:
                assistant_text = str(response)
        
        if not assistant_text:
            print(f"⚠️  [Voice] Empty assistant content. Raw response: {response}")
            return "I'm having trouble generating a response. Please try again."
        
        print(f"🤖 [Voice] Model response: {assistant_text[:150]}...")
        messages.append({"role": "assistant", "content": assistant_text})
        
        # Parse control JSON
        control = _extract_json_object(assistant_text)
        if not control:
            # Model didn't follow protocol - guide it back
            print(f"⚠️  [Voice] Model didn't return JSON; asking to follow protocol")
            messages.append({
                "role": "user",
                "content": (
                    "Invalid format. Respond with ONLY JSON:\n"
                    '{ "thought": "...", "action": "<tool>", "action_input": {...} }\n'
                    'or { "final": "<answer>" }'
                ),
            })
            continue
        
        # Check if final answer
        if "final" in control and control["final"]:
            final_text = str(control["final"]).strip()
            print(f"✅ [Voice] Final answer: {final_text}")
            break
        
        # Execute tool action
        action = control.get("action")
        action_input = control.get("action_input") or {}
        if not action:
            # No action specified, treat as final
            final_text = assistant_text
            break
        
        try:
            print(f"🔧 [Voice] Calling tool: {action} with args: {action_input}")
            observation = asyncio.run(_mcp_call(action, action_input))
            if len(observation) > VOICE_MAX_TOOL_OBSERVATION_CHARS:
                truncated_len = len(observation) - VOICE_MAX_TOOL_OBSERVATION_CHARS
                observation = observation[:VOICE_MAX_TOOL_OBSERVATION_CHARS] + f"... (truncated {truncated_len} chars)"
            print(f"✅ [Voice] Tool '{action}' result: {observation[:150]}...")
        except Exception as exc:
            error_msg = f"Tool error calling '{action}': {str(exc)}"
            print(f"❌ [Voice] {error_msg}")
            # Try to fetch schema to help model
            try:
                tool_schema = asyncio.run(_fetch_tool_schema(action))
            except Exception:
                tool_schema = None
            schema_hint = f"\n\nInput schema:\n{json.dumps(tool_schema, indent=2)}" if tool_schema else ""
            observation = error_msg + schema_hint
        
        # Pass observation back to model
        messages.append({
            "role": "user",
            "content": f"Tool '{action}' returned: {observation}",
        })
    
    # If no final answer, provide fallback
    if not final_text:
        final_text = "I need more information to help you. Could you provide more details?"
    
    # Save the final response to conversation history
    agent_conversations[session_id].append({"role": "assistant", "content": final_text})
    
    # Prune history to keep manageable (keep system + last 6 turns)
    system_msgs = [m for m in agent_conversations[session_id] if m.get("role") == "system"]
    convo = [m for m in agent_conversations[session_id] if m.get("role") in ("user", "assistant")]
    if len(convo) > 12:  # 6 user + 6 assistant
        convo = convo[-12:]
    agent_conversations[session_id] = system_msgs + convo
    
    return final_text


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

