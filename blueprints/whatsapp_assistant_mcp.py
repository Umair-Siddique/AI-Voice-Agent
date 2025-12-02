import os
import re
import asyncio
import json
from typing import Any, Dict, List, Optional

from flask import Blueprint, request, jsonify
from openai import OpenAI
from twilio.rest import Client
from fastmcp import Client as FastMCPClient

from config import Config
from utils import SYSTEM_MESSAGE
from celery_app import celery_app

# Initialize Twilio and OpenAI clients
twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

whatsapp_assistant_mcp_bp = Blueprint("whatsappmcp", __name__)

# Remote MCP configuration
SIMPLYBOOK_MCP_URL = os.getenv("SIMPLYBOOK_MCP_URL", "https://simplybook-mcp-server.onrender.com/sse")
SIMPLYBOOK_MCP_LABEL = os.getenv("SIMPLYBOOK_MCP_LABEL", "simplybook")
SIMPLYBOOK_MCP_HEADERS_JSON = os.getenv("SIMPLYBOOK_MCP_HEADERS_JSON", "").strip()
SIMPLYBOOK_MCP_REQUIRE_APPROVAL = os.getenv("SIMPLYBOOK_MCP_REQUIRE_APPROVAL", "").strip()

# WhatsApp specific tuning
WHATSAPP_HISTORY_LIMIT = int(os.getenv("WHATSAPP_HISTORY_LIMIT", "12"))  # user+assistant turns to keep
WHATSAPP_MAX_MESSAGE_CHARS = int(os.getenv("WHATSAPP_MAX_MESSAGE_CHARS", "1500"))  # Twilio limit 1600
WHATSAPP_AGENT_MODEL = os.getenv("WHATSAPP_AGENT_MODEL", "gpt-5")
WHATSAPP_AGENT_MAX_STEPS = int(os.getenv("WHATSAPP_AGENT_MAX_STEPS", "15"))  # Max ReAct loop iterations

# In-memory session storage (replace with persistent store for production)
conversations: Dict[str, List[Dict[str, Any]]] = {}


def _safe_to_string(value: Any) -> str:
    """Best-effort string conversion for tool observation."""
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
    """Call FastMCP tool directly (same as mcp_server.py)."""
    mcp_url = SIMPLYBOOK_MCP_URL.rstrip("/")
    try:
        client = FastMCPClient(mcp_url)
        async with client:
            await asyncio.sleep(0.5)
            result = await client.call_tool(tool_name, arguments or {})
            data = getattr(result, "data", result)
            return _safe_to_string(data)
    except Exception as exc:
        raise Exception(f"MCP call failed for '{tool_name}': {exc}")


async def _mcp_tools_brief(limit: int = 24) -> str:
    """Fetch list of tools for prompt priming (same as mcp_server.py)."""
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
    Extract the first JSON object from text and parse it (same as mcp_server.py).
    Accepts plain JSON or fenced ```json blocks.
    """
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None

    return None


async def _fetch_tool_schema(tool_name: str) -> Optional[Dict[str, Any]]:
    """Fetch tool schema to help the model fix mistakes (same as mcp_server.py)."""
    client = FastMCPClient(SIMPLYBOOK_MCP_URL.rstrip("/"))
    async with client:
        await asyncio.sleep(0.3)
        tools = await client.list_tools()
        for t in tools:
            if getattr(t, "name", None) == tool_name:
                schema = getattr(t, "inputSchema", None) or getattr(t, "schema", None)
                return schema if isinstance(schema, dict) else None
    return None


def _build_whatsapp_system_prompt(tools_list: str) -> str:
    """
    Build ReAct-style system prompt for WhatsApp (same as mcp_server.py /react endpoint).
    """
    return (
        "You are a helpful AI agent that can achieve goals by planning and "
        "invoking tools exposed by a SimplyBook MCP server. Use a tight loop: "
        "think -> act (tool) -> observe -> repeat, then provide a final answer.\n\n"
        "Available tools:\n"
        f"{tools_list}\n\n"
        "Scheduling safety:\n"
        "- Never assume which appointment the user wants to change.\n"
        "- If multiple bookings match their description (e.g., same day or client), list those options, ask them to choose, and wait for a confirmation.\n"
        "- Repeat the confirmed client + appointment details before calling any reschedule, cancel, or booking tool.\n\n"
        "Interaction protocol (STRICT):\n"
        "1) When you need to use a tool, respond with ONLY a single JSON object:\n"
        "{\n"
        '  "thought": "very brief reason",\n'
        '  "action": "<tool_name>",\n'
        '  "action_input": { /* JSON arguments for the tool */ }\n'
        "}\n"
        "2) When you are ready to answer the user, respond with ONLY:\n"
        "{\n"
        '  "final": "<natural language answer>"\n'
        "}\n"
        "Rules:\n"
        "- Never include explanatory text outside of the JSON object.\n"
        "- Keep 'thought' short. Output human-like, concise, helpful 'final'.\n"
        "- Keep final answers under 4 short sentences for WhatsApp.\n"
        "- Use plain text only (no markdown)."
    )


def _init_conversation(session_id: str) -> None:
    """
    Initialize a new WhatsApp conversation session with ReAct system prompt.
    """
    try:
        print(f"🔍 Discovering tools from MCP server for session {session_id}")
        tools_list = asyncio.run(_mcp_tools_brief())
        print(f"✅ Discovered tools successfully")
    except Exception as exc:
        print(f"❌ Failed to discover tools: {exc}")
        tools_list = f"- (failed to load tools: {exc})"

    conversations[session_id] = [
        {
            "role": "system",
            "content": _build_whatsapp_system_prompt(tools_list),
        },
    ]


def _truncate_for_whatsapp(text: str) -> str:
    """Ensure responses fit within WhatsApp message limits."""
    if len(text) <= WHATSAPP_MAX_MESSAGE_CHARS:
        return text
    return text[: WHATSAPP_MAX_MESSAGE_CHARS - 3] + "..."




def _send_whatsapp_message(body: str, to_number: str):
    """
    Send WhatsApp message respecting Twilio 1600 character limit by chunking if needed.
    Returns list of message SIDs.
    """
    chunk_size = min(WHATSAPP_MAX_MESSAGE_CHARS, 1500)
    sids = []
    for idx in range(0, len(body), chunk_size):
        chunk = body[idx : idx + chunk_size]
        msg = twilio_client.messages.create(
            body=chunk,
            from_=Config.TWILIO_WHATSAPP_NUMBER,
            to=to_number,
        )
        sids.append(msg.sid)
    return sids


@whatsapp_assistant_mcp_bp.route("/", methods=["GET"])
def index():
    """Simple status endpoint."""
    return jsonify(
        {
            "success": True,
            "message": "WhatsApp MCP assistant ready. Point Twilio webhook to /whatsappmcp/message",
            "mcp_server": SIMPLYBOOK_MCP_URL,
            "history_sessions": len(conversations),
            "history_limit": WHATSAPP_HISTORY_LIMIT,
        }
    )


def _run_whatsapp_agent_step_loop(session_id: str, incoming_msg: str) -> str:
    """
    ReAct-style agent loop for WhatsApp (same pattern as mcp_server.py /react endpoint).
    - Asks the model to emit strict JSON for tool actions or final answers.
    - Calls MCP tools locally via FastMCP.
    - Iterates until a final answer is produced or max steps reached.
    """
    if session_id not in conversations:
        _init_conversation(session_id)

    messages = conversations[session_id]
    messages.append({"role": "user", "content": incoming_msg})

    final_text: Optional[str] = None
    
    protocol_reminder = (
        "Invalid format. Respond with ONLY one JSON object per the protocol. "
        "Either an action call:\n"
        '{ "thought": "...", "action": "<tool_name>", "action_input": { ... } }\n'
        "or a final answer:\n"
        '{ "final": "<answer>" }'
    )

    for step_num in range(WHATSAPP_AGENT_MAX_STEPS):
        # Ask the model what to do next
        try:
            print(f"🔄 Step {step_num + 1}/{WHATSAPP_AGENT_MAX_STEPS} for session {session_id}")
            response = openai_client.responses.create(
                model=WHATSAPP_AGENT_MODEL,
                input=messages,
            )
        except Exception as exc:
            print(f"❌ OpenAI API error: {exc}")
            return "Sorry, I'm having trouble processing your request right now. Please try again."

        # Extract the assistant's raw text
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

        print(f"🤖 Model response: {assistant_text[:200]}...")
        messages.append({"role": "assistant", "content": assistant_text})

        # Try to parse the control JSON
        control = _extract_json_object(assistant_text)
        if not control:
            # If the model didn't follow protocol, add a rail and continue
            print(f"⚠️  Model didn't return JSON control object; asking to follow protocol")
            messages.append({
                "role": "user",
                "content": protocol_reminder,
            })
            continue

        if "final" in control and control["final"]:
            final_text = str(control["final"]).strip()
            print(f"✅ Got final answer: {final_text[:100]}...")
            break

        action = control.get("action")
        action_input = control.get("action_input") or {}
        if not action:
            # No action specified; assume final
            final_text = assistant_text
            break

        # Execute tool and add observation
        try:
            print(f"🔧 Calling tool: {action} with args: {action_input}")
            observation = asyncio.run(_mcp_call(action, action_input))
            print(f"✅ Tool '{action}' succeeded: {observation[:200]}...")
        except Exception as exc:
            error_msg = f"Tool error calling '{action}': {str(exc)}"
            print(f"❌ {error_msg}")
            # Try to fetch tool schema to help model correct inputs
            try:
                tool_schema = asyncio.run(_fetch_tool_schema(action))
            except Exception as schema_exc:
                tool_schema = None
                print(f"⚠️  Failed to fetch tool schema for '{action}': {schema_exc}")
            schema_hint = f"\n\nInput schema for '{action}':\n{json.dumps(tool_schema, indent=2)}" if tool_schema else ""
            observation = error_msg + schema_hint

        # Truncate observation if too long for WhatsApp context
        if len(observation) > WHATSAPP_MAX_MESSAGE_CHARS:
            over = len(observation) - WHATSAPP_MAX_MESSAGE_CHARS
            observation = observation[:WHATSAPP_MAX_MESSAGE_CHARS] + f"... (truncated {over} chars)"

        # Pass observation back to the model as a user message
        messages.append({
            "role": "user",
            "content": f"Tool '{action}' returned: {observation}",
        })

    if not final_text:
        final_text = "I stopped before reaching a final answer due to the step limit. You can provide missing details (e.g., provider_id or date) and try again."

    # Truncate for WhatsApp limits
    final_text = _truncate_for_whatsapp(final_text)

    # Save final answer to conversation history
    messages.append({"role": "assistant", "content": final_text})

    # Trim history (system + last N dialogue turns)
    system_prompts = [m for m in messages if m["role"] == "system"]
    dialogue = [m for m in messages if m["role"] in ("user", "assistant")]
    if len(dialogue) > WHATSAPP_HISTORY_LIMIT:
        dialogue = dialogue[-WHATSAPP_HISTORY_LIMIT :]
    conversations[session_id] = system_prompts + dialogue

    return final_text


@celery_app.task(name="whatsapp_mcp.process_message")
def process_whatsapp_mcp_message(session_id: str, from_number: str, to_number: str, incoming_msg: str) -> None:
    """
    Celery task: run the MCP-enabled WhatsApp agent and send the final reply via Twilio.
    """
    try:
        final_text = _run_whatsapp_agent_step_loop(session_id=session_id, incoming_msg=incoming_msg)
    except Exception as exc:
        print(f"❌ WhatsApp MCP background error: {exc}")
        try:
            _send_whatsapp_message("Sorry, I ran into an error. Please try again shortly.", from_number)
        except Exception as send_exc:
            print(f"❌ Failed to send error notification via WhatsApp (background): {send_exc}")
        return

    try:
        _send_whatsapp_message(final_text, from_number)
    except Exception as send_exc:
        print(f"❌ Failed to send WhatsApp MCP reply via Twilio: {send_exc}")


@whatsapp_assistant_mcp_bp.route("/message", methods=["POST"])
def handle_whatsapp_message():
    """
    Twilio WhatsApp webhook.

    This endpoint now only enqueues a Celery task and returns quickly so that
    Twilio always receives a fast 2xx response. The long-running MCP agent
    loop and Twilio reply are handled asynchronously in the worker.
    """
    incoming_msg = (request.values.get("Body") or "").strip()
    from_number = request.values.get("From", "").strip()
    to_number = request.values.get("To", "").strip()

    if not incoming_msg or not from_number:
        return jsonify({"success": False, "error": "Missing message or From number"}), 400

    session_id = from_number  # Use WhatsApp sender as session identifier

    try:
        process_whatsapp_mcp_message.delay(session_id, from_number, to_number, incoming_msg)
    except Exception as exc:
        print(f"❌ Failed to enqueue WhatsApp MCP Celery task: {exc}")
        return jsonify({"success": False, "error": "Failed to queue message for background processing"}), 500

    # Fast acknowledgment for Twilio; the actual reply is sent later via REST.
    return jsonify({"success": True, "queued": True}), 200


@whatsapp_assistant_mcp_bp.route("/clear-session", methods=["POST"])
def clear_session():
    """Clear a WhatsApp conversation identified by phone number."""
    payload = request.get_json(silent=True) or {}
    phone_number = payload.get("phone_number", "").strip()

    if phone_number and not phone_number.startswith("whatsapp:"):
        phone_number = f"whatsapp:{phone_number}"

    if phone_number in conversations:
        del conversations[phone_number]
        return jsonify({"success": True, "message": f"Cleared session for {phone_number}"}), 200

    return jsonify({"success": False, "message": "Session not found"}), 404


@whatsapp_assistant_mcp_bp.route("/clear-all", methods=["POST"])
def clear_all_sessions():
    """Clear every WhatsApp conversation session."""
    conversations.clear()
    return jsonify({"success": True, "message": "All WhatsApp MCP sessions cleared"}), 200

