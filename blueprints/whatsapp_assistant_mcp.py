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
SIMPLYBOOK_MCP_URL = os.getenv("SIMPLYBOOK_MCP_URL", "https://simplybook-mcp-server-q8v5.onrender.com/sse")
SIMPLYBOOK_MCP_LABEL = os.getenv("SIMPLYBOOK_MCP_LABEL", "simplybook")
SIMPLYBOOK_MCP_HEADERS_JSON = os.getenv("SIMPLYBOOK_MCP_HEADERS_JSON", "").strip()
SIMPLYBOOK_MCP_REQUIRE_APPROVAL = os.getenv("SIMPLYBOOK_MCP_REQUIRE_APPROVAL", "").strip()

# WhatsApp specific tuning
WHATSAPP_HISTORY_LIMIT = int(os.getenv("WHATSAPP_HISTORY_LIMIT", "12"))  # user+assistant turns to keep
WHATSAPP_MAX_MESSAGE_CHARS = int(os.getenv("WHATSAPP_MAX_MESSAGE_CHARS", "1500"))  # Twilio limit 1600
WHATSAPP_AGENT_MODEL = os.getenv("WHATSAPP_AGENT_MODEL", "gpt-5.2")
WHATSAPP_AGENT_MAX_STEPS = int(os.getenv("WHATSAPP_AGENT_MAX_STEPS", "15"))  # Max ReAct loop iterations
WHATSAPP_MAX_OUTPUT_TOKENS = int(os.getenv("WHATSAPP_MAX_OUTPUT_TOKENS", "1200"))

# In-memory session storage (replace with persistent store for production)
conversations: Dict[str, List[Dict[str, Any]]] = {}


def _build_mcp_tools_spec() -> List[Dict[str, Any]]:
    """
    Build the Responses API 'tools' specification for a remote MCP server
    following the official OpenAI documentation:
    https://platform.openai.com/docs/guides/tools-connectors-mcp
    """
    tool: Dict[str, Any] = {
        "type": "mcp",
        "server_label": SIMPLYBOOK_MCP_LABEL,
        "server_url": SIMPLYBOOK_MCP_URL,
    }

    if SIMPLYBOOK_MCP_HEADERS_JSON:
        try:
            headers = json.loads(SIMPLYBOOK_MCP_HEADERS_JSON)
            if isinstance(headers, dict) and headers:
                tool["headers"] = headers
        except json.JSONDecodeError:
            print("⚠️  WARNING: SIMPLYBOOK_MCP_HEADERS_JSON is not valid JSON; ignoring.")

    require_approval = SIMPLYBOOK_MCP_REQUIRE_APPROVAL or "never"
    tool["require_approval"] = require_approval

    return [tool]


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





def _build_whatsapp_system_prompt(tools_list: str) -> str:
    """
    Build system prompt for WhatsApp aligned with the OpenAI MCP connector flow.
    """
    return (
        "You are the Flexbody Solution WhatsApp assistant, helping clients with assisted stretching sessions.\n\n"
        "You have access to tools that fetch real-time information about services, availability, bookings, and contact details. "
        "Use these tools to get accurate data instead of relying on static information.\n\n"
        "Available tools:\n"
        f"{tools_list}\n\n"
        "BOOKING WORKFLOW (Follow these steps strictly):\n\n"
        "For NEW BOOKINGS:\n"
        "1. Gather required details: Ask for client name, phone number, email, preferred date/time, and service type if not provided\n"
        "2. Check availability: Use tools to verify the requested time slot is available\n"
        "3. Confirm with client: Present the details clearly and ask 'Should I confirm this booking for you?'\n"
        "4. Execute booking: Only after client confirms, use the booking tool to schedule\n"
        "5. Verify success: Check the tool result to ensure booking was created successfully\n"
        "6. Send confirmation: If booking succeeded, say something like 'All set! Your appointment is confirmed for [date] at [time]. See you then!'\n\n"
        "For RESCHEDULING:\n"
        "1. Identify booking: Ask which appointment they want to reschedule if unclear\n"
        "2. Get new time: Ask for their preferred new date/time\n"
        "3. Check availability: Verify the new time slot is available\n"
        "4. Confirm change: Show old vs new time and ask 'Should I go ahead with this change?'\n"
        "5. Execute reschedule: Only after confirmation, use the reschedule tool\n"
        "6. Verify success: Check tool result to confirm the change was made\n"
        "7. Send confirmation: If successful, say 'Done! Your appointment has been moved to [new date] at [new time].'\n\n"
        "For CANCELLATIONS:\n"
        "1. Identify booking: Ask which appointment they want to cancel if unclear\n"
        "2. Confirm cancellation: Show the appointment details and ask 'Are you sure you want to cancel this?'\n"
        "3. Execute cancellation: Only after confirmation, use the cancel tool\n"
        "4. Verify success: Check tool result to confirm cancellation\n"
        "5. Send confirmation: If successful, say 'Your appointment has been cancelled. Let me know if you need to book another time.'\n\n"
        "CRITICAL RULES:\n"
        "- ALWAYS check tool results to know if an action succeeded or failed\n"
        "- If a booking tool returns success, DO NOT ask for confirmation again - the booking is complete\n"
        "- If a tool returns an error, explain the issue clearly and offer alternatives\n"
        "- Never assume details - always ask if information is missing\n"
        "- Never take action without explicit customer confirmation first\n\n"
        "Response style:\n"
        "- Keep messages simple, clear, and under 4 short sentences\n"
        "- Plain text only (no markdown, asterisks, or code blocks)\n"
        "- Be warm, friendly, and professional\n"
        "- Focus on helping clients with their stretching appointments"
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




def _run_whatsapp_agent_step_loop(session_id: str, incoming_msg: str) -> str:
    """
    WhatsApp agent loop that uses the OpenAI Responses API with the official MCP
    connector so tools are discovered and executed server-side. We keep a
    bounded continuation loop to handle incomplete responses while preserving
    the existing agent feel.
    """
    if session_id not in conversations:
        _init_conversation(session_id)

    messages = conversations[session_id]
    messages.append({"role": "user", "content": incoming_msg})

    final_text: Optional[str] = None
    
    mcp_tools = _build_mcp_tools_spec()
    previous_response_id: Optional[str] = None

    def _extract_response_text(response_obj: Any) -> str:
        """Best-effort extraction of the assistant's final message."""
        try:
            output_text = getattr(response_obj, "output_text", None)
            if isinstance(output_text, str) and output_text.strip():
                return output_text.strip()
        except Exception:
            pass

        try:
            outputs = getattr(response_obj, "output", [])
            for output in outputs:
                if getattr(output, "type", None) == "message":
                    for content_item in getattr(output, "content", []):
                        if getattr(content_item, "type", None) == "text":
                            text = getattr(content_item, "text", "")
                            if text and text.strip():
                                return text.strip()
        except Exception:
            pass

        try:
            first_output = response_obj.output[0]
            first_content = first_output.content[0]
            text = getattr(first_content, "text", "")
            if text and text.strip():
                return text.strip()
        except Exception:
            pass

        return ""

    for step_num in range(WHATSAPP_AGENT_MAX_STEPS):
        try:
            print(f"🔄 Step {step_num + 1}/{WHATSAPP_AGENT_MAX_STEPS} for session {session_id} (MCP)")
            response_kwargs = {
                "model": WHATSAPP_AGENT_MODEL,
                "input": messages,
                "tools": mcp_tools,
                "max_output_tokens": WHATSAPP_MAX_OUTPUT_TOKENS,
                "max_tool_calls": WHATSAPP_AGENT_MAX_STEPS,
            }
            if previous_response_id:
                response_kwargs["previous_response_id"] = previous_response_id
            response = openai_client.responses.create(**response_kwargs)
        except Exception as exc:
            print(f"❌ OpenAI API error: {exc}")
            return "Sorry, I'm having trouble processing your request right now. Please try again."

        response_status = getattr(response, "status", "unknown")
        final_text = _extract_response_text(response)
        if final_text and response_status != "incomplete":
            print(f"✅ Got final answer: {final_text[:100]}...")
            break

        if response_status == "incomplete":
            previous_response_id = getattr(response, "id", None)
            print(f"ℹ️  Response incomplete, continuing with response_id={previous_response_id}")
            if not previous_response_id:
                break
            continue

        if final_text:
            break

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

