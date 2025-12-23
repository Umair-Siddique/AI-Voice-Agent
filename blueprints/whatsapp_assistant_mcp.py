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
        "You are the Flexbody Solution WhatsApp assistant, helping clients with assisted stretching sessions. "
        "Flexbody Solution specializes in improving mobility, flexibility, and overall physical well-being through "
        "personalized assisted stretching. We serve athletes, fitness enthusiasts, office workers, injured individuals, "
        "and seniors. Our services include one-on-one assisted stretch sessions and corporate wellness programs.\n\n"
        "You can call tools via the MCP server to manage bookings, check availability, and handle appointments. "
        "Plan briefly, use the available tools to fetch real data, and reply in plain text.\n\n"
        "Available tools:\n"
        f"{tools_list}\n\n"
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
        "3. Review the response to see what additional information is required\n"
        "4. For each REQUIRED field in the response:\n"
        "   - Ask the user for that information in a friendly, conversational way\n"
        "   - For select/dropdown fields, present the available options naturally\n"
        "   - Never mention technical field IDs or the phrase 'intake forms' to users\n"
        "   - Frame questions based on the field name (e.g., 'Health Conditions' → 'Do you have any health conditions?')\n"
         "5. Once you have ALL required information, present the complete booking details (date, time, service) and ask 'Should I confirm this booking for you (Mentioning the details provided by user)?' - wait for user confirmation\n"
         "6. Only after user confirms, call create_booking with:\n"
         "   - All standard booking parameters (service_id, provider_id, client_id, start_datetime, etc.)\n"
         "   - additional_fields array formatted as:\n"
         "     [{\"field\": \"field_id_from_get_additional_fields\", \"value\": \"user_provided_value\"}, ...]\n"
         "7. NEVER create a booking without first calling get_additional_fields\n"
         "8. NEVER skip required additional_fields - the booking WILL fail with a 400 error\n"
         "9. If get_additional_fields returns no required fields, you can proceed without additional_fields parameter\n\n"
        
        "Example conversation flow:\n"
        "User: 'I want to book a stretching session tomorrow at 3pm'\n"
        "AI: [internally calls get_services to identify service_id]\n"
        "AI: [internally calls get_additional_fields with service_id]\n"
        "AI: [sees 'Health Conditions' (textarea, required) and 'Preferred Contact' (select, required)]\n"
        "AI: 'Great! Before I book your session, do you have any injuries or health conditions I should know about?'\n"
        "User: 'I have lower back pain'\n"
        "AI: 'Thanks! How would you prefer we contact you - Email, Phone, or WhatsApp?'\n"
        "User: 'WhatsApp please'\n"
        "AI: [internally calls get_available_slots for tomorrow]\n"
        "AI: [internally calls create_booking with additional_fields: [\n"
        "       {\"field\": \"abc123...\", \"value\": \"Lower back pain\"},\n"
        "       {\"field\": \"def456...\", \"value\": \"WhatsApp\"}\n"
        "     ]]\n"
        "AI: 'Perfect! Your stretching session is confirmed for tomorrow at 3pm. See you then!'\n\n"
        # ========== END BOOKING WORKFLOW ==========
        
        "RESCHEDULING WORKFLOW:\n"
        "When a user wants to reschedule an appointment:\n"
        "1. Call get_booking_list with client_id or search to find their bookings\n"
        "2. If multiple bookings exist, ask user to confirm which one to reschedule\n"
        "3. Get the booking_id from the selected booking\n"
        "4. Call get_available_slots with the desired new date to show available times\n"
        "5. Once user confirms new time, call edit_booking with:\n"
        "   - booking_id: The ID from step 3\n"
        "   - booking_data: {\"start_datetime\": \"YYYY-MM-DD HH:mm:ss\"}\n"
        "6. Confirm the change to the user with old and new times\n\n"
        
        "CANCELLATION WORKFLOW:\n"
        "When a user wants to cancel an appointment:\n"
        "1. Call get_booking_list with client_id or search to find their bookings\n"
        "2. If multiple bookings exist, ask user to confirm which one to cancel\n"
        "3. Get the booking_id from the selected booking\n"
        "4. Show the appointment details and ask for final confirmation\n"
        "5. Call cancel_booking with the booking_id\n"
        "6. Confirm the cancellation to the user\n\n"
        
        "Example reschedule flow:\n"
        "User: 'I need to reschedule my appointment'\n"
        "AI: [calls get_booking_list for this client]\n"
        "AI: 'I see you have a session on Monday at 2pm. Is that the one you want to reschedule?'\n"
        "User: 'Yes'\n"
        "AI: 'What day works better for you?'\n"
        "User: 'Friday'\n"
        "AI: [calls get_available_slots for Friday]\n"
        "AI: 'I have openings at 10am, 2pm, and 4pm on Friday. Which works best?'\n"
        "User: '10am'\n"
        "AI: [calls edit_booking with new start_datetime]\n"
        "AI: 'Done! I've moved your appointment from Monday 2pm to Friday 10am.'\n\n"
        
        "Example cancellation flow:\n"
        "User: 'Cancel my appointment'\n"
        "AI: [calls get_booking_list for this client]\n"
        "AI: 'I see you have a session on Monday at 2pm. Is that the one you want to cancel?'\n"
        "User: 'Yes'\n"
        "AI: [calls cancel_booking]\n"
        "AI: 'Your appointment on Monday at 2pm has been cancelled.'\n\n"
        
        "Scheduling safety:\n"
        "- Never assume which appointment the user wants to change.\n"
        "- If multiple bookings match their description, list the options and ask them to choose before acting.\n"
        "- Repeat the confirmed client + appointment details before rescheduling, canceling, or booking.\n"
        "- Verify availability with get_available_slots before confirming new times.\n\n"
        "CRITICAL RULES:\n"
        "- We operate ONLY in Rotterdam - there is no Den Haag or other location\n"
        "- ALWAYS use Netherlands timezone for all bookings\n"
        "- ALWAYS assign any available therapist automatically - NEVER ask customers to choose or mention therapist names in messages\n"
        "- ALWAYS check tool results to know if an action succeeded or failed\n"
        "- If a booking tool returns success, DO NOT ask for confirmation again - the booking is complete\n"
        "- If get_additional_fields fails or returns an error, proceed without additional fields (some services may not require them)\n"
        "Response style:\n"
        "- Keep final answers in short and simple sentences.\n"
        "- Plain text only (no markdown or code blocks).\n"
        "- Be concise, friendly, and action-oriented.\n"
        "- Focus on helping clients improve their mobility and well-being through our stretching services.\n"
        "- When asking for additional information, be natural and conversational, not robotic."
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

