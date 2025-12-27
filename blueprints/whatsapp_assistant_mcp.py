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

