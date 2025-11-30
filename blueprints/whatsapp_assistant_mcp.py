import os
import re
import json
import asyncio
from typing import Any, Dict, List, Optional

import requests
from flask import Blueprint, request, jsonify
from fastmcp import Client as FastMCPClient
from openai import OpenAI
from twilio.rest import Client

from config import Config
from utils import SYSTEM_MESSAGE

# Initialize Twilio client
twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

whatsapp_assistant_mcp_bp = Blueprint('whatsappmcp', __name__)

# Simple in-memory conversation storage (for production, use a database)
conversations = {}

agent_conversations: Dict[str, List[Dict[str, Any]]] = {}
agent_cached_tool_schemas: Dict[str, Dict[str, Any]] = {}

SIMPLYBOOK_MCP_URL = os.getenv(
    "SIMPLYBOOK_MCP_URL",
    "https://simplybook-mcp-server.onrender.com/sse"
)
WHATSAPP_AGENT_MAX_STEPS = int(
    os.getenv("WHATSAPP_AGENT_MAX_STEPS", os.getenv("AGENT_MAX_STEPS", "15"))
)
WHATSAPP_AGENT_MODEL = os.getenv("WHATSAPP_AGENT_MODEL", "gpt-5")


def _safe_to_string(value: Any) -> str:
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
    """Call a remote MCP tool via FastMCP and return a printable result string."""
    mcp_url = SIMPLYBOOK_MCP_URL.rstrip("/")
    try:
        client = FastMCPClient(mcp_url)
        async with client:
            await asyncio.sleep(0.5)  # Give the SSE connection time to settle
            result = await client.call_tool(tool_name, arguments or {})
            data = getattr(result, "data", result)
            return _safe_to_string(data)
    except Exception as exc:
        raise Exception(f"MCP call failed for '{tool_name}': {exc}")


async def _mcp_tools_brief(limit: int = 24) -> str:
    """Return a short markdown list of available MCP tools."""
    mcp_url = SIMPLYBOOK_MCP_URL.rstrip("/")
    client = FastMCPClient(mcp_url)
    lines: List[str] = []
    async with client:
        await asyncio.sleep(0.5)
        tools = await client.list_tools()
        for idx, tool in enumerate(tools):
            if idx >= limit:
                lines.append(f"- ... and {len(tools) - limit} more")
                break
            name = getattr(tool, "name", "unknown_tool")
            description = getattr(tool, "description", "") or ""
            lines.append(f"- {name}: {description}")
    return "\n".join(lines) if lines else "- (no tools discovered)"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first JSON object from arbitrary text."""
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
    """Fetch and cache the schema for a given tool from the MCP server."""
    if tool_name in agent_cached_tool_schemas:
        return agent_cached_tool_schemas[tool_name]

    mcp_url = SIMPLYBOOK_MCP_URL.rstrip("/")
    client = FastMCPClient(mcp_url)
    async with client:
        await asyncio.sleep(0.3)
        tools = await client.list_tools()
        for tool in tools:
            if getattr(tool, "name", None) == tool_name:
                schema = getattr(tool, "inputSchema", None) or getattr(tool, "schema", None)
                if isinstance(schema, dict):
                    agent_cached_tool_schemas[tool_name] = schema
                    return schema
    return None


def _prime_agent_session(session_id: str) -> List[Dict[str, Any]]:
    """Ensure a ReAct session exists with the required system prompt."""
    if session_id in agent_conversations:
        return agent_conversations[session_id]

    try:
        tools_list = asyncio.run(_mcp_tools_brief())
        print(f"✅ Discovered MCP tools for WhatsApp session {session_id}")
    except Exception as exc:
        tools_list = f"- (failed to load tools: {exc})"
        print(f"❌ Could not discover MCP tools: {exc}")

    system_prompt = (
        "You are a WhatsApp AI agent that uses a SimplyBook MCP server to achieve goals. "
        "Employ the ReAct loop: think -> act -> observe -> repeat, then share a final answer.\n\n"
        "Available tools:\n"
        f"{tools_list}\n\n"
        "Scheduling guardrails:\n"
        "- Never assume which appointment should change; confirm with the user.\n"
        "- When multiple matches exist, list them and ask the user to choose.\n"
        "- Repeat the client and appointment details before booking, rescheduling, or cancelling.\n\n"
        "RESPONSE FORMAT (STRICT):\n"
        "• Tool call:\n"
        '{ "thought": "<brief reason>", "action": "<tool_name>", "action_input": { ... } }\n'
        "• Final answer:\n"
        '{ "final": "<natural language reply>" }\n'
        "Always return exactly one JSON object without extra narration."
    )

    agent_conversations[session_id] = [{"role": "system", "content": system_prompt}]
    return agent_conversations[session_id]


def _run_react_agent(openai_client, session_id: str, user_text: str, max_steps: int) -> str:
    """Run the ReAct loop to obtain a final answer capable of calling MCP tools."""
    messages = _prime_agent_session(session_id)
    messages.append({"role": "user", "content": user_text})

    final_text: Optional[str] = None
    for _ in range(max(1, max_steps)):
        response = _call_responses_api(messages)
        assistant_text = _response_to_text(response)
        if not assistant_text:
            assistant_text = str(response)

        print(f"🤖 WhatsApp MCP agent response: {assistant_text[:200]}...")
        messages.append({"role": "assistant", "content": assistant_text})

        control = _extract_json_object(assistant_text)
        if not control:
            print("⚠️ Model returned non-JSON control; reinforcing protocol.")
            messages.append({
                "role": "user",
                "content": (
                    "Follow the JSON-only protocol. Either call a tool:\n"
                    '{ "thought": "...", "action": "<tool_name>", "action_input": { ... } }\n'
                    "or provide the final answer:\n"
                    '{ "final": "<answer>" }'
                ),
            })
            continue

        if control.get("final"):
            final_text = str(control["final"]).strip()
            break

        action = control.get("action")
        action_input = control.get("action_input") or {}
        if not action:
            final_text = assistant_text
            break

        try:
            print(f"🔧 Invoking MCP tool '{action}' with args {action_input}")
            observation = asyncio.run(_mcp_call(action, action_input))
            print(f"✅ Tool '{action}' returned: {observation[:200]}...")
        except Exception as exc:
            error_msg = f"Tool error calling '{action}': {exc}"
            print(f"❌ {error_msg}")
            try:
                tool_schema = asyncio.run(_fetch_tool_schema(action))
            except Exception as schema_exc:
                tool_schema = None
                print(f"⚠️ Failed to fetch schema for '{action}': {schema_exc}")
            schema_hint = f"\n\nInput schema for '{action}':\n{json.dumps(tool_schema, indent=2)}" if tool_schema else ""
            observation = error_msg + schema_hint

        messages.append({
            "role": "user",
            "content": f"Tool '{action}' returned: {observation}",
        })

    if not final_text:
        final_text = (
            "I stopped before reaching a final answer because the step limit was hit. "
            "Share any missing appointment details and I'll try again."
        )

    messages.append({"role": "assistant", "content": final_text})
    return final_text


def _call_responses_api(messages: List[Dict[str, Any]]) -> Any:
    """
    Call the OpenAI Responses API.
    Falls back to a direct HTTP/1.1 requests call if the SDK hits a connection error
    (commonly seen on Render with HTTP/2).
    """
    try:
        return openai_client.responses.create(
            model=WHATSAPP_AGENT_MODEL,
            input=messages,
        )
    except Exception as exc:
        error_text = str(exc)
        if "Connection error" not in error_text and "ConnectionError" not in error_text:
            raise RuntimeError(f"OpenAI Responses API failed: {exc}") from exc

        print("⚠️  OpenAI SDK connection error; retrying with raw HTTP/1.1 request")
        try:
            resp = requests.post(
                "https://api.openai.com/v1/responses",
                headers={
                    "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": WHATSAPP_AGENT_MODEL,
                    "input": messages,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as http_exc:
            raise RuntimeError(f"OpenAI Responses API failed after HTTP fallback: {http_exc}") from http_exc


def _response_to_text(response: Any) -> str:
    """Extract assistant text from either an OpenAI SDK object or raw JSON dict."""
    if not response:
        return ""

    # Handle dict (raw JSON)
    if isinstance(response, dict):
        output_text = (response.get("output_text") or "").strip()
        if output_text:
            return output_text

        output = response.get("output") or []
        if output:
            first = output[0]
            if isinstance(first, dict):
                content = first.get("content") or []
                if content:
                    first_content = content[0]
                    if isinstance(first_content, dict):
                        text = (first_content.get("text") or "").strip()
                        if text:
                            return text
                    elif isinstance(first_content, str):
                        return first_content.strip()
        return ""

    # Handle SDK object
    try:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
    except Exception:
        pass

    try:
        first_output = response.output[0]
        first_content = first_output.content[0]
        text = (getattr(first_content, "text", "") or "").strip()
        if text:
            return text
    except Exception:
        pass

    return ""



@whatsapp_assistant_mcp_bp.route("/incoming-whatsapp", methods=["POST"])
def handle_incoming_whatsapp():
    """Handle incoming WhatsApp message and send AI-generated reply using Twilio SDK."""
    
    # Get the message from Twilio
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '')  # This will be in format 'whatsapp:+1234567890'
    to_number = request.values.get('To', '')
    
    print(f"Received WhatsApp from {from_number} to {to_number}: {incoming_msg}")
    
    # Get or create conversation history for this number
    if from_number not in conversations:
        conversations[from_number] = [
            {"role": "system", "content": SYSTEM_MESSAGE}
        ]
    
    # Add user message to conversation history
    conversations[from_number].append({
        "role": "user",
        "content": incoming_msg
    })
    
    # Keep only last 10 messages to avoid token limits
    if len(conversations[from_number]) > 11:  # 1 system + 10 messages
        conversations[from_number] = [conversations[from_number][0]] + conversations[from_number][-10:]
    
    try:
        # Generate AI response using local ReAct agent (gpt-5 + MCP tools)
        ai_message = _run_react_agent(
            openai_client=openai_client,
            session_id=from_number or "whatsapp_default",
            user_text=incoming_msg,
            max_steps=WHATSAPP_AGENT_MAX_STEPS,
        )
        
        # Limit message length for WhatsApp (4096 chars max per message)
        if len(ai_message) > 4096:
            ai_message = ai_message[:4093] + "..."
        
        # Add assistant response to conversation history
        conversations[from_number].append({
            "role": "assistant",
            "content": ai_message
        })
        
        print(f"AI Response to {from_number}: {ai_message}")
        
        # Send WhatsApp message using Twilio SDK
        # Note: from_number already includes 'whatsapp:' prefix from Twilio
        message = twilio_client.messages.create(
            body=ai_message,
            from_=Config.TWILIO_WHATSAPP_NUMBER,
            to=from_number  # from_number already has 'whatsapp:' prefix
        )
        
        print(f"WhatsApp message sent successfully! SID: {message.sid}, Status: {message.status}")
        
        return jsonify({
            "success": True,
            "message_sid": message.sid,
            "status": message.status
        }), 200
        
    except Exception as e:
        print(f"Error: {e}")
        
        # Try to send error message
        try:
            error_message = twilio_client.messages.create(
                body="Sorry, I'm having trouble processing your message right now. Please try again later.",
                from_=Config.TWILIO_WHATSAPP_NUMBER,
                to=from_number
            )
            print(f"Error message sent. SID: {error_message.sid}")
        except Exception as send_error:
            print(f"Failed to send error message: {send_error}")
        
        return jsonify({"success": False, "error": str(e)}), 500

@whatsapp_assistant_mcp_bp.route("/clear-conversation", methods=["POST"])
def clear_conversation():
    """Clear conversation history for a specific WhatsApp number."""
    from_number = request.json.get('phone_number', '')
    
    # Ensure whatsapp: prefix is added if not present
    if from_number and not from_number.startswith('whatsapp:'):
        from_number = f'whatsapp:{from_number}'
    
    cleared = False
    if from_number in conversations:
        del conversations[from_number]
        cleared = True
    if from_number in agent_conversations:
        del agent_conversations[from_number]
        cleared = True
    
    if cleared:
        return {"message": f"Conversation cleared for {from_number}"}, 200
    
    return {"message": "No conversation found for this number"}, 404

@whatsapp_assistant_mcp_bp.route("/clear-all-conversations", methods=["POST"])
def clear_all_conversations():
    """Clear all conversation histories."""
    conversations.clear()
    agent_conversations.clear()
    return {"message": "All conversations cleared"}, 200

@whatsapp_assistant_mcp_bp.route("/test-whatsapp", methods=["POST"])
def test_whatsapp():
    """Test endpoint with simple hardcoded response using Twilio SDK."""
    print("Test WhatsApp endpoint hit!")
    
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '')
    
    print(f"Received test WhatsApp from {from_number}: {incoming_msg}")
    
    try:
        # Send simple test message using Twilio SDK
        message = twilio_client.messages.create(
            body="Hello! This is a test response from your AI WhatsApp assistant.",
            from_=Config.TWILIO_WHATSAPP_NUMBER,
            to=from_number
        )
        
        print(f"Test message sent! SID: {message.sid}, Status: {message.status}")
        
        return jsonify({
            "success": True,
            "message_sid": message.sid,
            "status": message.status
        }), 200
        
    except Exception as e:
        print(f"Error sending test message: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

