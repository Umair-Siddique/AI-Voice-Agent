import os
import re
import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request
from openai import OpenAI
from fastmcp import Client as FastMCPClient
from utils import SYSTEM_MESSAGE
from config import Config

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️  WARNING: OPENAI_API_KEY is not set. Set it in your environment.")

openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

# Remote MCP server configuration (FastMCP over SSE).
# These values are used to register the MCP server as a tool with the
# OpenAI Responses API so the model can call tools directly.
SIMPLYBOOK_MCP_URL = os.getenv("SIMPLYBOOK_MCP_URL", "https://simplybook-mcp-server.onrender.com/sse")
SIMPLYBOOK_MCP_LABEL = os.getenv("SIMPLYBOOK_MCP_LABEL", "simplybook")
# Optional: JSON object for headers to include when calling the MCP server
# e.g. {"Authorization": "Bearer <token>"}
SIMPLYBOOK_MCP_HEADERS_JSON = os.getenv("SIMPLYBOOK_MCP_HEADERS_JSON", "").strip()
# Optional: approval policy for tool calls. Common values in docs include "auto"
# (allow automatically) or "always" (require explicit approval). If omitted,
# the platform default will apply.
SIMPLYBOOK_MCP_REQUIRE_APPROVAL = os.getenv("SIMPLYBOOK_MCP_REQUIRE_APPROVAL", "").strip()

AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "15"))
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-5")

def _build_mcp_tools_spec() -> List[Dict[str, Any]]:
    """
    Build the Responses API 'tools' specification for a remote MCP server.

    This follows the OpenAI 'Connectors and MCP servers' guide. The exact field
    names here match current public docs; unknown/empty optional values are
    omitted to maintain compatibility across SDK versions.
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
    if SIMPLYBOOK_MCP_REQUIRE_APPROVAL:
        tool["require_approval"] = SIMPLYBOOK_MCP_REQUIRE_APPROVAL

    return [tool]

mcp_bp = Blueprint("mcp_bp", __name__)
conversations: Dict[str, List[Dict[str, Any]]] = {}
agent_conversations: Dict[str, List[Dict[str, Any]]] = {}
agent_cached_tool_schemas: Dict[str, Dict[str, Any]] = {}


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
    """
    Call a tool on the FastMCP server and return a printable string result.
    """
    # Strip trailing slash for consistency
    mcp_url = SIMPLYBOOK_MCP_URL.rstrip("/")
    
    try:
        # FastMCP Client doesn't support headers in constructor
        # If authentication is needed, it should be handled at the MCP server level
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
    # FastMCP Client doesn't support headers in constructor
    # If authentication is needed, it should be handled at the MCP server level
    client = FastMCPClient(mcp_url)
    lines: List[str] = []
    async with client:
        # Wait a moment for SSE connection to establish
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

    # Try direct parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Look for fenced code block ```json ... ```
    fenced = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    # Fallback: find first {...} greedily
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
    """
    Return the input schema for a given tool name by querying the MCP server.
    Cached per-session in agent_cached_tool_schemas when possible.
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


@mcp_bp.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_text = (data.get("text") or "").strip()
    session_id = data.get("session_id", "default")

    if not user_text:
        return jsonify({"success": False, "error": "Missing 'text' in body"}), 400

    if session_id not in conversations:
        conversations[session_id] = [{
            "role": "system",
            "content": SYSTEM_MESSAGE,
        }]

    conversations[session_id].append({"role": "user", "content": user_text})
    messages = conversations[session_id]

    # Attempt to register the remote MCP server so the model can call tools.
    # If this fails (e.g., due to an older SDK), gracefully fall back to plain chat.
    try:
        response = openai_client.responses.create(
            model="gpt-5",
            input=messages,
            tools=_build_mcp_tools_spec(),
        )
    except Exception as exc:
        print(f"⚠️  MCP tools call failed, falling back without tools: {exc}")
        try:
            response = openai_client.responses.create(
                model="gpt-5",
                input=messages,
            )
        except Exception as inner_exc:
            return jsonify({"success": False, "error": f"{inner_exc}"}), 500

    # Prefer the unified 'output_text' if available; fall back to structured output.
    final_text = ""
    try:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            final_text = output_text.strip()
    except Exception:
        pass

    if not final_text:
        try:
            first_output = response.output[0]
            first_content = first_output.content[0]
            final_text = (getattr(first_content, "text", "") or "").strip()
        except Exception:
            final_text = str(response)

    conversations[session_id].append({"role": "assistant", "content": final_text})
    return jsonify({"success": True, "response": final_text}), 200


@mcp_bp.route("/react", methods=["POST"])
def chat_react():
    """
    ReAct-style agent loop that:
    - Asks the model to emit strict JSON for tool actions or final answers.
    - Calls MCP tools locally via FastMCP.
    - Iterates until a final answer is produced or max steps reached.
    """
    data = request.get_json() or {}
    user_text = (data.get("text") or "").strip()
    session_id = data.get("session_id", "default")
    max_steps = int(data.get("max_steps", AGENT_MAX_STEPS))

    if not user_text:
        return jsonify({"success": False, "error": "Missing 'text' in body"}), 400

    # Build/prime the session with ReAct protocol
    if session_id not in agent_conversations:
        # Discover tools list for better guidance
        try:
            print(f"🔍 Discovering tools from MCP server: {SIMPLYBOOK_MCP_URL}")
            tools_list = asyncio.run(_mcp_tools_brief())
            print(f"✅ Discovered tools successfully")
        except Exception as exc:
            print(f"❌ Failed to discover tools: {exc}")
            tools_list = f"- (failed to load tools: {exc})"

        system_prompt = (
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
            '{\n'
            '  "thought": "very brief reason",\n'
            '  "action": "<tool_name>",\n'
            '  "action_input": { /* JSON arguments for the tool */ }\n'
            "}\n"
            "2) When you are ready to answer the user, respond with ONLY:\n"
            '{\n'
            '  "final": "<natural language answer>"\n'
            "}\n"
            "Rules:\n"
            "- Never include explanatory text outside of the JSON object.\n"
            "- Keep 'thought' short. Output human-like, concise, helpful 'final'."
        )
        agent_conversations[session_id] = [{"role": "system", "content": system_prompt}]

    # Append the new user turn
    agent_conversations[session_id].append({"role": "user", "content": user_text})
    messages = agent_conversations[session_id]

    final_text: Optional[str] = None
    for _ in range(max_steps):
        # Ask the model what to do next
        try:
            response = openai_client.responses.create(
                model=AGENT_MODEL,
                input=messages,
            )
        except Exception as exc:
            return jsonify({"success": False, "error": f"{exc}"}), 500

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
                "content": (
                    "Invalid format. Respond with ONLY one JSON object per the protocol. "
                    "Either an action call:\n"
                    '{ "thought": "...", "action": "<tool_name>", "action_input": { ... } }\n'
                    "or a final answer:\n"
                    '{ "final": "<answer>" }'
                ),
            })
            continue

        if "final" in control and control["final"]:
            final_text = str(control["final"]).strip()
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

        # Pass observation back to the model as a user message (Responses API doesn't support 'tool' role)
        messages.append({
            "role": "user",
            "content": f"Tool '{action}' returned: {observation}",
        })

    if not final_text:
        final_text = "I stopped before reaching a final answer due to the step limit. You can provide missing details (e.g., provider_id or date) and try again."

    agent_conversations[session_id].append({"role": "assistant", "content": final_text})
    return jsonify({"success": True, "response": final_text}), 200