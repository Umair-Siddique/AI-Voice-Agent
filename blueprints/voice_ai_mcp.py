import os
import sys
import json
import asyncio
import importlib.util
import traceback
from typing import Any, Dict, List, Optional

# Ensure project root is importable when run as a script
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local voice deps
try:
    import speech_recognition as sr
except Exception as _e:
    sr = None
try:
    import pyttsx3
except Exception as _e:
    pyttsx3 = None

from flask import Blueprint, jsonify, request, Response, current_app
from openai import OpenAI
from twilio.twiml.voice_response import VoiceResponse
from config import Config
from utils import SYSTEM_MESSAGE

voice_mcp_bp = Blueprint("voice_mcp", __name__)
_TWILIO_SESSIONS: Dict[str, Dict[str, Any]] = {}
_TWILIO_SAY_VOICE = os.getenv("VOICE_MCP_TWILIO_VOICE", "Google.en-US-Chirp3-HD-Aoede")
_DEFAULT_OPENAI_CLIENT: Optional[OpenAI] = None

# Optional MCP imports for tool calling (like mcp_server)
_MCP_AVAILABLE = False
_MCP_IMPORT_ERROR = ""
openai_client: Optional[OpenAI] = None
_build_mcp_tools_spec = None
_extract_json_object = None
_mcp_call = None
_mcp_tools_brief = None
_fetch_tool_schema = None
AGENT_MODEL: Optional[str] = None
AGENT_MAX_STEPS: Optional[int] = None
SIMPLYBOOK_MCP_URL: Optional[str] = None
try:
    if __package__:
        from . import mcp_server as _mcp_module
    else:
        module_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        spec = importlib.util.spec_from_file_location("voice_mcp_server", module_path)
        if not spec or not spec.loader:
            raise ImportError("Unable to load mcp_server module spec")
        _mcp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_mcp_module)

    openai_client = _mcp_module.openai_client
    _build_mcp_tools_spec = _mcp_module._build_mcp_tools_spec
    _extract_json_object = _mcp_module._extract_json_object
    _mcp_call = _mcp_module._mcp_call
    _mcp_tools_brief = _mcp_module._mcp_tools_brief
    _fetch_tool_schema = _mcp_module._fetch_tool_schema
    AGENT_MODEL = _mcp_module.AGENT_MODEL
    AGENT_MAX_STEPS = _mcp_module.AGENT_MAX_STEPS
    SIMPLYBOOK_MCP_URL = _mcp_module.SIMPLYBOOK_MCP_URL
    _MCP_AVAILABLE = True
except Exception as _e:
    _MCP_IMPORT_ERROR = str(_e)
    _MCP_AVAILABLE = False
    openai_client = None


def _get_openai_client() -> OpenAI:
    """
    Prefer the Flask-initialised OpenAI client (used by other blueprints) so
    credentials/config stay consistent in every deployment. Fall back to a
    lazily-created client for CLI/testing contexts where Flask isn't running.
    """
    global _DEFAULT_OPENAI_CLIENT
    try:
        if current_app:
            app_client = getattr(current_app, "openai_client", None)
            if app_client is not None:
                return app_client
    except RuntimeError:
        # No application context (CLI usage), fall back to local client.
        pass

    if _DEFAULT_OPENAI_CLIENT is None:
        if not Config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. Set it in your environment "
                "or provide it via Flask app config."
            )
        _DEFAULT_OPENAI_CLIENT = OpenAI(api_key=Config.OPENAI_API_KEY)
    return _DEFAULT_OPENAI_CLIENT


def _log_openai_exception(context: str, exc: Exception) -> None:
    """
    Emit detailed diagnostics for OpenAI client failures so production logs
    clarify whether the issue is auth, networking, etc.
    """
    print(f"[OpenAI error] {context}: {exc} ({exc.__class__.__name__})")
    traceback.print_exc()


# Always align with the app-level OpenAI client so credentials and transports
# match the other voice/SMS blueprints.
openai_client = _get_openai_client()


@voice_mcp_bp.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "success": True,
            "message": "Simple Voice Assistant is ready. Run `python -m blueprints.voice_ai_mcp` or execute the file directly to test locally.",
        }
    )


def register_websocket_routes(sock, app):
    # Local CLI voice testing does not require websockets,
    # but this stub keeps app startup intact.
    return


class LocalTTS:
    def __init__(self) -> None:
        self.enabled = pyttsx3 is not None

    def say(self, text: str) -> None:
        if not self.enabled:
            print(f"[TTS disabled] {text}")
            return
        engine = None
        try:
            # Create a fresh engine each time to avoid SAPI5 state issues
            engine = pyttsx3.init()  # defaults to 'sapi5' on Windows
            try:
                engine.setProperty("rate", 200)
                engine.setProperty("volume", 1.0)
            except Exception:
                pass
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS error] {e}")
            print(f"[Assistant] {text}")
        finally:
            try:
                if engine is not None:
                    engine.stop()
            except Exception:
                pass


class LocalSTT:
    def __init__(self) -> None:
        self.enabled = sr is not None
        self.recognizer = None
        self.microphone = None

        if self.enabled:
            try:
                self.recognizer = sr.Recognizer()
                # Balanced end-of-speech detection for natural pauses
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 1.0
                self.recognizer.non_speaking_duration = 0.5
                self.recognizer.phrase_threshold = 0.3
                self._calibrated = False
                # Microphone requires PyAudio. If not available, this will raise.
                self.microphone = sr.Microphone()
            except Exception as e:
                print(f"[Microphone error] {e}")
                self.enabled = False
                self.recognizer = None
                self.microphone = None

    def listen_once(self, timeout: float = 4.0, phrase_time_limit: float = 12.0) -> Optional[str]:
        if not self.enabled or not self.recognizer or not self.microphone:
            return None
        try:
            with self.microphone as source:
                if not getattr(self, "_calibrated", False):
                    print("Adjusting for ambient noise...")
                    # One-time calibration
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.6)
                    self._calibrated = True
                print("Listening... (speak now)")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
        except sr.WaitTimeoutError:
            print("No speech detected (timeout).")
            return None
        except Exception as e:
            print(f"[Listen error] {e}")
            return None

        try:
            # Uses Google's free web API (internet required).
            text = self.recognizer.recognize_google(audio)
            return text.strip()
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            print(f"Speech API error: {e}")
        except Exception as e:
            print(f"[Recognition error] {e}")
        return None


class SimpleVoiceAgent:
    """
    Minimal chat-style agent: user speech -> GPT reply -> TTS.
    No MCP tools; pure conversation with the configured model.
    """

    def __init__(self, session_id: str = "local-voice") -> None:
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "system",
                "content": (
                    "Speak in ≤2 short sentences. Before acting on bookings, always confirm the "
                    "specific client and appointment time so there is zero ambiguity."
                ),
            },
        ]
        # Allow overriding to a faster model via env var if desired
        self.model = (os.getenv("GPT_FAST_MODEL") or Config.GPT_MODEL or "gpt-5")
        if not Config.OPENAI_API_KEY:
            print("⚠️  OPENAI_API_KEY is not set. Set it via environment or .env")
        self.client = _get_openai_client()

    def _prune_history(self, max_turns: int = 2) -> None:
        """
        Keep system prompts and last N user/assistant turns to reduce tokens/latency.
        """
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        convo = [m for m in self.messages if m.get("role") in ("user", "assistant")]
        # Keep only the last max_turns*2 (user+assistant) messages
        trimmed = convo[-(max_turns * 2):] if len(convo) > (max_turns * 2) else convo
        self.messages = system_msgs + trimmed

    def run_turn(self, user_text: str) -> str:
        # Trim history for speed
        self._prune_history(max_turns=2)
        self.messages.append({"role": "user", "content": user_text})
        try:
            # Prefer short, fast responses; avoid unsupported params
            response = self.client.responses.create(
                model=self.model,
                input=self.messages,
            )
        except Exception as exc:
            _log_openai_exception("SimpleVoiceAgent.run_turn", exc)
            return "I hit an error contacting the AI model. Please try again."

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

        self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text


def _print_banner() -> None:
    print("")
    print("=== Simple Local Voice Assistant ===")
    print("Speak when prompted. Say 'exit' or 'quit' to stop.")
    print("Tip: If mic fails, you'll be asked to type instead.")
    print("")


class ToolsAPIEnabledVoiceAgent:
    """
    Voice agent that registers the remote MCP server with the OpenAI Responses API.
    The platform will call MCP tools automatically. We also print a compact
    view of any tool-related items if present in the response.
    """

    def __init__(self, session_id: str = "local-voice") -> None:
        if not _MCP_AVAILABLE:
            raise RuntimeError("MCP is not available. Ensure blueprints.mcp_server is importable.")
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {
                "role": "system",
                "content": (
                    "Be concise and speakable. Use tools when helpful. Confirm exact client identity "
                    "and appointment slot before calling any reschedule, cancel, or booking tool."
                ),
            },
        ]
        # Use same model family as server
        self.model = AGENT_MODEL

    def _prune_history(self, max_turns: int = 3) -> None:
        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        convo = [m for m in self.messages if m.get("role") in ("user", "assistant")]
        trimmed = convo[-(max_turns * 2):] if len(convo) > (max_turns * 2) else convo
        self.messages = system_msgs + trimmed

    def _print_tool_items(self, response: Any) -> None:
        # Best-effort introspection to show tool activity
        try:
            # Some SDK builds have structured output
            outputs = getattr(response, "output", None)
            if outputs:
                for idx, out in enumerate(outputs):
                    items = getattr(out, "content", []) or []
                    for it in items:
                        t = getattr(it, "type", None) or ""
                        if "tool" in str(t).lower():
                            try:
                                name = getattr(it, "name", None) or getattr(it, "tool_name", None) or ""
                                print(f"🧰 Tool item detected (output[{idx}]): type={t} name={name}")
                            except Exception:
                                print(f"🧰 Tool item detected (output[{idx}]): type={t}")
                return
            # Fallback: print a compact JSON if possible
            to_json = getattr(response, "model_dump_json", None)
            if callable(to_json):
                raw = to_json()
                if isinstance(raw, str) and '"tool' in raw:
                    print("🧰 Tool-related fields present in response (truncated):")
                    print(raw[:800])
        except Exception:
            pass

    def run_turn(self, user_text: str) -> str:
        self._prune_history(max_turns=3)
        self.messages.append({"role": "user", "content": user_text})
        print("🧩 Registering MCP tools with the model (platform-managed)...")
        try:
            response = openai_client.responses.create(
                model=self.model,
                input=self.messages,
                tools=_build_mcp_tools_spec(),
            )
        except Exception as exc:
            _log_openai_exception("ToolsAPIEnabledVoiceAgent.run_turn", exc)
            return "I hit an error contacting the AI model. Please try again."

        # Try to report any tool-related items
        self._print_tool_items(response)

        assistant_text = ""
        try:
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text.strip():
                assistant_text = output_text.strip()
        except Exception:
            pass
        if not assistant_text:
            outputs = getattr(response, "output", None) or []
            for out in outputs:
                content_items = getattr(out, "content", []) or []
                for item in content_items:
                    text_val = getattr(item, "text", None)
                    if isinstance(text_val, str) and text_val.strip():
                        assistant_text = text_val.strip()
                        break
                if assistant_text:
                    break
        if not assistant_text:
            assistant_text = "I called the tools but did not get a readable reply. Please try again."

        self.messages.append({"role": "assistant", "content": assistant_text})
        return assistant_text


class MCPVoiceAgent:
    """
    ReAct-style agent loop with MCP tool calling (mirrors /mcp/react).
    Prints tool calls to the terminal.
    """

    def __init__(self, session_id: str = "local-voice", max_steps: int = None) -> None:
        if not _MCP_AVAILABLE:
            raise RuntimeError("MCP is not available. Ensure blueprints.mcp_server is importable.")
        self.session_id = session_id
        self.max_steps = max_steps or AGENT_MAX_STEPS
        self.messages: List[Dict[str, str]] = []
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        try:
            print(f"🔍 Discovering tools from MCP server: {SIMPLYBOOK_MCP_URL}")
            tools_list = asyncio.run(_mcp_tools_brief())
            print("✅ Discovered tools successfully")
        except Exception as exc:
            print(f"❌ Failed to discover tools: {exc}")
            tools_list = f"- (failed to load tools: {exc})"

        system_prompt = (
            SYSTEM_MESSAGE
            + "\n\n"
            + "You are a helpful AI agent that can achieve goals by planning and invoking tools "
            + "exposed by a SimplyBook MCP server. Use a tight loop: "
            + "think -> act (tool) -> observe -> repeat, then provide a final answer.\n\n"
            + "Available tools:\n"
            + f"{tools_list}\n\n"
            + "Scheduling safety:\n"
            + "- Never assume which appointment the user wants to change.\n"
            + "- If more than one booking matches their description, list the options, ask which client/time they mean, and wait for confirmation.\n"
            + "- Repeat the confirmed client + appointment details before calling any reschedule, cancel, or booking tool.\n\n"
            + "Interaction protocol (STRICT):\n"
            + "1) When you need to use a tool, respond with ONLY a single JSON object:\n"
            + '{ "thought": "very brief reason", "action": "<tool_name>", "action_input": { ... } }\n'
            + "2) When you are ready to answer the user, respond with ONLY:\n"
            + '{ "final": "<natural language answer>" }\n'
            + "Rules: Never include explanatory text outside of the JSON object. "
            + "Keep replies concise and speakable."
        )
        self.messages = [{"role": "system", "content": system_prompt}]
        self._initialized = True

    def run_turn(self, user_text: str) -> str:
        self._ensure_initialized()
        self.messages.append({"role": "user", "content": user_text})

        final_text: Optional[str] = None
        for _ in range(self.max_steps):
            # Ask the model what to do next
            try:
                response = openai_client.responses.create(
                    model=AGENT_MODEL,
                    input=self.messages,
                )
            except Exception as exc:
                _log_openai_exception("MCPVoiceAgent.run_turn", exc)
                return "I hit an error contacting the AI model. Please try again."

            # Extract assistant text
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
            self.messages.append({"role": "assistant", "content": assistant_text})

            # Parse control JSON
            control = _extract_json_object(assistant_text)
            if not control:
                print("⚠️  Model didn't return JSON control object; reminding protocol")
                self.messages.append({
                    "role": "user",
                    "content": (
                        "Invalid format. Respond with ONLY one JSON object per the protocol. "
                        'Either an action call:\n'
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
                final_text = assistant_text
                break

            # Execute tool and add observation
            try:
                print(f"🔧 Calling tool: {action} with args: {action_input}")
                observation = asyncio.run(_mcp_call(action, action_input))
                print(f"✅ Tool '{action}' succeeded: {str(observation)[:200]}...")
            except Exception as exc:
                error_msg = f"Tool error calling '{action}': {str(exc)}"
                print(f"❌ {error_msg}")
                try:
                    tool_schema = asyncio.run(_fetch_tool_schema(action))
                except Exception as schema_exc:
                    tool_schema = None
                    print(f"⚠️  Failed to fetch tool schema for '{action}': {schema_exc}")
                schema_hint = f"\n\nInput schema for '{action}':\n{json.dumps(tool_schema, indent=2)}" if tool_schema else ""
                observation = error_msg + schema_hint

            # Pass observation back to the model
            self.messages.append({
                "role": "user",
                "content": f"Tool '{action}' returned: {observation}",
            })

        if not final_text:
            final_text = (
                "I stopped before reaching a final answer due to the step limit. "
                "You can provide missing details and try again."
            )
        return final_text


def _create_voice_agent(session_id: str, emit_logs: bool = True):
    """
    Instantiate the appropriate agent for the given session, mirroring the CLI logic.
    """
    def _log(message: str) -> None:
        if emit_logs:
            print(message)

    use_mcp = (os.getenv("VOICE_USE_MCP", "1").lower() not in ("0", "false", "no"))
    mcp_mode = os.getenv("VOICE_MCP_MODE", "react").lower()
    agent = None

    if use_mcp and _MCP_AVAILABLE:
        if mcp_mode == "react":
            _log("Mode: MCP tools ENABLED (ReAct loop).")
            try:
                agent = MCPVoiceAgent(session_id=session_id)
            except Exception as e:
                _log(f"Failed to initialize MCP ReAct agent, falling back to platform tools: {e}")
                try:
                    agent = ToolsAPIEnabledVoiceAgent(session_id=session_id)
                except Exception as e2:
                    _log(f"Failed to initialize platform tools agent, falling back to chat-only: {e2}")
                    agent = SimpleVoiceAgent(session_id=session_id)
        else:
            _log("Mode: MCP tools ENABLED (platform-managed tools).")
            try:
                agent = ToolsAPIEnabledVoiceAgent(session_id=session_id)
            except Exception as e:
                _log(f"Failed to initialize platform tools agent, falling back to ReAct: {e}")
                try:
                    agent = MCPVoiceAgent(session_id=session_id)
                except Exception as e2:
                    _log(f"Failed to initialize MCP ReAct agent, falling back to chat-only: {e2}")
                    agent = SimpleVoiceAgent(session_id=session_id)
    else:
        if use_mcp and not _MCP_AVAILABLE:
            reason = f" (import error: {_MCP_IMPORT_ERROR})" if _MCP_IMPORT_ERROR else ""
            _log(f"MCP requested but not available{reason}; falling back to chat-only.")
        else:
            _log("Mode: Chat-only (no tools).")
        agent = SimpleVoiceAgent(session_id=session_id)

    return agent


def _get_twilio_session(call_sid: str) -> Dict[str, Any]:
    """
    Return or create the state for a Twilio-driven conversation.
    """
    session = _TWILIO_SESSIONS.get(call_sid)
    if session:
        return session

    agent = _create_voice_agent(session_id=f"twilio-{call_sid}", emit_logs=False)
    session = {"agent": agent, "greeted": False}
    _TWILIO_SESSIONS[call_sid] = session
    return session


def _cleanup_twilio_session(call_sid: str) -> None:
    """
    Remove cached session data for a finished call.
    """
    _TWILIO_SESSIONS.pop(call_sid, None)


def _enqueue_twilio_gather(response_obj: VoiceResponse, action_url: str, greeted: bool) -> None:
    """
    Append a Gather prompt that keeps the conversation going.
    """
    gather = response_obj.gather(
        input="speech",
        action=action_url,
        method="POST",
        speech_timeout="auto",
    )
    if not greeted:
        gather.say(
            "Hi! You're connected to Flexbody Solution's assistant. "
            "Please tell me the exact appointment or question you have.",
            voice=_TWILIO_SAY_VOICE,
        )
    else:
        gather.say(
            "You can share more booking details or say exit to finish.",
            voice=_TWILIO_SAY_VOICE,
        )


def _twilio_xml_response(voice_response: VoiceResponse, status: int = 200) -> Response:
    """
    Wrap a VoiceResponse in a Flask Response object.
    """
    return Response(str(voice_response), mimetype="application/xml", status=status)


@voice_mcp_bp.route("/twilio/voice", methods=["GET", "POST"])
def twilio_voice_webhook():
    """
    Twilio Programmable Voice webhook that funnels speech into the MCP agents.
    """
    if request.method == "GET":
        return jsonify({
            "success": True,
            "message": "Twilio webhook ready. Configure this URL as your Voice webhook.",
        })

    call_sid = (request.values.get("CallSid") or "").strip()
    resp = VoiceResponse()

    if not call_sid:
        resp.say("Missing call identifier. Please try again later.", voice=_TWILIO_SAY_VOICE)
        resp.hangup()
        return _twilio_xml_response(resp, status=400)

    call_status = (request.values.get("CallStatus") or "").lower()
    if call_status and call_status in {"completed", "canceled", "failed", "busy", "no-answer"}:
        _cleanup_twilio_session(call_sid)

    session = _get_twilio_session(call_sid)
    user_text = (
        request.values.get("SpeechResult")
        or request.values.get("TranscriptionText")
        or request.values.get("Digits")
        or ""
    ).strip()

    action_url = request.url

    if not user_text:
        _enqueue_twilio_gather(resp, action_url, greeted=session.get("greeted", False))
        session["greeted"] = True
        return _twilio_xml_response(resp)

    normalized = user_text.lower()
    if normalized in {"exit", "quit", "stop", "bye", "goodbye"}:
        resp.say("Thanks for calling Flexbody Solution. Goodbye!", voice=_TWILIO_SAY_VOICE)
        resp.hangup()
        _cleanup_twilio_session(call_sid)
        return _twilio_xml_response(resp)

    agent = session["agent"]
    reply = agent.run_turn(user_text)
    resp.say(reply, voice=_TWILIO_SAY_VOICE)
    _enqueue_twilio_gather(resp, action_url, greeted=True)

    return _twilio_xml_response(resp)


def run_cli() -> None:
    _print_banner()

    tts = LocalTTS()
    stt = LocalSTT()

    agent = _create_voice_agent(session_id="cli-voice", emit_logs=True)

    # If mic unavailable, fall back to text mode
    use_text_fallback = not stt.enabled
    if use_text_fallback:
        print("Microphone not available. Falling back to text input mode.")

    while True:
        try:
            if use_text_fallback:
                user_text = input("You (type): ").strip()
                if not user_text:
                    continue
            else:
                heard = stt.listen_once()
                if not heard:
                    # If no audio captured, loop again
                    continue
                user_text = heard

            print(f"You said: {user_text}")
            if user_text.lower() in {"exit", "quit", "stop"}:
                print("Goodbye!")
                break

            reply = agent.run_turn(user_text)
            print(f"Assistant: {reply}")
            tts.say(reply)
        except KeyboardInterrupt:
            print("\nInterrupted. Goodbye!")
            break
        except Exception as e:
            print(f"[Loop error] {e}")


