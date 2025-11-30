import os
import json
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from flask import Blueprint, request, jsonify
from twilio.rest import Client
from openai import OpenAI

from config import Config
from utils import SYSTEM_MESSAGE

# Initialize Twilio and OpenAI clients
twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

whatsapp_assistant_mcp_bp = Blueprint('whatsappmcp', __name__)

# Simple in-memory conversation storage (for production, use a database)
conversations = {}

# Thread pool for running blocking OpenAI calls outside gevent
executor = ThreadPoolExecutor(max_workers=5)

WHATSAPP_AGENT_MAX_STEPS = int(os.getenv("WHATSAPP_AGENT_MAX_STEPS", "5"))
WHATSAPP_AGENT_MODEL = os.getenv("WHATSAPP_AGENT_MODEL", "gpt-4o")  # Fallback to gpt-4o since gpt-5 might not be available


def call_openai_sync(messages: List[Dict[str, Any]]) -> str:
    """
    Call OpenAI in a blocking manner. This will be run in a thread pool.
    Returns the assistant's text response.
    """
    print(f"🔄 Calling OpenAI API...")
    print(f"📊 Model: {WHATSAPP_AGENT_MODEL}")
    print(f"📊 Messages count: {len(messages)}")
    
    try:
        response = openai_client.chat.completions.create(
            model=WHATSAPP_AGENT_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=500,
        )
        result = response.choices[0].message.content.strip()
        print(f"✅ OpenAI API call successful")
        return result
    except Exception as exc:
        print(f"❌ OpenAI API call failed: {exc}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"OpenAI API call failed: {exc}")


def simple_react_loop(user_text: str, session_id: str, max_steps: int = 5) -> str:
    """
    Simplified ReAct loop without MCP tools for now.
    Just returns a conversational response.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant. "
                "Respond naturally to user queries about services, appointments, or general questions. "
                "Be concise and friendly."
            )
        },
        {"role": "user", "content": user_text}
    ]
    
    try:
        # Run in thread pool to avoid gevent issues
        future = executor.submit(call_openai_sync, messages)
        response = future.result(timeout=60)  # 60 second timeout
        return response
    except FuturesTimeoutError:
        return "Sorry, the request took too long. Please try again."
    except Exception as exc:
        print(f"Error in ReAct loop: {exc}")
        return "Sorry, I encountered an error processing your request."


@whatsapp_assistant_mcp_bp.route("/incoming-whatsapp", methods=["POST"])
def handle_incoming_whatsapp():
    """Handle incoming WhatsApp message and send AI-generated reply using Twilio SDK."""
    
    print("=" * 80)
    print("📥 INCOMING WHATSAPP WEBHOOK HIT")
    print("=" * 80)
    
    # Get the message from Twilio
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '')  # This will be in format 'whatsapp:+1234567890'
    to_number = request.values.get('To', '')
    
    print(f"📱 From: {from_number}")
    print(f"📱 To: {to_number}")
    print(f"💬 Message: {incoming_msg}")
    print(f"🔑 OPENAI_API_KEY present: {bool(Config.OPENAI_API_KEY)}")
    print(f"🔑 TWILIO credentials present: {bool(Config.TWILIO_ACCOUNT_SID and Config.TWILIO_AUTH_TOKEN)}")
    
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
    
    ai_message = None
    try:
        # Generate AI response using simple agent
        print(f"🤖 Starting AI response generation...")
        print(f"📊 Current conversation length: {len(conversations.get(from_number, []))}")
        
        ai_message = simple_react_loop(
            user_text=incoming_msg,
            session_id=from_number or "whatsapp_default",
            max_steps=WHATSAPP_AGENT_MAX_STEPS,
        )
        
        print(f"✅ AI Response generated successfully")
        print(f"📝 Response: {ai_message[:200]}...")
        
        # Limit message length for WhatsApp (4096 chars max per message)
        if len(ai_message) > 4096:
            ai_message = ai_message[:4093] + "..."
        
        # Add assistant response to conversation history
        conversations[from_number].append({
            "role": "assistant",
            "content": ai_message
        })
        
        print(f"💾 Response saved to conversation history")
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        import traceback
        traceback.print_exc()
        ai_message = "Sorry, I'm having trouble processing your message right now. Please try again later."
    
    # Send WhatsApp message using Twilio SDK (do this regardless of AI success)
    try:
        print(f"📤 Attempting to send WhatsApp message...")
        print(f"📤 From: {Config.TWILIO_WHATSAPP_NUMBER}")
        print(f"📤 To: {from_number}")
        print(f"📤 Body length: {len(ai_message)} chars")
        
        message = twilio_client.messages.create(
            body=ai_message,
            from_=Config.TWILIO_WHATSAPP_NUMBER,
            to=from_number  # from_number already has 'whatsapp:' prefix
        )
        
        print(f"✅ WhatsApp message sent successfully!")
        print(f"📋 SID: {message.sid}")
        print(f"📊 Status: {message.status}")
        print("=" * 80)
        
        return jsonify({
            "success": True,
            "message_sid": message.sid,
            "status": message.status
        }), 200
        
    except Exception as e:
        print(f"❌ Failed to send WhatsApp message: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return jsonify({"success": False, "error": str(e)}), 500


@whatsapp_assistant_mcp_bp.route("/clear-conversation", methods=["POST"])
def clear_conversation():
    """Clear conversation history for a specific WhatsApp number."""
    from_number = request.json.get('phone_number', '')
    
    # Ensure whatsapp: prefix is added if not present
    if from_number and not from_number.startswith('whatsapp:'):
        from_number = f'whatsapp:{from_number}'
    
    if from_number in conversations:
        del conversations[from_number]
        return {"message": f"Conversation cleared for {from_number}"}, 200
    
    return {"message": "No conversation found for this number"}, 404


@whatsapp_assistant_mcp_bp.route("/clear-all-conversations", methods=["POST"])
def clear_all_conversations():
    """Clear all conversation histories."""
    conversations.clear()
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
