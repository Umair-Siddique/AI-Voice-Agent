import os
from flask import Blueprint, request, jsonify
from twilio.rest import Client
from openai import OpenAI
from threading import Thread

from config import Config
from utils import SYSTEM_MESSAGE

# Initialize Twilio and OpenAI clients
twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

whatsapp_assistant_mcp_bp = Blueprint('whatsappmcp', __name__)

# Simple in-memory conversation storage (for production, use a database)
conversations = {}

WHATSAPP_AGENT_MODEL = os.getenv("WHATSAPP_AGENT_MODEL", "gpt-4o")


def process_and_respond(from_number: str, to_number: str, incoming_msg: str):
    """
    Process the message and send response in a background thread.
    This runs AFTER we've already responded to Twilio with 200 OK.
    """
    try:
        print("=" * 80)
        print(f"🔄 Background processing started")
        print(f"📱 From: {from_number}")
        print(f"💬 Message: {incoming_msg}")
        print("=" * 80)
        
        # Get or create conversation history
        if from_number not in conversations:
            conversations[from_number] = [
                {"role": "system", "content": SYSTEM_MESSAGE}
            ]
        
        # Add user message
        conversations[from_number].append({
            "role": "user",
            "content": incoming_msg
        })
        
        # Keep only last 10 messages
        if len(conversations[from_number]) > 11:
            conversations[from_number] = [conversations[from_number][0]] + conversations[from_number][-10:]
        
        print(f"🤖 Calling OpenAI API...")
        
        # Call OpenAI
        response = openai_client.chat.completions.create(
            model=WHATSAPP_AGENT_MODEL,
            messages=conversations[from_number],
            temperature=0.7,
            max_tokens=500,
            timeout=30,
        )
        
        ai_message = response.choices[0].message.content.strip()
        
        print(f"✅ OpenAI response received: {ai_message[:100]}...")
        
        # Limit message length for WhatsApp
        if len(ai_message) > 4096:
            ai_message = ai_message[:4093] + "..."
        
        # Add to conversation history
        conversations[from_number].append({
            "role": "assistant",
            "content": ai_message
        })
        
        print(f"📤 Sending WhatsApp reply...")
        
        # Send WhatsApp message
        message = twilio_client.messages.create(
            body=ai_message,
            from_=Config.TWILIO_WHATSAPP_NUMBER,
            to=from_number
        )
        
        print(f"✅ WhatsApp sent! SID: {message.sid}, Status: {message.status}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error in background processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to send error message
        try:
            twilio_client.messages.create(
                body="Sorry, I'm having trouble processing your message. Please try again.",
                from_=Config.TWILIO_WHATSAPP_NUMBER,
                to=from_number
            )
        except Exception as send_error:
            print(f"❌ Failed to send error message: {send_error}")
        
        print("=" * 80)


@whatsapp_assistant_mcp_bp.route("/incoming-whatsapp", methods=["POST"])
def handle_incoming_whatsapp():
    """
    Handle incoming WhatsApp message.
    Responds immediately to Twilio, then processes message in background.
    """
    print("=" * 80)
    print("📥 WEBHOOK HIT - Responding immediately to Twilio")
    print("=" * 80)
    
    # Get the message from Twilio
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '')
    to_number = request.values.get('To', '')
    
    print(f"📱 From: {from_number}")
    print(f"📱 To: {to_number}")
    print(f"💬 Message: {incoming_msg}")
    
    if not incoming_msg or not from_number:
        print("❌ Missing required fields")
        return jsonify({"success": False, "error": "Missing Body or From"}), 400
    
    # Start background thread to process and respond
    thread = Thread(
        target=process_and_respond,
        args=(from_number, to_number, incoming_msg),
        daemon=True
    )
    thread.start()
    
    print("🚀 Background thread started, responding 200 OK to Twilio")
    print("=" * 80)
    
    # Respond immediately to Twilio with 200 OK
    # This prevents the 15-second timeout
    return jsonify({"success": True, "status": "processing"}), 200


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
