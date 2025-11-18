import os
from flask import Blueprint, request, current_app, Response, jsonify
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

# Configuration
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate. "
    "Keep your responses concise and friendly since this is SMS."
)

# Initialize Twilio client
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER')

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

sms_assistant_bp = Blueprint('sms_assistant', __name__)

# Simple in-memory conversation storage (for production, use a database)
conversations = {}

@sms_assistant_bp.route("/", methods=["GET"])
def index_page():
    return {"message": "Twilio SMS Assistant Server is running!"}

@sms_assistant_bp.route("/incoming-sms", methods=["POST"])
def handle_incoming_sms():
    """Handle incoming SMS and send AI-generated reply using Twilio SDK."""
    
    # Get the message from Twilio
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '')
    to_number = request.values.get('To', '')
    
    print(f"Received SMS from {from_number} to {to_number}: {incoming_msg}")
    
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
        # Get OpenAI client from app
        openai_client = current_app.openai_client
        
        # Generate AI response
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use gpt-4o-mini for cost-effective SMS responses
            messages=conversations[from_number],
            max_tokens=150,  # Reduced for SMS
            temperature=0.8
        )
        
        ai_message = response.choices[0].message.content.strip()
        
        # Limit message length for SMS (160 chars recommended, 1600 max)
        if len(ai_message) > 1600:
            ai_message = ai_message[:1597] + "..."
        
        # Add assistant response to conversation history
        conversations[from_number].append({
            "role": "assistant",
            "content": ai_message
        })
        
        print(f"AI Response to {from_number}: {ai_message}")
        
        # Send SMS using Twilio SDK
        message = twilio_client.messages.create(
            body=ai_message,
            from_=TWILIO_PHONE_NUMBER,
            to=from_number
        )
        
        print(f"Message sent successfully! SID: {message.sid}, Status: {message.status}")
        
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
                from_=TWILIO_PHONE_NUMBER,
                to=from_number
            )
            print(f"Error message sent. SID: {error_message.sid}")
        except Exception as send_error:
            print(f"Failed to send error message: {send_error}")
        
        return jsonify({"success": False, "error": str(e)}), 500

@sms_assistant_bp.route("/clear-conversation", methods=["POST"])
def clear_conversation():
    """Clear conversation history for a specific phone number."""
    from_number = request.json.get('phone_number', '')
    
    if from_number in conversations:
        del conversations[from_number]
        return {"message": f"Conversation cleared for {from_number}"}, 200
    
    return {"message": "No conversation found for this number"}, 404

@sms_assistant_bp.route("/clear-all-conversations", methods=["POST"])
def clear_all_conversations():
    """Clear all conversation histories."""
    conversations.clear()
    return {"message": "All conversations cleared"}, 200

@sms_assistant_bp.route("/test-sms", methods=["POST"])
def test_sms():
    """Test endpoint with simple hardcoded response using Twilio SDK."""
    print("Test SMS endpoint hit!")
    
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '')
    
    print(f"Received test SMS from {from_number}: {incoming_msg}")
    
    try:
        # Send simple test message using Twilio SDK
        message = twilio_client.messages.create(
            body="Hello! This is a test response from your AI assistant.",
            from_=TWILIO_PHONE_NUMBER,
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

