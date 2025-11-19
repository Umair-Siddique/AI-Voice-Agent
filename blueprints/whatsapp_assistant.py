import os
from flask import Blueprint, request, current_app, Response, jsonify
from twilio.rest import Client
from dotenv import load_dotenv

load_dotenv()

# Configuration
SYSTEM_MESSAGE = (
    "You are a Conssultant in Legal Domain with name Aima."
    "Answer to user questio carefully."
)

# Initialize Twilio client
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN')
TWILIO_WHATSAPP_NUMBER = 'whatsapp:+14155238886'  # Twilio WhatsApp trial number

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

whatsapp_assistant_bp = Blueprint('whatsapp_assistant', __name__)

# Simple in-memory conversation storage (for production, use a database)
conversations = {}



@whatsapp_assistant_bp.route("/incoming-whatsapp", methods=["POST"])
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
        # Get OpenAI client from app
        openai_client = current_app.openai_client
        
        # Generate AI response
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Use gpt-4o-mini for cost-effective WhatsApp responses
            messages=conversations[from_number],
            max_tokens=200,  # Slightly more than SMS since WhatsApp supports longer messages
            temperature=0.8
        )
        
        ai_message = response.choices[0].message.content.strip()
        
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
            from_=TWILIO_WHATSAPP_NUMBER,
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
                from_=TWILIO_WHATSAPP_NUMBER,
                to=from_number
            )
            print(f"Error message sent. SID: {error_message.sid}")
        except Exception as send_error:
            print(f"Failed to send error message: {send_error}")
        
        return jsonify({"success": False, "error": str(e)}), 500

@whatsapp_assistant_bp.route("/clear-conversation", methods=["POST"])
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

@whatsapp_assistant_bp.route("/clear-all-conversations", methods=["POST"])
def clear_all_conversations():
    """Clear all conversation histories."""
    conversations.clear()
    return {"message": "All conversations cleared"}, 200

@whatsapp_assistant_bp.route("/test-whatsapp", methods=["POST"])
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
            from_=TWILIO_WHATSAPP_NUMBER,
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

