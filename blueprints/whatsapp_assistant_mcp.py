import os
import requests
from flask import Blueprint, request, jsonify
from twilio.rest import Client

from config import Config
from utils import SYSTEM_MESSAGE

# Initialize Twilio client
twilio_client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

whatsapp_assistant_mcp_bp = Blueprint('whatsappmcp', __name__)

# Simple in-memory conversation storage (for production, use a database)
conversations = {}

# MCP server configuration
WHATSAPP_AGENT_MAX_STEPS = int(
    os.getenv("WHATSAPP_AGENT_MAX_STEPS", os.getenv("AGENT_MAX_STEPS", "15"))
)

# Internal MCP ReAct endpoint (your existing /mcp/react that works locally)
INTERNAL_MCP_ENDPOINT = os.getenv(
    "INTERNAL_MCP_ENDPOINT",
    "http://localhost:5000/mcp/react"  # On Render, set this to https://ai-voice-agent-wt2m.onrender.com/mcp/react
)


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
        # Call the internal MCP ReAct endpoint (which already works locally)
        print(f"🔄 Proxying to internal MCP endpoint: {INTERNAL_MCP_ENDPOINT}")
        
        response = requests.post(
            INTERNAL_MCP_ENDPOINT,
            json={
                "text": incoming_msg,
                "session_id": from_number or "whatsapp_default",
                "max_steps": WHATSAPP_AGENT_MAX_STEPS,
            },
            timeout=120,  # Give MCP tools enough time
        )
        response.raise_for_status()
        
        result = response.json()
        if not result.get("success"):
            raise Exception(f"MCP endpoint returned error: {result.get('error', 'Unknown error')}")
        
        ai_message = result.get("response", "Sorry, I couldn't process that.")
        
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
        error_msg = str(e)
        print(f"Error: {error_msg}")
        
        # Send simple error message (don't try to be fancy to avoid recursion)
        simple_error = "Sorry, I'm having trouble right now. Please try again later."
        
        try:
            error_message = twilio_client.messages.create(
                body=simple_error,
                from_=Config.TWILIO_WHATSAPP_NUMBER,
                to=from_number
            )
            print(f"Error message sent. SID: {error_message.sid}")
        except Exception as send_error:
            print(f"Failed to send error message: {send_error}")
        
        return jsonify({"success": False, "error": error_msg}), 500


@whatsapp_assistant_mcp_bp.route("/clear-conversation", methods=["POST"])
def clear_conversation():
    """Clear conversation history for a specific WhatsApp number."""
    from_number = request.json.get('phone_number', '')
    
    # Ensure whatsapp: prefix is added if not present
    if from_number and not from_number.startswith('whatsapp:'):
        from_number = f'whatsapp:{from_number}'
    
    if from_number in conversations:
        del conversations[from_number]
        
        # Also clear the MCP ReAct session by calling the internal endpoint
        try:
            # You'd need to add a clear endpoint to mcp_server.py if you want this
            pass
        except Exception:
            pass
        
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
            body="Hello! This is a test response from your AI WhatsApp assistant with MCP tools.",
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
