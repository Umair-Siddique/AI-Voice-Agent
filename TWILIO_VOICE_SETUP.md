# Twilio ConversationRelay Voice AI Setup Guide

## Quick Start

Your voice AI assistant is ready to use with Twilio ConversationRelay! Here's how to test it:

## Step 1: Start Your Server

```bash
python run.py
```

The server will start on `http://localhost:5001`

**Test the endpoint:**
```bash
curl http://localhost:5001/voicemcp/test
```

You should see:
```json
{
  "status": "ok",
  "message": "Voice AI MCP endpoint is working",
  "openai_configured": true,
  "python_version": "3.x.x",
  "active_sessions": 0
}
```

## Step 2: Expose Your Server with ngrok

Open a new terminal and run:

```bash
ngrok http 5001
```

You'll see output like:
```
Forwarding    https://abc123.ngrok-free.app -> http://localhost:5001
```

**Copy the `https://` URL** (e.g., `https://abc123.ngrok-free.app`)

**Test the public endpoint:**
```bash
curl https://abc123.ngrok-free.app/voicemcp/test
```

## Step 3: Configure Your Twilio Phone Number

1. Go to [Twilio Console](https://console.twilio.com/)
2. Navigate to **Phone Numbers** → **Manage** → **Active Numbers**
3. Click on your Twilio phone number
4. Scroll down to **Voice Configuration**
5. Under **"A CALL COMES IN"**, configure:
   - **Webhook**: `https://abc123.ngrok-free.app/voicemcp/twiml`
   - **HTTP Method**: `POST`
6. Click **Save**

## Step 4: Test Your Voice AI

1. Call your Twilio phone number from any phone
2. You'll hear: "Hello! I'm your AI assistant. How can I help you today?"
3. Start talking! The AI will:
   - Listen to your speech (Twilio STT)
   - Generate intelligent responses (OpenAI)
   - Speak back to you (Twilio TTS)

## Troubleshooting

### "Application error has occurred" when calling?

**1. Check server logs**
Look for these log messages:
- `[TwiML] Incoming call, WebSocket URL: wss://...`
- `[WebSocket] New connection established`
- `[WebSocket] Received event: setup`

**2. Verify ngrok is running**
```bash
curl https://your-ngrok-url.ngrok-free.app/voicemcp/test
```

**3. Check WebSocket URL format**
The logs should show:
```
[TwiML] Incoming call, WebSocket URL: wss://abc123.ngrok-free.app/voicemcp/websocket
```

Note: Must be `wss://` (secure WebSocket), not `ws://`

**4. Verify OpenAI API Key**
```bash
curl http://localhost:5001/voicemcp/test
```
Should show `"openai_configured": true`

**5. Check Twilio Debugger**
- Go to Twilio Console → Monitor → Logs → Debugger
- Look for your recent call
- Check for any red error messages
- Verify the webhook URL was called

**6. Test TwiML endpoint directly**
```bash
curl -X POST https://your-ngrok-url.ngrok-free.app/voicemcp/twiml
```

Should return XML like:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect action="/voicemcp/connect-status">
        <ConversationRelay url="wss://..." voice="Polly.Joanna-Neural" welcomeGreeting="Hello! I'm your AI assistant. How can I help you today?" />
    </Connect>
</Response>
```

### No response when speaking?

**Check server logs for:**
```
[WebSocket] Received event: prompt
[WebSocket] User said: <your speech>
[WebSocket] AI response: <response>
[WebSocket] Sent text response
```

**If missing:**
- Verify OpenAI API key is set in `.env`
- Check for Python errors in terminal
- Ensure you have OpenAI credits available

### WebSocket connection fails?

**Verify:**
- ngrok is using HTTPS (not HTTP)
- URL in Twilio uses your current ngrok URL
- Server logs show `[WebSocket] New connection established`

**Common issues:**
- Old ngrok URL (ngrok URLs change each restart)
- Firewall blocking WebSocket connections
- ngrok free tier rate limits

## Available Endpoints

| Endpoint | Purpose | Method |
|----------|---------|--------|
| `GET /voicemcp/` | Health check & info | GET |
| `GET /voicemcp/test` | Test endpoint | GET |
| `POST /voicemcp/twiml` | TwiML webhook for calls | POST |
| `POST /voicemcp/connect-status` | Connection status callback | POST |
| `WS /voicemcp/websocket` | WebSocket for ConversationRelay | WebSocket |

## How It Works

```
Phone Call → Twilio → /voicemcp/twiml (returns TwiML)
                           ↓
                    ConversationRelay connects to WebSocket
                           ↓
              Setup message → Session initialized
                           ↓
              User speaks → STT (Twilio) → Prompt message
                           ↓
              AI processes → OpenAI Chat Completion
                           ↓
              Text token message → TTS (Twilio) → User hears response
                           ↓
              (repeat conversation loop)
```

## Features

✅ **No local STT/TTS needed** - Twilio handles all speech processing  
✅ **Low latency** - Optimized for real-time voice conversations  
✅ **Conversation memory** - Maintains context throughout the call  
✅ **Production ready** - Built with Twilio's enterprise-grade infrastructure  
✅ **Simple setup** - Just configure webhook URL in Twilio console

## Testing Tips

- **Check server logs**: You'll see WebSocket events and AI responses
- **Monitor Twilio logs**: Go to Twilio Console → Monitor → Logs → Calls
- **Test phrases**: 
  - "What can you help me with?"
  - "Tell me about your capabilities"
  - "What's the weather like?" (if you add tools later)

## Customization

### Change the welcome greeting

Edit `blueprints/voice_ai_mcp.py`:

```python
welcomeGreeting="Your custom greeting here!"
```

### Change the voice

Available voices (examples):
- `Polly.Joanna-Neural` (default)
- `Polly.Matthew-Neural`
- `Polly.Amy-Neural`
- `Google.en-US-Neural2-A`

### Adjust response length

In `generate_voice_response()`, modify:

```python
max_tokens=150,  # Increase for longer responses
```

## Troubleshooting

**Call connects but no response?**
- Check server logs for WebSocket connection
- Verify OpenAI API key is set in `.env`
- Check ngrok tunnel is active

**WebSocket connection fails?**
- Ensure you're using `https://` in Twilio (not `http://`)
- ngrok automatically provides HTTPS

**AI response is slow?**
- Check your OpenAI API rate limits
- Consider using `gpt-3.5-turbo` for faster responses

## Production Deployment

For production, replace ngrok with:
- **Heroku**: `https://your-app.herokuapp.com/voicemcp/twiml`
- **AWS/Azure**: `https://your-domain.com/voicemcp/twiml`
- **Custom domain**: `https://api.yourdomain.com/voicemcp/twiml`

Remember to update your Twilio webhook URL accordingly!

## Next Steps

Want to add capabilities? Consider:
- Adding MCP tools for booking appointments
- Integrating with your database for personalized responses
- Adding sentiment analysis
- Implementing call recording and transcription

