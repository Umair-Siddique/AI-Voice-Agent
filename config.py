import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
    GPT_MODEL = os.getenv("GPT_MODEL", "gpt-5")
    STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
    TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
    RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL")
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
