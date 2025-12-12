from openai import OpenAI as OpenAIClient
from supabase import create_client, Client
from config import Config

def init_openai(app):
    app.openai_client = OpenAIClient(api_key=Config.OPENAI_API_KEY)

def init_supabase(app):
    """Initialize Supabase client"""
    if Config.SUPABASE_URL and Config.SUPABASE_SECRET_KEY:
        client = create_client(Config.SUPABASE_URL, Config.SUPABASE_SECRET_KEY)
        app.supabase_client = client
        
        # Set client for background thread access
        from db_helpers import ConversationDB
        ConversationDB.set_client(client)
        
        print("✅ Supabase client initialized successfully")
    else:
        app.supabase_client = None
        print("⚠️  Supabase not configured - conversation storage disabled")