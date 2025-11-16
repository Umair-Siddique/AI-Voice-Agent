from openai import OpenAI as OpenAIClient
from config import Config

def init_openai(app):
    app.openai_client = OpenAIClient(api_key=Config.OPENAI_API_KEY)