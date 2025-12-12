from flask import Flask
from config import Config

from extensions import init_openai, init_supabase

from blueprints.sms_assistant import sms_assistant_bp
from blueprints.whatsapp_assistant import whatsapp_assistant_bp
from blueprints.mcp_server import mcp_bp
from blueprints.voice_ai_mcp import voice_mcp_bp
from blueprints.whatsapp_assistant_mcp import whatsapp_assistant_mcp_bp

from flask_cors import CORS
from flask_sock import Sock

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Allow all origins with comprehensive settings
    CORS(app, 
         supports_credentials=True, 
         origins="*",
         methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
         allow_headers=['Content-Type', 'Authorization', 'X-Requested-With', 'Accept', 'Origin'],
         expose_headers=['Content-Type', 'Authorization']
    )

    # Initialize WebSocket support
    sock = Sock(app)

    # Initialize extensions
    init_openai(app)
    init_supabase(app)

    # Register blueprints

    app.register_blueprint(sms_assistant_bp, url_prefix="/sms")
    app.register_blueprint(whatsapp_assistant_bp, url_prefix="/whatsapp")
    app.register_blueprint(whatsapp_assistant_mcp_bp, url_prefix="/whatsappmcp")
    app.register_blueprint(mcp_bp, url_prefix="/mcp")
    app.register_blueprint(voice_mcp_bp,url_prefix="/voicemcp")

    
    # Register WebSocket routes for voice_mcp
    from blueprints.voice_ai_mcp import register_websocket_routes as register_voice_mcp_websocket_routes
    register_voice_mcp_websocket_routes(sock, app)

    
    return app