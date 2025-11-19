from flask import Flask
from config import Config

from extensions import init_openai
from blueprints.voice_assistant import voice_assistant_bp
from blueprints.sms_assistant import sms_assistant_bp
from blueprints.whatsapp_assistant import whatsapp_assistant_bp

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

    # Register blueprints
    app.register_blueprint(voice_assistant_bp, url_prefix="/voice")
    app.register_blueprint(sms_assistant_bp, url_prefix="/sms")
    app.register_blueprint(whatsapp_assistant_bp, url_prefix="/whatsapp")

    # Register WebSocket routes
    from blueprints.voice_assistant import register_websocket_routes
    register_websocket_routes(sock, app)

    
    return app