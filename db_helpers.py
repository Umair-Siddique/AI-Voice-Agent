"""
Database helper functions for conversation management with Supabase.
Handles conversation and message storage asynchronously to avoid affecting response times.
"""
import asyncio
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
from flask import current_app
import json


class ConversationDB:
    """Helper class for managing conversation data in Supabase"""
    
    # Store the Supabase client globally for background thread access
    _supabase_client = None
    
    @classmethod
    def set_client(cls, client):
        """Set the Supabase client for use in background threads"""
        cls._supabase_client = client
    
    @staticmethod
    def get_supabase_client():
        """Get Supabase client - tries Flask context first, then global fallback"""
        try:
            # Try Flask app context first
            return current_app.supabase_client
        except RuntimeError:
            # Outside of Flask app context, use global client
            return ConversationDB._supabase_client
    
    @staticmethod
    def create_conversation_sync(session_id: str, call_sid: str, user_phone: str, 
                               twilio_number: str, call_direction: str = "inbound") -> Optional[str]:
        """
        Create a new conversation record in Supabase (synchronous).
        Returns conversation_id if successful, None otherwise.
        """
        client = ConversationDB.get_supabase_client()
        if not client:
            print("⚠️  [DB] Supabase client not available")
            return None
        
        try:
            conversation_data = {
                "session_id": session_id,
                "call_sid": call_sid,
                "user_phone_number": user_phone,
                "twilio_number": twilio_number,
                "call_direction": call_direction,
                "started_at": datetime.utcnow().isoformat(),
                "total_messages": 0
            }
            
            result = client.table("conversations").insert(conversation_data).execute()
            
            if result.data:
                conversation_id = result.data[0]["id"]
                print(f"✅ [DB] Created conversation: {conversation_id}")
                return conversation_id
            else:
                print("❌ [DB] Failed to create conversation - no data returned")
                return None
                
        except Exception as e:
            print(f"❌ [DB] Error creating conversation: {e}")
            return None
    
    @staticmethod
    def end_conversation_sync(session_id: str) -> bool:
        """
        Mark a conversation as ended (synchronous).
        Returns True if successful, False otherwise.
        """
        client = ConversationDB.get_supabase_client()
        if not client:
            return False
        
        try:
            result = client.table("conversations").update({
                "ended_at": datetime.utcnow().isoformat()
            }).eq("session_id", session_id).execute()
            
            if result.data:
                print(f"✅ [DB] Ended conversation for session: {session_id}")
                return True
            else:
                print(f"⚠️  [DB] No conversation found to end for session: {session_id}")
                return False
                
        except Exception as e:
            print(f"❌ [DB] Error ending conversation: {e}")
            return False
    
    @staticmethod
    def save_message_sync(conversation_id: str, role: str, content: str, message_order: int) -> bool:
        """
        Save a single message to Supabase (synchronous).
        Returns True if successful, False otherwise.
        """
        client = ConversationDB.get_supabase_client()
        if not client:
            return False
        
        try:
            message_data = {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "message_order": message_order,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            result = client.table("messages").insert(message_data).execute()
            
            if result.data:
                # Update message count in conversation
                client.table("conversations").update({
                    "total_messages": message_order,
                    "updated_at": datetime.utcnow().isoformat()
                }).eq("id", conversation_id).execute()
                
                print(f"✅ [DB] Saved message {message_order} for conversation {conversation_id}")
                return True
            else:
                print(f"❌ [DB] Failed to save message - no data returned")
                return False
                
        except Exception as e:
            print(f"❌ [DB] Error saving message: {e}")
            return False
    
    @staticmethod
    def run_in_background(func, *args, **kwargs):
        """
        Run a function in a background thread to avoid blocking the main response.
        This ensures database operations don't affect response time.
        """
        def worker():
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"❌ [DB] Background task error: {e}")
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


# Async wrapper functions for non-blocking database operations
def create_conversation_async(session_id: str, call_sid: str, user_phone: str, 
                            twilio_number: str, call_direction: str = "inbound") -> None:
    """Create conversation record asynchronously (non-blocking)"""
    def create_and_store():
        conversation_id = ConversationDB.create_conversation_sync(
            session_id, call_sid, user_phone, twilio_number, call_direction
        )
        if conversation_id:
            # Update the session metadata with the actual conversation ID
            set_conversation_metadata(session_id, conversation_id)
    
    ConversationDB.run_in_background(create_and_store)


def end_conversation_async(session_id: str) -> None:
    """End conversation record asynchronously (non-blocking)"""
    ConversationDB.run_in_background(
        ConversationDB.end_conversation_sync,
        session_id
    )


def save_message_async(conversation_id: str, role: str, content: str, message_order: int) -> None:
    """Save message asynchronously (non-blocking)"""
    ConversationDB.run_in_background(
        ConversationDB.save_message_sync,
        conversation_id, role, content, message_order
    )


# Session storage for tracking conversation IDs
# Maps session_id -> {"conversation_id": str, "message_count": int}
conversation_metadata: Dict[str, Dict[str, Any]] = {}


def get_conversation_id(session_id: str) -> Optional[str]:
    """Get conversation_id for a session"""
    return conversation_metadata.get(session_id, {}).get("conversation_id")


def set_conversation_metadata(session_id: str, conversation_id: str) -> None:
    """Set conversation metadata for a session"""
    conversation_metadata[session_id] = {
        "conversation_id": conversation_id,
        "message_count": 0
    }


def increment_message_count(session_id: str) -> int:
    """Increment and return message count for a session"""
    if session_id in conversation_metadata:
        conversation_metadata[session_id]["message_count"] += 1
        return conversation_metadata[session_id]["message_count"]
    return 1


def cleanup_session_metadata(session_id: str) -> None:
    """Clean up session metadata when conversation ends"""
    if session_id in conversation_metadata:
        del conversation_metadata[session_id]
        print(f"🧹 [DB] Cleaned up metadata for session: {session_id}")


def get_user_conversations(user_phone: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get recent conversations for a user (synchronous, for API endpoints).
    This is a blocking operation intended for API calls, not real-time chat.
    """
    client = ConversationDB.get_supabase_client()
    if not client:
        return []
    
    try:
        result = client.table("conversations").select(
            "id, session_id, call_sid, started_at, ended_at, total_messages"
        ).eq("user_phone_number", user_phone).order(
            "started_at", desc=True
        ).limit(limit).execute()
        
        return result.data if result.data else []
        
    except Exception as e:
        print(f"❌ [DB] Error fetching user conversations: {e}")
        return []


def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Get all messages for a conversation (synchronous, for API endpoints).
    This is a blocking operation intended for API calls, not real-time chat.
    """
    client = ConversationDB.get_supabase_client()
    if not client:
        return []
    
    try:
        result = client.table("messages").select(
            "role, content, timestamp, message_order"
        ).eq("conversation_id", conversation_id).order(
            "message_order", desc=False
        ).execute()
        
        return result.data if result.data else []
        
    except Exception as e:
        print(f"❌ [DB] Error fetching conversation messages: {e}")
        return []
