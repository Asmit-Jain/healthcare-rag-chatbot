import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# Load environment variables
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

# Singleton instances for MongoDB connection
_mongo_client = None
DB_NAME = "health_awareness_rag"
COLLECTION_NAME = "chat_sessions"

def get_mongo_client():
    """
    Initializes and returns a cached PyMongo client.
    """
    global _mongo_client
    if _mongo_client is None:
        if not MONGO_URI:
            raise ValueError("MONGO_URI environment variable is not set in .env file.")
        _mongo_client = MongoClient(
            MONGO_URI,
            server_api=ServerApi('1'),
            serverSelectionTimeoutMS=5000  # 5 second timeout for fast diagnostics check
        )
    return _mongo_client

def get_sessions_collection():
    """
    Returns the chat_sessions MongoDB collection.
    """
    client = get_mongo_client()
    db = client[DB_NAME]
    return db[COLLECTION_NAME]

def check_mongo_connection():
    """
    Pings the MongoDB server to verify if the connection is active.
    Returns True if connected successfully, False otherwise.
    """
    try:
        client = get_mongo_client()
        # The admin command 'ping' is a lightweight way to verify server connectivity
        client.admin.command('ping')
        return True
    except Exception as e:
        print(f"[ERROR] MongoDB Connection Check Failed: {e}")
        return False

# --- PHASE 2: DATABASE CRUD HELPER FUNCTIONS ---

def create_chat_session(session_id, title="New Chat", language="English"):
    """
    Creates a new chat session document in MongoDB.
    """
    try:
        collection = get_sessions_collection()
        session_doc = {
            "session_id": session_id,
            "title": title,
            "language": language,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": []
        }
        collection.insert_one(session_doc)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create chat session {session_id}: {e}")
        return False

def append_message_to_session(session_id, role, content, chunks=None, distance=None):
    """
    Appends a user or assistant message to the specified chat session.
    """
    try:
        collection = get_sessions_collection()
        msg_obj = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        if chunks is not None:
            msg_obj["chunks"] = chunks
        if distance is not None:
            msg_obj["distance"] = distance

        result = collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": msg_obj},
                "$set": {"updated_at": datetime.now(timezone.utc).isoformat()}
            }
        )
        return result.modified_count > 0
    except Exception as e:
        print(f"[ERROR] Failed to append message to session {session_id}: {e}")
        return False

def update_session_title(session_id, new_title):
    """
    Updates the display title of a chat session.
    """
    try:
        collection = get_sessions_collection()
        collection.update_one(
            {"session_id": session_id},
            {"$set": {"title": new_title, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update session title for {session_id}: {e}")
        return False

def update_session_language(session_id, language):
    """
    Updates the target language preference of a chat session.
    """
    try:
        collection = get_sessions_collection()
        collection.update_one(
            {"session_id": session_id},
            {"$set": {"language": language, "updated_at": datetime.now(timezone.utc).isoformat()}}
        )
        return True
    except Exception as e:
        print(f"[ERROR] Failed to update session language for {session_id}: {e}")
        return False

def get_all_sessions():
    """
    Retrieves all chat sessions sorted by updated_at descending (newest first).
    Returns a list of dicts containing session metadata.
    """
    try:
        collection = get_sessions_collection()
        sessions = list(collection.find(
            {},
            {"_id": 0, "session_id": 1, "title": 1, "language": 1, "created_at": 1, "updated_at": 1}
        ).sort("updated_at", -1))
        return sessions
    except Exception as e:
        print(f"[ERROR] Failed to fetch sessions: {e}")
        return []

def get_session_messages(session_id):
    """
    Retrieves the messages array for a given session_id.
    """
    try:
        collection = get_sessions_collection()
        doc = collection.find_one({"session_id": session_id}, {"_id": 0, "messages": 1})
        if doc and "messages" in doc:
            return doc["messages"]
        return []
    except Exception as e:
        print(f"[ERROR] Failed to fetch messages for session {session_id}: {e}")
        return []

def get_session_language(session_id):
    """
    Retrieves the language preference for a given session_id.
    """
    try:
        collection = get_sessions_collection()
        doc = collection.find_one({"session_id": session_id}, {"_id": 0, "language": 1})
        if doc and "language" in doc:
            return doc.get("language", "English")
        return "English"
    except Exception as e:
        print(f"[ERROR] Failed to fetch language for session {session_id}: {e}")
        return "English"

def delete_session(session_id):
    """
    Deletes a session document by session_id.
    """
    try:
        collection = get_sessions_collection()
        collection.delete_one({"session_id": session_id})
        return True
    except Exception as e:
        print(f"[ERROR] Failed to delete session {session_id}: {e}")
        return False

if __name__ == "__main__":
    print("Testing MongoDB Connection...")
    if check_mongo_connection():
        print("[SUCCESS] Successfully connected to MongoDB Atlas!")
    else:
        print("[FAILURE] Failed to connect to MongoDB Atlas.")
