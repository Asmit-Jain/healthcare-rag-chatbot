import os
import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# Secret key for JWT signing (loads from .env or uses default fallback)
JWT_SECRET = os.getenv("JWT_SECRET", "medlink_healthcare_rag_jwt_secret_key_2026")
JWT_ALGORITHM = "HS256"

def hash_password(plain_password: str) -> str:
    """
    Hashes a plain-text password using bcrypt with a random salt.
    """
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain-text password against a bcrypt password hash.
    """
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        print(f"[ERROR] Password verification error: {e}")
        return False

def create_access_token(user_data: dict, expires_days: int = 7) -> str:
    """
    Generates a signed JWT access token containing user identity claims.
    """
    expire = datetime.now(timezone.utc) + timedelta(days=expires_days)
    payload = {
        "sub": user_data.get("email"),
        "email": user_data.get("email"),
        "full_name": user_data.get("full_name", ""),
        "exp": expire,
        "iat": datetime.now(timezone.utc)
    }
    encoded_jwt = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def verify_access_token(token: str) -> dict:
    """
    Decodes and validates a JWT token. Returns the payload dict if valid, or None if expired/invalid.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        print("[WARNING] JWT Token signature expired.")
        return None
    except jwt.InvalidTokenError as e:
        print(f"[WARNING] Invalid JWT Token: {e}")
        return None
