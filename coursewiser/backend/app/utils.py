"""
Utility functions for the backend
"""
import json
import secrets
import string
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import os
from dotenv import load_dotenv

load_dotenv()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24


def format_sources_json(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks as JSON string for database storage
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        JSON string
    """
    sources = []
    for chunk in chunks:
        metadata = chunk.get('metadata', {})
        sources.append({
            'text': chunk.get('text', '')[:200],  # Store first 200 chars
            'filename': metadata.get('filename', 'Unknown'),
            'chunk_index': metadata.get('chunk_index', 0),
            'pdf_id': metadata.get('pdf_id', None)
        })
    return json.dumps(sources)


def parse_sources_json(sources_json: str) -> List[Dict]:
    """
    Parse sources JSON from database
    
    Args:
        sources_json: JSON string from database
        
    Returns:
        List of source dictionaries
    """
    try:
        return json.loads(sources_json) if sources_json else []
    except:
        return []


def validate_role(role: str) -> bool:
    """
    Validate user role
    
    Args:
        role: Role string
        
    Returns:
        True if valid role
    """
    return role in ['student', 'professor']


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int, username: str, role: str, email: str) -> str:
    """
    Create a JWT access token for professor authentication
    
    Args:
        user_id: User ID
        username: Username
        role: User role
        email: User email
        
    Returns:
        JWT token string
    """
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "email": email,
        "exp": expire
    }
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT access token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token data or None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def generate_class_code(length: int = 8) -> str:
    """
    Generate a random alphanumeric class code
    
    Args:
        length: Length of the code (default 8)
        
    Returns:
        Random class code
    """
    alphabet = string.ascii_uppercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

