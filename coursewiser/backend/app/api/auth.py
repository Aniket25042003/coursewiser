"""
Authentication API endpoints and Firebase token verification
"""
from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import firebase_admin
from firebase_admin import credentials, auth
import os
from dotenv import load_dotenv

from app.database import get_db
from app.models import User

load_dotenv()

router = APIRouter(prefix="/api/auth", tags=["auth"])

# Initialize Firebase Admin SDK
try:
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
    if cred_path and os.path.exists(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase Admin SDK initialized")
    else:
        print("⚠️  Warning: Firebase credentials not found. Authentication will not work.")
except Exception as e:
    print(f"⚠️  Warning: Failed to initialize Firebase Admin SDK: {e}")


class VerifyTokenRequest(BaseModel):
    id_token: str
    role: Optional[str] = "student"


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    google_id: str

    class Config:
        from_attributes = True


async def get_current_user(
    authorization: str = Header(...),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to verify Firebase ID token and return current user
    
    Args:
        authorization: Authorization header with Bearer token
        db: Database session
        
    Returns:
        User object
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    id_token = authorization.split("Bearer ")[1]
    
    try:
        # Verify Firebase token
        decoded_token = auth.verify_id_token(id_token)
        google_id = decoded_token['uid']
        email = decoded_token.get('email', '')
        
        # Find user in database
        user = db.query(User).filter(User.google_id == google_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found. Please complete registration.")
        
        return user
        
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")


@router.post("/verify", response_model=UserResponse)
async def verify_token(
    request: VerifyTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Verify Firebase ID token and create/retrieve user
    
    This endpoint is called after successful Google Sign-In on the frontend.
    It verifies the Firebase token and creates a user record if it doesn't exist.
    """
    try:
        # Verify Firebase token
        decoded_token = auth.verify_id_token(request.id_token)
        google_id = decoded_token['uid']
        email = decoded_token.get('email', '')
        name = decoded_token.get('name', email.split('@')[0])
        
        # Check if user exists
        user = db.query(User).filter(User.google_id == google_id).first()
        
        if not user:
            # Create new user
            user = User(
                google_id=google_id,
                email=email,
                name=name,
                role=request.role if request.role in ['student', 'professor'] else 'student'
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"✅ Created new user: {email}")
        
        return user
        
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying token: {str(e)}")

