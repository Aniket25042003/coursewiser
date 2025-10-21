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
from app.utils import (
    hash_password, 
    verify_password, 
    create_access_token, 
    verify_access_token
)

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


class ProfessorLoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: str
    google_id: Optional[str] = None
    username: Optional[str] = None
    must_change_password: Optional[bool] = False

    class Config:
        from_attributes = True


class ProfessorLoginResponse(BaseModel):
    user: UserResponse
    access_token: str
    token_type: str = "bearer"


async def get_current_user(
    authorization: str = Header(...),
    db: Session = Depends(get_db)
) -> User:
    """
    Dependency to verify token (Firebase for students, JWT for professors) and return current user
    
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
    
    token = authorization.split("Bearer ")[1]
    
    # Try JWT token first (for professors)
    jwt_payload = verify_access_token(token)
    if jwt_payload:
        user_id = int(jwt_payload.get('sub'))
        user = db.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user
    
    # Try Firebase token (for students)
    try:
        decoded_token = auth.verify_id_token(token)
        google_id = decoded_token['uid']
        
        # Find user in database
        user = db.query(User).filter(User.google_id == google_id).first()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found. Please complete registration.")
        
        return user
        
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")


@router.post("/verify", response_model=UserResponse)
async def verify_token(
    request: VerifyTokenRequest,
    db: Session = Depends(get_db)
):
    """
    Verify Firebase ID token and create/retrieve student user
    
    This endpoint is called after successful Google Sign-In on the frontend.
    It verifies the Firebase token and creates a student user record if it doesn't exist.
    Professors cannot sign up through this endpoint.
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
            # Create new student user
            user = User(
                google_id=google_id,
                email=email,
                name=name,
                role='student'
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            print(f"✅ Created new student: {email}")
        
        return user
        
    except auth.InvalidIdTokenError:
        raise HTTPException(status_code=401, detail="Invalid Firebase token")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error verifying token: {str(e)}")


@router.post("/professor/login", response_model=ProfessorLoginResponse)
async def professor_login(
    request: ProfessorLoginRequest,
    db: Session = Depends(get_db)
):
    """
    Professor login with username and password
    
    Returns JWT token for authentication
    """
    # Find professor by username
    user = db.query(User).filter(
        User.username == request.username,
        User.role == 'professor'
    ).first()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Verify password
    if not user.password_hash or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Create JWT token
    access_token = create_access_token(
        user_id=user.id,
        username=user.username,
        role=user.role,
        email=user.email
    )
    
    print(f"✅ Professor login: {user.username}")
    
    return {
        "user": user,
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.post("/professor/change_password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Change professor password
    
    Requires current password verification
    """
    # Only professors can change password
    if current_user.role != 'professor':
        raise HTTPException(status_code=403, detail="Only professors can change password")
    
    # Verify old password
    if not current_user.password_hash or not verify_password(request.old_password, current_user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid current password")
    
    # Update password
    current_user.password_hash = hash_password(request.new_password)
    current_user.must_change_password = False
    
    db.commit()
    
    print(f"✅ Password changed for: {current_user.username}")
    
    return {"success": True, "message": "Password changed successfully"}

