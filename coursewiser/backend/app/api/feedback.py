"""
Feedback API endpoints for student satisfaction ratings
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.database import get_db
from app.models import User, Chat, Feedback
from app.api.auth import get_current_user

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class SubmitFeedbackRequest(BaseModel):
    chat_id: int
    satisfaction_score: int  # -1 (thumbs down), 0 (neutral), 1 (thumbs up)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    chat_id: int
    satisfaction_score: int
    comment: Optional[str]

    class Config:
        from_attributes = True


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(
    request: SubmitFeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Submit feedback for a chat response
    
    Args:
        request: Feedback data with chat_id, satisfaction_score, and optional comment
        
    Returns:
        Created feedback record
    """
    # Validate satisfaction score
    if request.satisfaction_score not in [-1, 0, 1]:
        raise HTTPException(
            status_code=400,
            detail="Satisfaction score must be -1 (thumbs down), 0 (neutral), or 1 (thumbs up)"
        )
    
    # Verify chat exists and belongs to user
    chat = db.query(Chat).filter(
        Chat.id == request.chat_id,
        Chat.user_id == current_user.id
    ).first()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Check if feedback already exists
    existing_feedback = db.query(Feedback).filter(
        Feedback.chat_id == request.chat_id
    ).first()
    
    try:
        if existing_feedback:
            # Update existing feedback
            existing_feedback.satisfaction_score = request.satisfaction_score
            existing_feedback.comment = request.comment
            db.commit()
            db.refresh(existing_feedback)
            print(f"✅ Updated feedback for chat {request.chat_id}")
            return existing_feedback
        else:
            # Create new feedback
            feedback = Feedback(
                chat_id=request.chat_id,
                satisfaction_score=request.satisfaction_score,
                comment=request.comment
            )
            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            print(f"✅ Created feedback for chat {request.chat_id}")
            return feedback
            
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error submitting feedback: {str(e)}")


@router.get("/chat/{chat_id}", response_model=Optional[FeedbackResponse])
async def get_feedback(
    chat_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get feedback for a specific chat
    """
    # Verify chat belongs to user
    chat = db.query(Chat).filter(
        Chat.id == chat_id,
        Chat.user_id == current_user.id
    ).first()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    feedback = db.query(Feedback).filter(Feedback.chat_id == chat_id).first()
    return feedback


@router.delete("/{feedback_id}")
async def delete_feedback(
    feedback_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete feedback
    """
    # Get feedback and verify it belongs to user's chat
    feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Verify the chat belongs to the user
    chat = db.query(Chat).filter(
        Chat.id == feedback.chat_id,
        Chat.user_id == current_user.id
    ).first()
    
    if not chat:
        raise HTTPException(status_code=403, detail="Not authorized to delete this feedback")
    
    try:
        db.delete(feedback)
        db.commit()
        return {"success": True, "message": "Feedback deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting feedback: {str(e)}")

