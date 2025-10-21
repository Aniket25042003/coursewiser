"""
Professor dashboard API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta

from app.database import get_db
from app.models import User, Chat, Feedback, Class
from app.api.auth import get_current_user
from app.services.gemini import get_gemini_service
from app.utils import parse_sources_json

router = APIRouter(prefix="/api/professor", tags=["professor"])


class FeedbackSummaryStats(BaseModel):
    total_chats: int
    positive_feedback: int
    negative_feedback: int
    neutral_feedback: int
    no_feedback: int
    average_score: Optional[float]


class LowRatedChat(BaseModel):
    chat_id: int
    message: str
    response: str
    satisfaction_score: int
    comment: Optional[str]
    timestamp: datetime
    student_name: str


class GeminiSummaryResponse(BaseModel):
    summary: str
    low_rated_count: int
    generated_at: datetime


def verify_professor(current_user: User):
    """
    Verify that the current user is a professor
    """
    if current_user.role != "professor":
        raise HTTPException(status_code=403, detail="Access denied. Professor role required.")


@router.get("/stats", response_model=FeedbackSummaryStats)
async def get_feedback_stats(
    class_id: int,
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get summary statistics of student feedback for a specific class
    
    Args:
        class_id: ID of the class to get stats for
        days: Number of days to look back (default: 30)
    """
    verify_professor(current_user)
    
    # Verify professor owns this class
    cls = db.query(Class).filter(
        Class.id == class_id,
        Class.professor_id == current_user.id
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    
    # Calculate date threshold
    date_threshold = datetime.utcnow() - timedelta(days=days)
    
    # Total chats in period for this class
    total_chats = db.query(func.count(Chat.id)).filter(
        Chat.timestamp >= date_threshold,
        Chat.class_id == class_id
    ).scalar()
    
    # Get feedback counts for this class
    positive = db.query(func.count(Feedback.id)).join(Chat).filter(
        Chat.timestamp >= date_threshold,
        Chat.class_id == class_id,
        Feedback.satisfaction_score == 1
    ).scalar()
    
    negative = db.query(func.count(Feedback.id)).join(Chat).filter(
        Chat.timestamp >= date_threshold,
        Chat.class_id == class_id,
        Feedback.satisfaction_score == -1
    ).scalar()
    
    neutral = db.query(func.count(Feedback.id)).join(Chat).filter(
        Chat.timestamp >= date_threshold,
        Chat.class_id == class_id,
        Feedback.satisfaction_score == 0
    ).scalar()
    
    # Calculate average score for this class
    avg_score = db.query(func.avg(Feedback.satisfaction_score)).join(Chat).filter(
        Chat.timestamp >= date_threshold,
        Chat.class_id == class_id
    ).scalar()
    
    no_feedback = total_chats - (positive + negative + neutral)
    
    return {
        "total_chats": total_chats or 0,
        "positive_feedback": positive or 0,
        "negative_feedback": negative or 0,
        "neutral_feedback": neutral or 0,
        "no_feedback": no_feedback or 0,
        "average_score": float(avg_score) if avg_score else None
    }


@router.get("/low_rated", response_model=List[LowRatedChat])
async def get_low_rated_chats(
    class_id: int,
    days: int = 30,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all chats with low satisfaction ratings (thumbs down) for a specific class
    
    Args:
        class_id: ID of the class to get low-rated chats for
        days: Number of days to look back
        limit: Maximum number of results
    """
    verify_professor(current_user)
    
    # Verify professor owns this class
    cls = db.query(Class).filter(
        Class.id == class_id,
        Class.professor_id == current_user.id
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    
    date_threshold = datetime.utcnow() - timedelta(days=days)
    
    # Query for low-rated chats in this class
    results = db.query(Chat, Feedback, User).join(
        Feedback, Chat.id == Feedback.chat_id
    ).join(
        User, Chat.user_id == User.id
    ).filter(
        Chat.timestamp >= date_threshold,
        Chat.class_id == class_id,
        Feedback.satisfaction_score == -1
    ).order_by(Chat.timestamp.desc()).limit(limit).all()
    
    low_rated = []
    for chat, feedback, user in results:
        low_rated.append({
            "chat_id": chat.id,
            "message": chat.message,
            "response": chat.response,
            "satisfaction_score": feedback.satisfaction_score,
            "comment": feedback.comment,
            "timestamp": chat.timestamp,
            "student_name": user.name
        })
    
    return low_rated


@router.get("/summary", response_model=GeminiSummaryResponse)
async def get_gemini_summary(
    class_id: int,
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate AI-powered summary of common student issues using Gemini for a specific class
    
    This endpoint aggregates low-rated Q&A pairs and sends them to Gemini
    for analysis and summarization.
    """
    verify_professor(current_user)
    
    # Verify professor owns this class
    cls = db.query(Class).filter(
        Class.id == class_id,
        Class.professor_id == current_user.id
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    
    date_threshold = datetime.utcnow() - timedelta(days=days)
    
    # Get low-rated chats for this class
    results = db.query(Chat, Feedback, User).join(
        Feedback, Chat.id == Feedback.chat_id
    ).join(
        User, Chat.user_id == User.id
    ).filter(
        Chat.timestamp >= date_threshold,
        Chat.class_id == class_id,
        Feedback.satisfaction_score == -1
    ).order_by(Chat.timestamp.desc()).limit(100).all()
    
    if not results:
        return {
            "summary": "No low-rated feedback found in the specified time period for this class.",
            "low_rated_count": 0,
            "generated_at": datetime.utcnow()
        }
    
    # Prepare data for Gemini
    feedback_data = []
    for chat, feedback, user in results:
        feedback_data.append({
            "message": chat.message,
            "response": chat.response,
            "comment": feedback.comment,
            "student_name": user.name
        })
    
    # Call Gemini API
    gemini_service = get_gemini_service()
    print(f"🤖 Generating Gemini summary for {len(feedback_data)} low-rated chats...")
    summary = await gemini_service.generate_summary(feedback_data)
    
    return {
        "summary": summary,
        "low_rated_count": len(feedback_data),
        "generated_at": datetime.utcnow()
    }


@router.get("/export_csv")
async def export_low_rated_csv(
    class_id: int,
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Export low-rated Q&A pairs as CSV for a specific class
    """
    verify_professor(current_user)
    
    # Verify professor owns this class
    cls = db.query(Class).filter(
        Class.id == class_id,
        Class.professor_id == current_user.id
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    
    date_threshold = datetime.utcnow() - timedelta(days=days)
    
    results = db.query(Chat, Feedback, User).join(
        Feedback, Chat.id == Feedback.chat_id
    ).join(
        User, Chat.user_id == User.id
    ).filter(
        Chat.timestamp >= date_threshold,
        Chat.class_id == class_id,
        Feedback.satisfaction_score == -1
    ).order_by(Chat.timestamp.desc()).all()
    
    # Build CSV content
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['Chat ID', 'Student', 'Question', 'Response', 'Comment', 'Timestamp'])
    
    # Write data
    for chat, feedback, user in results:
        writer.writerow([
            chat.id,
            user.name,
            chat.message,
            chat.response,
            feedback.comment or '',
            chat.timestamp.isoformat()
        ])
    
    csv_content = output.getvalue()
    
    from fastapi.responses import Response
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=low_rated_feedback_{cls.name}_{datetime.now().strftime('%Y%m%d')}.csv"}
    )

