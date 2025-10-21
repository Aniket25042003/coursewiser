"""
Chat API endpoints for interacting with the fine-tuned model
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from app.database import get_db
from app.models import User, Chat
from app.api.auth import get_current_user
from app.services.rag import get_rag_service
from app.services.inference import generate_response_with_rag
from app.utils import format_sources_json, parse_sources_json

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    class_id: int
    use_pdf_ids: Optional[List[int]] = None
    top_k: int = 3
    max_new_tokens: int = 200


class ChatHistoryItem(BaseModel):
    message: str
    response: str


class ChatResponse(BaseModel):
    response: str
    chat_id: int
    sources: List[dict]
    timestamp: datetime


class HistoryResponse(BaseModel):
    id: int
    message: str
    response: str
    timestamp: datetime
    sources: List[dict]

    class Config:
        from_attributes = True


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Main chat endpoint with RAG support
    
    Process:
    1. Retrieve relevant chunks from ChromaDB based on user's PDFs
    2. Get conversation history (last 4 turns)
    3. Build RAG prompt with context and history
    4. Generate response using fine-tuned model
    5. Save chat and return response
    """
    try:
        rag_service = get_rag_service()
        
        # Get conversation history (last 4 turns)
        recent_chats = db.query(Chat).filter(
            Chat.user_id == current_user.id
        ).order_by(Chat.timestamp.desc()).limit(4).all()
        
        conversation_history = [
            {"message": chat.message, "response": chat.response}
            for chat in reversed(recent_chats)
        ]
        
        # Retrieve relevant chunks from vector database (class materials + personal PDFs)
        retrieved_chunks = rag_service.search(
            query=request.message,
            top_k=request.top_k,
            user_id=current_user.id,
            pdf_ids=request.use_pdf_ids,
            class_id=request.class_id
        )
        
        print(f"🔍 Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Generate response with RAG
        print(f"🤖 Generating response...")
        response_text = generate_response_with_rag(
            user_question=request.message,
            retrieved_chunks=retrieved_chunks,
            conversation_history=conversation_history,
            max_new_tokens=request.max_new_tokens
        )
        
        # Format sources for database
        sources_json = format_sources_json(retrieved_chunks)
        
        # Save chat to database
        chat_record = Chat(
            user_id=current_user.id,
            class_id=request.class_id,
            message=request.message,
            response=response_text,
            sources_json=sources_json
        )
        db.add(chat_record)
        db.commit()
        db.refresh(chat_record)
        
        # Parse sources for response
        sources = parse_sources_json(sources_json)
        
        print(f"✅ Chat saved with ID: {chat_record.id}")
        
        return {
            "response": response_text,
            "chat_id": chat_record.id,
            "sources": sources,
            "timestamp": chat_record.timestamp
        }
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@router.get("/history", response_model=List[HistoryResponse])
async def get_chat_history(
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's chat history with pagination
    """
    chats = db.query(Chat).filter(
        Chat.user_id == current_user.id
    ).order_by(Chat.timestamp.desc()).offset(offset).limit(limit).all()
    
    result = []
    for chat in chats:
        sources = parse_sources_json(chat.sources_json)
        result.append({
            "id": chat.id,
            "message": chat.message,
            "response": chat.response,
            "timestamp": chat.timestamp,
            "sources": sources
        })
    
    return result


@router.delete("/chat/{chat_id}")
async def delete_chat(
    chat_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a specific chat message
    """
    chat = db.query(Chat).filter(
        Chat.id == chat_id,
        Chat.user_id == current_user.id
    ).first()
    
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    try:
        db.delete(chat)
        db.commit()
        return {"success": True, "message": "Chat deleted successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")


@router.delete("/history/clear")
async def clear_chat_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Clear all chat history for the current user
    """
    try:
        db.query(Chat).filter(Chat.user_id == current_user.id).delete()
        db.commit()
        return {"success": True, "message": "Chat history cleared successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

