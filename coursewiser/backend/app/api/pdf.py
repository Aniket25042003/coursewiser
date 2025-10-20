"""
PDF upload and management API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List
import os
import tempfile
from datetime import datetime

from app.database import get_db
from app.models import User, PdfDocument, PdfChunk
from app.api.auth import get_current_user
from app.services.rag import get_rag_service

router = APIRouter(prefix="/api/pdf", tags=["pdf"])


class UploadPdfRequest(BaseModel):
    firebase_storage_path: str
    filename: str


class PdfDocumentResponse(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    chunk_count: int

    class Config:
        from_attributes = True


@router.post("/upload_pdf")
async def upload_pdf(
    request: UploadPdfRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Process an uploaded PDF: extract text, chunk, embed, and index
    
    Frontend should first upload PDF to Firebase Storage, then call this endpoint
    with the Firebase storage path.
    """
    try:
        rag_service = get_rag_service()
        
        # Note: In production, you would download the PDF from Firebase Storage
        # For now, we'll accept a local file path for testing
        # TODO: Implement Firebase Storage download
        
        # For local testing, treat firebase_storage_path as a local path
        pdf_path = request.firebase_storage_path
        
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=400, detail="PDF file not found at specified path")
        
        # Extract text from PDF
        print(f"📄 Extracting text from {request.filename}...")
        text = rag_service.extract_text_from_pdf(pdf_path)
        
        # Chunk the text
        print(f"✂️  Chunking text...")
        chunks = rag_service.chunk_text(text)
        
        # Create PDF document record
        pdf_doc = PdfDocument(
            user_id=current_user.id,
            filename=request.filename,
            firebase_storage_path=request.firebase_storage_path
        )
        db.add(pdf_doc)
        db.commit()
        db.refresh(pdf_doc)
        
        # Create chunk records
        chunk_records = []
        for i, chunk_text in enumerate(chunks):
            chunk_record = PdfChunk(
                pdf_id=pdf_doc.id,
                chunk_text=chunk_text,
                chunk_index=i,
                metadata_json=f'{{"page": "auto", "position": {i}}}'
            )
            chunk_records.append(chunk_record)
        
        db.add_all(chunk_records)
        db.commit()
        
        # Get chunk IDs
        chunk_ids = [c.id for c in chunk_records]
        
        # Index in ChromaDB
        print(f"🔍 Indexing {len(chunks)} chunks in ChromaDB...")
        rag_service.index_pdf_chunks(
            chunks=chunks,
            pdf_id=pdf_doc.id,
            chunk_ids=chunk_ids,
            user_id=current_user.id,
            filename=request.filename
        )
        
        print(f"✅ Successfully processed {request.filename}")
        
        return {
            "success": True,
            "pdf_id": pdf_doc.id,
            "filename": request.filename,
            "chunk_count": len(chunks),
            "summary": text[:200] + "..." if len(text) > 200 else text
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.post("/upload_file")
async def upload_file(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload PDF file directly (alternative to Firebase Storage)
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        rag_service = get_rag_service()
        
        # Extract text from PDF
        print(f"📄 Extracting text from {file.filename}...")
        text = rag_service.extract_text_from_pdf(tmp_path)
        
        # Chunk the text
        print(f"✂️  Chunking text...")
        chunks = rag_service.chunk_text(text)
        
        # Create PDF document record
        pdf_doc = PdfDocument(
            user_id=current_user.id,
            filename=file.filename,
            firebase_storage_path=tmp_path  # Store local path for now
        )
        db.add(pdf_doc)
        db.commit()
        db.refresh(pdf_doc)
        
        # Create chunk records
        chunk_records = []
        for i, chunk_text in enumerate(chunks):
            chunk_record = PdfChunk(
                pdf_id=pdf_doc.id,
                chunk_text=chunk_text,
                chunk_index=i,
                metadata_json=f'{{"page": "auto", "position": {i}}}'
            )
            chunk_records.append(chunk_record)
        
        db.add_all(chunk_records)
        db.commit()
        
        # Get chunk IDs
        chunk_ids = [c.id for c in chunk_records]
        
        # Index in ChromaDB
        print(f"🔍 Indexing {len(chunks)} chunks in ChromaDB...")
        rag_service.index_pdf_chunks(
            chunks=chunks,
            pdf_id=pdf_doc.id,
            chunk_ids=chunk_ids,
            user_id=current_user.id,
            filename=file.filename
        )
        
        print(f"✅ Successfully processed {file.filename}")
        
        return {
            "success": True,
            "pdf_id": pdf_doc.id,
            "filename": file.filename,
            "chunk_count": len(chunks),
            "summary": text[:200] + "..." if len(text) > 200 else text
        }
        
    except Exception as e:
        db.rollback()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@router.get("/user_pdfs", response_model=List[PdfDocumentResponse])
async def get_user_pdfs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all PDFs uploaded by the current user
    """
    pdfs = db.query(PdfDocument).filter(PdfDocument.user_id == current_user.id).all()
    
    result = []
    for pdf in pdfs:
        chunk_count = db.query(PdfChunk).filter(PdfChunk.pdf_id == pdf.id).count()
        result.append({
            "id": pdf.id,
            "filename": pdf.filename,
            "upload_timestamp": pdf.upload_timestamp,
            "chunk_count": chunk_count
        })
    
    return result


@router.delete("/pdf/{pdf_id}")
async def delete_pdf(
    pdf_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a PDF and its chunks
    """
    pdf = db.query(PdfDocument).filter(
        PdfDocument.id == pdf_id,
        PdfDocument.user_id == current_user.id
    ).first()
    
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")
    
    try:
        # Get chunk IDs
        chunks = db.query(PdfChunk).filter(PdfChunk.pdf_id == pdf_id).all()
        chunk_ids = [c.id for c in chunks]
        
        # Delete from ChromaDB
        if chunk_ids:
            rag_service = get_rag_service()
            rag_service.delete_pdf_chunks(chunk_ids)
        
        # Delete from database (cascades to chunks)
        db.delete(pdf)
        db.commit()
        
        return {"success": True, "message": "PDF deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting PDF: {str(e)}")

