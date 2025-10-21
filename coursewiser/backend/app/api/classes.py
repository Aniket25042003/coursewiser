"""
Class management API endpoints for professors and students
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import shutil

from app.database import get_db
from app.models import User, Class, ClassEnrollment, ClassMaterial, ClassMaterialChunk
from app.api.auth import get_current_user
from app.utils import generate_class_code
from app.services.rag import get_rag_service

router = APIRouter(prefix="/api/classes", tags=["classes"])


# Pydantic models
class CreateClassRequest(BaseModel):
    name: str
    description: Optional[str] = None


class UpdateClassRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class ClassResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    class_code: str
    professor_id: int
    professor_name: str
    created_at: datetime
    is_active: bool
    enrolled_count: Optional[int] = 0

    class Config:
        from_attributes = True


class JoinClassRequest(BaseModel):
    class_code: str


class EnrolledClassResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    professor_name: str
    joined_at: datetime

    class Config:
        from_attributes = True


class StudentResponse(BaseModel):
    id: int
    name: str
    email: str
    joined_at: datetime

    class Config:
        from_attributes = True


class MaterialResponse(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime
    class_id: int

    class Config:
        from_attributes = True


# Professor endpoints
@router.post("", response_model=ClassResponse)
async def create_class(
    request: CreateClassRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new class (professor only)
    """
    if current_user.role != 'professor':
        raise HTTPException(status_code=403, detail="Only professors can create classes")
    
    # Generate unique class code
    class_code = generate_class_code()
    while db.query(Class).filter(Class.class_code == class_code).first():
        class_code = generate_class_code()
    
    # Create class
    new_class = Class(
        name=request.name,
        description=request.description,
        class_code=class_code,
        professor_id=current_user.id
    )
    
    db.add(new_class)
    db.commit()
    db.refresh(new_class)
    
    print(f"✅ Class created: {new_class.name} ({new_class.class_code})")
    
    return {
        **new_class.__dict__,
        "professor_name": current_user.name,
        "enrolled_count": 0
    }


@router.get("/my_classes", response_model=List[ClassResponse])
async def get_my_classes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all classes created by the professor
    """
    if current_user.role != 'professor':
        raise HTTPException(status_code=403, detail="Only professors can access this endpoint")
    
    classes = db.query(Class).filter(Class.professor_id == current_user.id).all()
    
    result = []
    for cls in classes:
        enrolled_count = db.query(ClassEnrollment).filter(ClassEnrollment.class_id == cls.id).count()
        result.append({
            **cls.__dict__,
            "professor_name": current_user.name,
            "enrolled_count": enrolled_count
        })
    
    return result


@router.put("/{class_id}", response_model=ClassResponse)
async def update_class(
    class_id: int,
    request: UpdateClassRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update class details (professor only)
    """
    if current_user.role != 'professor':
        raise HTTPException(status_code=403, detail="Only professors can update classes")
    
    cls = db.query(Class).filter(
        Class.id == class_id,
        Class.professor_id == current_user.id
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    
    # Update fields
    if request.name is not None:
        cls.name = request.name
    if request.description is not None:
        cls.description = request.description
    if request.is_active is not None:
        cls.is_active = request.is_active
    
    db.commit()
    db.refresh(cls)
    
    enrolled_count = db.query(ClassEnrollment).filter(ClassEnrollment.class_id == cls.id).count()
    
    return {
        **cls.__dict__,
        "professor_name": current_user.name,
        "enrolled_count": enrolled_count
    }


@router.delete("/{class_id}")
async def delete_class(
    class_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Deactivate a class (professor only)
    """
    if current_user.role != 'professor':
        raise HTTPException(status_code=403, detail="Only professors can delete classes")
    
    cls = db.query(Class).filter(
        Class.id == class_id,
        Class.professor_id == current_user.id
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    
    # Deactivate instead of deleting
    cls.is_active = False
    db.commit()
    
    return {"success": True, "message": "Class deactivated successfully"}


@router.get("/{class_id}/students", response_model=List[StudentResponse])
async def get_class_students(
    class_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all students enrolled in a class (professor only)
    """
    if current_user.role != 'professor':
        raise HTTPException(status_code=403, detail="Only professors can view class students")
    
    # Verify professor owns this class
    cls = db.query(Class).filter(
        Class.id == class_id,
        Class.professor_id == current_user.id
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    
    enrollments = db.query(ClassEnrollment).filter(
        ClassEnrollment.class_id == class_id
    ).all()
    
    result = []
    for enrollment in enrollments:
        student = enrollment.student
        result.append({
            "id": student.id,
            "name": student.name,
            "email": student.email,
            "joined_at": enrollment.joined_at
        })
    
    return result


# Student endpoints
@router.post("/join", response_model=EnrolledClassResponse)
async def join_class(
    request: JoinClassRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Join a class using class code (student only)
    """
    if current_user.role != 'student':
        raise HTTPException(status_code=403, detail="Only students can join classes")
    
    # Find class by code
    cls = db.query(Class).filter(
        Class.class_code == request.class_code,
        Class.is_active == True
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Invalid class code or class is inactive")
    
    # Check if already enrolled
    existing = db.query(ClassEnrollment).filter(
        ClassEnrollment.class_id == cls.id,
        ClassEnrollment.student_id == current_user.id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Already enrolled in this class")
    
    # Create enrollment
    enrollment = ClassEnrollment(
        class_id=cls.id,
        student_id=current_user.id
    )
    
    db.add(enrollment)
    db.commit()
    db.refresh(enrollment)
    
    print(f"✅ Student {current_user.name} joined class {cls.name}")
    
    professor = db.query(User).filter(User.id == cls.professor_id).first()
    
    return {
        "id": cls.id,
        "name": cls.name,
        "description": cls.description,
        "professor_name": professor.name if professor else "Unknown",
        "joined_at": enrollment.joined_at
    }


@router.get("/enrolled", response_model=List[EnrolledClassResponse])
async def get_enrolled_classes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all classes the student is enrolled in
    """
    if current_user.role != 'student':
        raise HTTPException(status_code=403, detail="Only students can access this endpoint")
    
    enrollments = db.query(ClassEnrollment).filter(
        ClassEnrollment.student_id == current_user.id
    ).all()
    
    result = []
    for enrollment in enrollments:
        cls = enrollment.class_obj
        if cls.is_active:
            professor = db.query(User).filter(User.id == cls.professor_id).first()
            result.append({
                "id": cls.id,
                "name": cls.name,
                "description": cls.description,
                "professor_name": professor.name if professor else "Unknown",
                "joined_at": enrollment.joined_at
            })
    
    return result


@router.post("/leave/{class_id}")
async def leave_class(
    class_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Leave a class (student only)
    """
    if current_user.role != 'student':
        raise HTTPException(status_code=403, detail="Only students can leave classes")
    
    enrollment = db.query(ClassEnrollment).filter(
        ClassEnrollment.class_id == class_id,
        ClassEnrollment.student_id == current_user.id
    ).first()
    
    if not enrollment:
        raise HTTPException(status_code=404, detail="Not enrolled in this class")
    
    db.delete(enrollment)
    db.commit()
    
    return {"success": True, "message": "Left class successfully"}


# Material management endpoints
@router.post("/{class_id}/materials", response_model=MaterialResponse)
async def upload_class_material(
    class_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload a PDF material to a class (professor only)
    """
    if current_user.role != 'professor':
        raise HTTPException(status_code=403, detail="Only professors can upload class materials")
    
    # Verify professor owns this class
    cls = db.query(Class).filter(
        Class.id == class_id,
        Class.professor_id == current_user.id
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save file locally
        upload_dir = os.path.join("data", "class_materials", str(class_id))
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process PDF with RAG service
        rag_service = get_rag_service()
        
        # Extract text and create chunks
        text = rag_service.extract_text_from_pdf(file_path)
        chunks = rag_service.chunk_text(text)
        
        # Save material metadata
        material = ClassMaterial(
            class_id=class_id,
            filename=file.filename,
            firebase_storage_path=file_path,  # Using local path for now
            uploaded_by=current_user.id
        )
        
        db.add(material)
        db.commit()
        db.refresh(material)
        
        # Save chunks to database
        chunk_records = []
        chunk_ids = []
        for i, chunk_text in enumerate(chunks):
            chunk_record = ClassMaterialChunk(
                material_id=material.id,
                chunk_text=chunk_text,
                chunk_index=i
            )
            db.add(chunk_record)
            chunk_records.append(chunk_record)
        
        db.commit()
        
        # Refresh to get IDs
        for chunk in chunk_records:
            db.refresh(chunk)
            chunk_ids.append(chunk.id)
        
        # Index in ChromaDB
        rag_service.index_class_material_chunks(
            chunks=chunks,
            material_id=material.id,
            chunk_ids=chunk_ids,
            class_id=class_id,
            filename=file.filename
        )
        
        print(f"✅ Material uploaded: {file.filename} for class {cls.name}")
        
        return material
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error uploading material: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading material: {str(e)}")


@router.get("/{class_id}/materials", response_model=List[MaterialResponse])
async def get_class_materials(
    class_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all materials for a class
    """
    # Verify user has access to this class (professor or enrolled student)
    if current_user.role == 'professor':
        cls = db.query(Class).filter(
            Class.id == class_id,
            Class.professor_id == current_user.id
        ).first()
        if not cls:
            raise HTTPException(status_code=404, detail="Class not found")
    else:
        enrollment = db.query(ClassEnrollment).filter(
            ClassEnrollment.class_id == class_id,
            ClassEnrollment.student_id == current_user.id
        ).first()
        if not enrollment:
            raise HTTPException(status_code=403, detail="Not enrolled in this class")
    
    materials = db.query(ClassMaterial).filter(
        ClassMaterial.class_id == class_id
    ).all()
    
    return materials


@router.delete("/{class_id}/materials/{material_id}")
async def delete_class_material(
    class_id: int,
    material_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a class material (professor only)
    """
    if current_user.role != 'professor':
        raise HTTPException(status_code=403, detail="Only professors can delete class materials")
    
    # Verify professor owns this class
    cls = db.query(Class).filter(
        Class.id == class_id,
        Class.professor_id == current_user.id
    ).first()
    
    if not cls:
        raise HTTPException(status_code=404, detail="Class not found")
    
    material = db.query(ClassMaterial).filter(
        ClassMaterial.id == material_id,
        ClassMaterial.class_id == class_id
    ).first()
    
    if not material:
        raise HTTPException(status_code=404, detail="Material not found")
    
    try:
        # Get chunk IDs for ChromaDB deletion
        chunks = db.query(ClassMaterialChunk).filter(
            ClassMaterialChunk.material_id == material_id
        ).all()
        chunk_ids = [chunk.id for chunk in chunks]
        
        # Delete from ChromaDB
        rag_service = get_rag_service()
        rag_service.delete_class_material_chunks(chunk_ids)
        
        # Delete file
        if os.path.exists(material.firebase_storage_path):
            os.remove(material.firebase_storage_path)
        
        # Delete from database (cascades to chunks)
        db.delete(material)
        db.commit()
        
        print(f"✅ Material deleted: {material.filename}")
        
        return {"success": True, "message": "Material deleted successfully"}
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error deleting material: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting material: {str(e)}")

