"""
SQLAlchemy ORM models for PostgreSQL database
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index, Boolean, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="student")  # student or professor
    email = Column(String(255), unique=True, nullable=False, index=True)
    google_id = Column(String(255), unique=True, nullable=True, index=True)  # Nullable for professors
    username = Column(String(255), unique=True, nullable=True, index=True)  # For professors
    password_hash = Column(String(255), nullable=True)  # For professors
    must_change_password = Column(Boolean, default=False)  # For professors
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    pdf_documents = relationship("PdfDocument", back_populates="user", cascade="all, delete-orphan")
    created_classes = relationship("Class", back_populates="professor", cascade="all, delete-orphan")
    enrollments = relationship("ClassEnrollment", back_populates="student", cascade="all, delete-orphan")


class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    class_id = Column(Integer, ForeignKey("classes.id", ondelete="SET NULL"), nullable=True, index=True)  # Class context
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    sources_json = Column(Text, nullable=True)  # JSON string with source chunks

    # Relationships
    user = relationship("User", back_populates="chats")
    class_context = relationship("Class", back_populates="chats")
    feedback = relationship("Feedback", back_populates="chat", uselist=False, cascade="all, delete-orphan")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey("chats.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    satisfaction_score = Column(Integer, nullable=False)  # -1 (thumbs down), 0 (neutral), 1 (thumbs up)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chat = relationship("Chat", back_populates="feedback")


class PdfDocument(Base):
    __tablename__ = "pdf_documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    class_id = Column(Integer, ForeignKey("classes.id", ondelete="CASCADE"), nullable=True, index=True)  # Associate PDF with class
    filename = Column(String(500), nullable=False)
    firebase_storage_path = Column(Text, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="pdf_documents")
    chunks = relationship("PdfChunk", back_populates="pdf_document", cascade="all, delete-orphan")
    class_context = relationship("Class", foreign_keys=[class_id])


class PdfChunk(Base):
    __tablename__ = "pdf_chunks"

    id = Column(Integer, primary_key=True, index=True)
    pdf_id = Column(Integer, ForeignKey("pdf_documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    metadata_json = Column(Text, nullable=True)  # JSON string with additional metadata

    # Relationships
    pdf_document = relationship("PdfDocument", back_populates="chunks")

    __table_args__ = (
        Index('idx_pdf_chunk', 'pdf_id', 'chunk_index'),
    )


class Class(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    class_code = Column(String(50), unique=True, nullable=False, index=True)  # Unique join code
    professor_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    professor = relationship("User", back_populates="created_classes")
    enrollments = relationship("ClassEnrollment", back_populates="class_obj", cascade="all, delete-orphan")
    materials = relationship("ClassMaterial", back_populates="class_obj", cascade="all, delete-orphan")
    chats = relationship("Chat", back_populates="class_context")


class ClassEnrollment(Base):
    __tablename__ = "class_enrollments"

    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, ForeignKey("classes.id", ondelete="CASCADE"), nullable=False, index=True)
    student_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    joined_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    class_obj = relationship("Class", back_populates="enrollments")
    student = relationship("User", back_populates="enrollments")

    # Unique constraint: a student can only enroll once per class
    __table_args__ = (
        UniqueConstraint('class_id', 'student_id', name='uq_class_student'),
    )


class ClassMaterial(Base):
    __tablename__ = "class_materials"

    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, ForeignKey("classes.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(500), nullable=False)
    firebase_storage_path = Column(Text, nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    uploaded_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    # Relationships
    class_obj = relationship("Class", back_populates="materials")
    chunks = relationship("ClassMaterialChunk", back_populates="material", cascade="all, delete-orphan")


class ClassMaterialChunk(Base):
    __tablename__ = "class_material_chunks"

    id = Column(Integer, primary_key=True, index=True)
    material_id = Column(Integer, ForeignKey("class_materials.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    metadata_json = Column(Text, nullable=True)  # JSON string with additional metadata

    # Relationships
    material = relationship("ClassMaterial", back_populates="chunks")

    __table_args__ = (
        Index('idx_material_chunk', 'material_id', 'chunk_index'),
    )

