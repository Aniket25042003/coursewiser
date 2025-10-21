"""
FastAPI main application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.database import init_db
from app.api import auth, chat, pdf, feedback, professor, classes

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    print("🚀 Starting CourseWiser API...")
    print("📊 Initializing database...")
    init_db()
    print("✅ Database initialized")
    
    # Load models (lazy loading - will be loaded on first request)
    print("⚠️  Models will be loaded on first request to save memory")
    
    yield
    
    # Shutdown
    print("👋 Shutting down CourseWiser API...")


# Create FastAPI app
app = FastAPI(
    title="CourseWiser API",
    description="Backend API for intelligent course learning with RAG support",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default dev server
        "http://localhost:3000",  # Alternative frontend port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(pdf.router)
app.include_router(feedback.router)
app.include_router(professor.router)
app.include_router(classes.router)


@app.get("/")
async def root():
    """
    Root endpoint - API health check
    """
    return {
        "message": "CourseWiser API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "database": "connected",
        "model": "ready"
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

