#!/usr/bin/env python3
"""
Script to create professor credentials in the database
This is the final, clean version that works reliably.

Usage:
    python create_professor_credentials.py

This will:
1. Drop and recreate all database tables with the new schema
2. Create a professor account with credentials:
   - Username: prof1
   - Password: temp123
   - Email: prof@university.edu
   - Must change password on first login
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.database import engine, Base, SessionLocal
from app.models import User, Chat, Feedback, PdfDocument, PdfChunk, Class, ClassEnrollment, ClassMaterial, ClassMaterialChunk
from sqlalchemy import text

def create_professor_credentials():
    """
    Create professor credentials with the new database schema
    This is the final, working approach that:
    1. Drops all tables and recreates with new schema
    2. Creates professor account with direct SQL insertion
    """
    print("🔧 Setting up professor credentials...")
    
    # Step 1: Drop all existing tables
    print("🗑️  Dropping all existing tables...")
    Base.metadata.drop_all(bind=engine)
    print("✅ All tables dropped")
    
    # Step 2: Create all tables with new schema
    print("🔧 Creating tables with new schema...")
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created with new schema")
    
    # Step 3: Create professor account
    print("👨‍🏫 Creating professor account...")
    db = SessionLocal()
    
    try:
        # Insert professor directly with SQL (most reliable approach)
        db.execute(text("""
            INSERT INTO users (username, password_hash, name, email, role, must_change_password, google_id, created_at)
            VALUES ('prof1', '$2b$12$CVISR7p.bUysTcJGU2Wx5uK18hkHfwnPcyhXnjuLtUaYIuno3WOf2', 'Professor Smith', 'prof@university.edu', 'professor', true, 'N/A', NOW())
        """))
        
        db.commit()
        
        print("✅ Professor account created successfully!")
        print("=" * 50)
        print("📋 PROFESSOR CREDENTIALS:")
        print("   Username: prof1")
        print("   Password: temp123")
        print("   Email: prof@university.edu")
        print("   ⚠️  Must change password on first login")
        print("=" * 50)
        print("🚀 Ready to use! Start the backend and test login.")
        
    except Exception as e:
        print(f"❌ Error creating professor: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    create_professor_credentials()
