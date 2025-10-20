#!/bin/bash

# Development startup script for backend

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Run setup first."
    exit 1
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Creating from example..."
    if [ -f ".env.example.txt" ]; then
        cp .env.example.txt .env
        echo "📝 Created .env file. Please edit it with your configuration."
        echo "Required: FIREBASE_CREDENTIALS_PATH, GEMINI_API_KEY"
        exit 1
    fi
fi

# Create data directories
mkdir -p data/chroma_db data/uploads

# Run the backend
echo "🚀 Starting FastAPI backend..."
python -m app.main

