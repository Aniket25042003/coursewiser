#!/bin/bash

# Development startup script for frontend

cd "$(dirname "$0")"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. Creating from example..."
    if [ -f ".env.example.txt" ]; then
        cp .env.example.txt .env
        echo "📝 Created .env file. Please edit it with your Firebase configuration."
        exit 1
    fi
fi

# Run the frontend
echo "🚀 Starting Vite dev server..."
npm run dev

