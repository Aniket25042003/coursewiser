# CourseWiser - DSA Q&A Application

A full-stack web application that allows students to ask Data Structures & Algorithms questions to a fine-tuned LLaMA model with RAG (Retrieval-Augmented Generation) support via PDF uploads.

## Features

### Student Features
- 🔐 Google Sign-In via Firebase Authentication
- 💬 Chat interface with fine-tuned LLaMA model
- 📄 PDF upload for context-aware responses (RAG)
- 📚 Chat history tracking
- 👍👎 Feedback system (thumbs up/down with optional comments)
- 🔍 Source citation for answers

### Professor Features
- 📊 Analytics dashboard with feedback metrics
- 📉 View all low-rated Q&A pairs
- 🤖 AI-powered insights using Google Gemini API
- 📥 Export feedback data as CSV
- 📈 Track student engagement and satisfaction

## Tech Stack

### Frontend
- React 18 with TypeScript
- Vite (build tool)
- Tailwind CSS (styling)
- React Router (routing)
- Firebase SDK (authentication)
- Axios (API communication)
- Lucide React (icons)

### Backend
- FastAPI (Python web framework)
- PostgreSQL (relational database)
- ChromaDB (vector database for embeddings)
- Firebase Admin SDK (token verification)
- PyTorch + Transformers (model inference)
- sentence-transformers (embeddings)
- LangChain (text splitting)
- PyMuPDF (PDF processing)
- Google Gemini API (professor summaries)

## Project Structure

```
coursewiser/
├── frontend/                  # React + TypeScript frontend
│   ├── src/
│   │   ├── components/       # Reusable components
│   │   ├── pages/            # Student & Professor pages
│   │   ├── services/         # API & Firebase services
│   │   ├── App.tsx           # Main app component
│   │   └── main.tsx          # Entry point
│   ├── package.json
│   └── vite.config.ts
├── backend/                   # FastAPI backend
│   ├── app/
│   │   ├── api/              # API endpoints
│   │   ├── services/         # Business logic
│   │   ├── models.py         # Database models
│   │   ├── database.py       # DB configuration
│   │   └── main.py           # FastAPI app
│   └── requirements.txt
├── docker-compose.yml         # PostgreSQL container
└── README.md
```

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- Firebase project with Google Sign-In enabled
- Google Gemini API key
- Fine-tuned LLaMA model

### 1. Database Setup

Start PostgreSQL using Docker Compose:

```bash
cd coursewiser
docker-compose up -d
```

This will start:
- PostgreSQL on port 5432
- pgAdmin on port 5050 (optional, for database management)

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (copy from .env.example.txt)
cp .env.example.txt .env

# Edit .env with your configuration:
# - DATABASE_URL
# - FIREBASE_CREDENTIALS_PATH (download from Firebase Console)
# - GEMINI_API_KEY
# - MERGED_MODEL_PATH (path to your fine-tuned model)

# Run the backend
python -m app.main
```

The backend will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env file (copy from .env.example.txt)
cp .env.example.txt .env

# Edit .env with your Firebase configuration
# Get these values from Firebase Console > Project Settings

# Run the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

### 4. Firebase Configuration

1. Create a Firebase project at https://console.firebase.google.com
2. Enable Google Sign-In in Authentication > Sign-in method
3. Get Firebase config for frontend (.env variables)
4. Download service account key for backend (Project Settings > Service Accounts)
5. Update Firebase Storage rules if using file upload

### 5. Google Gemini API Setup

1. Get API key from https://makersuite.google.com/app/apikey
2. Add to backend `.env` as `GEMINI_API_KEY`

## Usage

### For Students

1. Go to `http://localhost:5173`
2. Click "Sign in with Google"
3. Select "Student" role
4. Upload PDFs (optional) for context-aware answers
5. Ask DSA questions in the chat
6. View sources and provide feedback on answers

### For Professors

1. Sign in with Google
2. Select "Professor" role
3. View analytics dashboard with feedback statistics
4. Click "Generate Summary" to get AI insights on common issues
5. Review low-rated Q&A pairs
6. Export feedback data as CSV

## Database Schema

```sql
users (
  id SERIAL PRIMARY KEY,
  name TEXT,
  role TEXT,  -- 'student' or 'professor'
  email TEXT UNIQUE,
  google_id TEXT UNIQUE,
  created_at TIMESTAMP
)

chats (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  message TEXT,
  response TEXT,
  timestamp TIMESTAMP,
  sources_json TEXT
)

feedback (
  id SERIAL PRIMARY KEY,
  chat_id INTEGER REFERENCES chats(id),
  satisfaction_score INTEGER,  -- -1 (thumbs down), 1 (thumbs up)
  comment TEXT,
  created_at TIMESTAMP
)

pdf_documents (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  filename TEXT,
  firebase_storage_path TEXT,
  upload_timestamp TIMESTAMP
)

pdf_chunks (
  id SERIAL PRIMARY KEY,
  pdf_id INTEGER REFERENCES pdf_documents(id),
  chunk_text TEXT,
  chunk_index INTEGER,
  metadata_json TEXT
)
```

## API Endpoints

### Authentication
- `POST /api/auth/verify` - Verify Firebase token and create/get user

### Chat
- `POST /api/chat` - Send message and get AI response
- `GET /api/history` - Get chat history
- `DELETE /api/history/clear` - Clear all history

### PDF
- `POST /api/pdf/upload_file` - Upload and process PDF
- `GET /api/pdf/user_pdfs` - Get user's uploaded PDFs
- `DELETE /api/pdf/pdf/{pdf_id}` - Delete PDF

### Feedback
- `POST /api/feedback/submit` - Submit feedback for a chat
- `GET /api/feedback/chat/{chat_id}` - Get feedback for a chat

### Professor
- `GET /api/professor/stats` - Get feedback statistics
- `GET /api/professor/low_rated` - Get low-rated chats
- `GET /api/professor/summary` - Get Gemini AI summary
- `GET /api/professor/export_csv` - Export feedback as CSV

## Development

### Backend Development

```bash
cd backend
source venv/bin/activate
python -m app.main
```

### Frontend Development

```bash
cd frontend
npm run dev
```

### Build for Production

Frontend:
```bash
cd frontend
npm run build
# Output in dist/
```

Backend: Use Docker or deploy directly to EC2/server

## Deployment (AWS EC2)

### Backend Deployment

1. Launch EC2 instance (Ubuntu 22.04, t2.large or larger for model)
2. Install dependencies:
   ```bash
   sudo apt update
   sudo apt install python3.9 python3-pip postgresql-client
   ```
3. Clone repository and set up backend
4. Configure environment variables
5. Use systemd service for auto-restart:
   ```bash
   sudo systemctl enable coursewiser-backend
   sudo systemctl start coursewiser-backend
   ```

### Database Deployment

1. Create AWS RDS PostgreSQL instance
2. Update `DATABASE_URL` in backend .env
3. Run migrations

### Frontend Deployment

1. Build frontend: `npm run build`
2. Serve with Nginx or deploy to S3 + CloudFront
3. Update CORS settings in backend

## Troubleshooting

### Model Loading Issues
- Ensure GPU is available or use CPU mode
- Check model path in .env
- Verify sufficient RAM/VRAM (8GB+ recommended)

### Database Connection Issues
- Verify PostgreSQL is running: `docker-compose ps`
- Check DATABASE_URL in .env
- Ensure port 5432 is not blocked

### Firebase Authentication Issues
- Verify Firebase config in frontend .env
- Check service account key path in backend .env
- Ensure Firebase Auth is enabled in console

### ChromaDB Issues
- Check CHROMA_PERSIST_DIR exists and has write permissions
- Clear ChromaDB data if corrupted: `rm -rf data/chroma_db`

## License

MIT License

## Contributors

- Aniket Patel

## Support

For issues or questions, please contact [your-email@example.com]

