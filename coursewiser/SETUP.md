# CourseWiser Setup Guide

## Quick Start Guide

### 1. Clone and Navigate

```bash
cd /Users/aniketpatel/Desktop/CS460/coursewiser
```

### 2. Start Database

```bash
docker-compose up -d
```

Verify it's running:
```bash
docker-compose ps
```

### 3. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
DATABASE_URL=postgresql://coursewiser:coursewiser123@localhost:5432/coursewiser
FIREBASE_CREDENTIALS_PATH=/path/to/your/serviceAccountKey.json
GEMINI_API_KEY=your_gemini_api_key
MERGED_MODEL_PATH=/Users/aniketpatel/Desktop/CS460/final_model
CHROMA_PERSIST_DIR=./data/chroma_db
HOST=0.0.0.0
PORT=8000
EOF

# Create data directories
mkdir -p data/chroma_db data/uploads

# Run backend
python -m app.main
```

### 4. Frontend Setup

Open a new terminal:

```bash
cd /Users/aniketpatel/Desktop/CS460/coursewiser/frontend

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
VITE_FIREBASE_API_KEY=your_api_key
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=your-app-id
VITE_API_URL=http://localhost:8000
EOF

# Run frontend
npm run dev
```

### 5. Access the Application

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- pgAdmin: http://localhost:5050 (admin@coursewiser.com / admin123)

## Firebase Setup Instructions

### Get Firebase Configuration

1. Go to https://console.firebase.google.com
2. Create a new project or select existing
3. Click the gear icon > Project settings
4. Scroll down to "Your apps" section
5. Click the web icon (</>) to create a web app
6. Copy the configuration values to frontend `.env`

### Enable Google Sign-In

1. In Firebase Console, go to Authentication
2. Click "Get started" if not already enabled
3. Click "Sign-in method" tab
4. Click "Google" provider
5. Toggle "Enable"
6. Add your support email
7. Click "Save"

### Download Service Account Key

1. In Firebase Console, go to Project settings
2. Click "Service accounts" tab
3. Click "Generate new private key"
4. Download the JSON file
5. Save it securely (e.g., `~/firebase-keys/serviceAccountKey.json`)
6. Update `FIREBASE_CREDENTIALS_PATH` in backend `.env`

## Google Gemini API Key

1. Go to https://makersuite.google.com/app/apikey
2. Click "Create API key"
3. Copy the key
4. Add to backend `.env` as `GEMINI_API_KEY`

## Testing the Setup

### Test Backend

```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","database":"connected","model":"ready"}
```

### Test Database

```bash
docker exec -it coursewiser_postgres psql -U coursewiser -d coursewiser -c "\dt"
# Should list tables: users, chats, feedback, pdf_documents, pdf_chunks
```

### Test Frontend

1. Open http://localhost:5173
2. Should see login page
3. Try signing in with Google

## Troubleshooting

### Backend won't start

**Error: Model not found**
- Check `MERGED_MODEL_PATH` in .env
- Verify the model exists at that location

**Error: Database connection failed**
- Check if PostgreSQL is running: `docker-compose ps`
- Verify DATABASE_URL in .env

**Error: Firebase Admin SDK initialization failed**
- Check `FIREBASE_CREDENTIALS_PATH` in .env
- Verify the file exists and is valid JSON

### Frontend won't start

**Error: Cannot connect to backend**
- Check if backend is running on port 8000
- Verify `VITE_API_URL` in frontend .env

**Error: Firebase auth not working**
- Verify all Firebase config variables in .env
- Check Firebase console that Google Sign-In is enabled

### Model loading is slow

The model is loaded lazily on first request to save memory. First chat request may take 30-60 seconds while the model loads. Subsequent requests will be much faster.

### ChromaDB errors

If ChromaDB shows corruption errors:
```bash
cd backend
rm -rf data/chroma_db
# Restart backend - it will recreate the database
```

## Development Tips

### Backend Auto-reload

The backend runs with `reload=True` by default, so it will auto-reload on code changes.

### Frontend Hot Reload

Vite provides instant hot module replacement (HMR). Changes appear immediately in the browser.

### Database Management

Access pgAdmin at http://localhost:5050:
- Email: admin@coursewiser.com
- Password: admin123

Add server connection:
- Host: postgres (or localhost)
- Port: 5432
- Username: coursewiser
- Password: coursewiser123
- Database: coursewiser

### View Logs

Backend logs:
```bash
# Check for errors in terminal where backend is running
```

Database logs:
```bash
docker-compose logs -f postgres
```

### Reset Everything

To start fresh:
```bash
# Stop containers
docker-compose down -v

# Remove backend data
cd backend
rm -rf data/

# Restart
docker-compose up -d
cd backend && python -m app.main
```

