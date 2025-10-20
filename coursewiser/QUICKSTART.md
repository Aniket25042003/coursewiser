# CourseWiser - Quick Start Guide

## ⚡ 5-Minute Setup

### Prerequisites Check
```bash
python3 --version  # Should be 3.9+
node --version     # Should be 18+
docker --version   # Should be installed
```

### Step 1: Start Database (30 seconds)
```bash
cd /Users/aniketpatel/Desktop/CS460/coursewiser
docker-compose up -d
```

### Step 2: Configure Environment

**Backend** - Create `backend/.env`:
```bash
cd backend
cat > .env << 'EOF'
DATABASE_URL=postgresql://coursewiser:coursewiser123@localhost:5432/coursewiser
FIREBASE_CREDENTIALS_PATH=/path/to/your/serviceAccountKey.json
GEMINI_API_KEY=your_gemini_api_key_here
MERGED_MODEL_PATH=/Users/aniketpatel/Desktop/CS460/final_model
CHROMA_PERSIST_DIR=./data/chroma_db
HOST=0.0.0.0
PORT=8000
EOF
```

**Frontend** - Create `frontend/.env`:
```bash
cd ../frontend
cat > .env << 'EOF'
VITE_FIREBASE_API_KEY=your_firebase_api_key
VITE_FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=your-project-id
VITE_FIREBASE_STORAGE_BUCKET=your-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=your-app-id
VITE_API_URL=http://localhost:8000
EOF
```

### Step 3: Install Dependencies

**Backend** (2 minutes):
```bash
cd /Users/aniketpatel/Desktop/CS460/coursewiser/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Frontend** (1 minute):
```bash
cd /Users/aniketpatel/Desktop/CS460/coursewiser/frontend
npm install
```

### Step 4: Start Services

**Terminal 1 - Backend**:
```bash
cd /Users/aniketpatel/Desktop/CS460/coursewiser/backend
source venv/bin/activate
python -m app.main
```

**Terminal 2 - Frontend**:
```bash
cd /Users/aniketpatel/Desktop/CS460/coursewiser/frontend
npm run dev
```

### Step 5: Access Application

- 🌐 **Frontend**: http://localhost:5173
- 🔧 **Backend**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs
- 🗄️ **pgAdmin**: http://localhost:5050

## 🔥 First-Time Use

1. Open http://localhost:5173
2. Click "Sign in with Google"
3. Select "Student" or "Professor"
4. Start chatting!

## 📝 What You Need

### Essential (Get These First)
1. **Firebase Service Account Key**
   - Go to https://console.firebase.google.com
   - Project Settings → Service Accounts
   - Click "Generate new private key"
   - Save as `serviceAccountKey.json`

2. **Firebase Web Config**
   - Firebase Console → Project Settings
   - Scroll to "Your apps"
   - Copy config values to frontend `.env`

3. **Gemini API Key**
   - Visit https://makersuite.google.com/app/apikey
   - Create API key
   - Add to backend `.env`

### Optional (For Later)
- AWS account for deployment
- Custom domain
- SSL certificate

## 🧪 Testing the Setup

### Test Backend
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","database":"connected","model":"ready"}
```

### Test Database
```bash
docker exec -it coursewiser_postgres psql -U coursewiser -d coursewiser -c "\dt"
# Expected: List of tables (users, chats, feedback, pdf_documents, pdf_chunks)
```

### Test Frontend
1. Open http://localhost:5173
2. Should see login page
3. Try signing in

## 🐛 Common Issues

### "Model not found"
**Fix**: Check `MERGED_MODEL_PATH` in backend `.env`
```bash
ls /Users/aniketpatel/Desktop/CS460/final_model
# Should show model files
```

### "Database connection failed"
**Fix**: Ensure PostgreSQL is running
```bash
docker-compose ps
# Should show postgres as "Up"
```

### "Firebase error"
**Fix**: Verify credentials
- Backend: Check `FIREBASE_CREDENTIALS_PATH` points to valid JSON
- Frontend: Check all `VITE_FIREBASE_*` variables are set

### "Port already in use"
**Fix**: Change ports in `.env` files
- Backend: Change `PORT=8000` to `PORT=8001`
- Frontend: Change in `vite.config.ts`

## 🚀 Next Steps

After setup works:

1. **Upload a PDF** (Student view)
   - Click "Upload PDF" in sidebar
   - Drop a DSA textbook PDF
   - Wait for processing

2. **Ask Questions** (Student view)
   - Try: "Explain binary search trees"
   - Check sources in response
   - Give feedback (👍/👎)

3. **View Analytics** (Professor view)
   - Sign in as Professor
   - Check feedback statistics
   - Generate AI summary
   - Export CSV

## 📖 More Documentation

- **README.md** - Full documentation
- **SETUP.md** - Detailed setup instructions
- **ARCHITECTURE.md** - Technical details
- **PROJECT_SUMMARY.md** - What was built

## 💡 Pro Tips

1. **Model loads on first request** - First chat will take 30-60s
2. **Use GPU if available** - Much faster inference
3. **Test with small PDFs first** - Large PDFs take time to process
4. **Keep terminal windows open** - To see logs
5. **Check API docs** - http://localhost:8000/docs for API exploration

## 🆘 Need Help?

1. Check terminal logs for errors
2. Verify all environment variables are set
3. Ensure all services are running
4. Review SETUP.md for detailed troubleshooting

## 🎯 Quick Commands Reference

```bash
# Start everything
docker-compose up -d
cd backend && source venv/bin/activate && python -m app.main &
cd frontend && npm run dev

# Stop everything
docker-compose down
# Ctrl+C in backend terminal
# Ctrl+C in frontend terminal

# Reset database
docker-compose down -v
docker-compose up -d

# View logs
docker-compose logs -f postgres
# Check terminal where backend/frontend is running

# Rebuild frontend
cd frontend && npm run build

# Test API
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

---

**Ready to start?** Follow the 5 steps above, then open http://localhost:5173 🚀

