# CourseWiser - Intelligent Learning Assistant

A full-stack web application that provides an intelligent, class-based learning system powered by a fine-tuned LLaMA model with RAG (Retrieval-Augmented Generation) capabilities. CourseWiser supports role-based authentication for students and professors, multi-class management, and context-aware AI assistance.

## 🎯 Overview

CourseWiser transforms traditional learning by providing:
- **For Students**: Multi-class enrollment, AI-powered chat with class materials, supplementary note uploads
- **For Professors**: Class management, material uploads, per-class analytics, AI insights
- **For Administrators**: Secure professor account creation, system monitoring

## ✨ Key Features

### 🎓 Student Features
- **Google Sign-In Authentication** - Secure login with Firebase
- **Multi-Class Support** - Join multiple classes with unique codes
- **Class-Based Access Control** - Must join a class before chatting
- **ChatGPT-Style Interface** - Modern, one-page chat layout
- **Context-Aware AI** - Uses both class materials and personal notes
- **Supplementary Uploads** - Upload your own PDF notes
- **Real-Time Feedback** - Rate AI responses with comments
- **Chat History** - View all conversations
- **Class Switching** - Easy navigation between enrolled classes

### 👨‍🏫 Professor Features
- **Secure Authentication** - Username/password with JWT tokens
- **Class Management** - Create unlimited classes with auto-generated codes
- **Material Upload** - Upload course PDFs for AI context
- **Student Monitoring** - View enrolled students per class
- **Per-Class Analytics** - Detailed statistics and insights
- **AI-Powered Summaries** - Gemini-generated insights
- **Data Export** - CSV downloads for analysis
- **Password Security** - Forced password change on first login

### 🔐 Security Features
- **Role-Based Access Control** - Separate authentication flows
- **Password Hashing** - bcrypt for secure storage
- **JWT Token Management** - 24-hour expiration
- **Class Ownership Verification** - Professors manage only their classes
- **Enrollment Validation** - Students access only enrolled classes
- **Protected API Endpoints** - Secure backend access

## 🏗️ Architecture

### System Components
```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                             │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Login    │  │   Student    │  │  Professor   │        │
│  │  (Google)  │  │     Page     │  │  Dashboard   │        │
│  └────────────┘  └──────────────┘  └──────────────┘        │
│         │                │                  │                │
│         └────────────────┴──────────────────┘                │
│                  ┌───────▼────────┐                          │
│                  │  Firebase SDK  │                          │
│                  │  (Auth + Storage)                         │
│                  └────────────────┘                          │
└──────────────────────────│──────────────────────────────────┘
                           │ HTTP + Bearer Token
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                      BACKEND (FastAPI)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     Auth     │  │     Chat     │  │     PDF      │      │
│  │   Endpoints  │  │   Endpoints  │  │   Endpoints  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Feedback   │  │   Professor  │  │   Classes   │      │
│  │   Endpoints  │  │   Endpoints  │  │   Endpoints  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                  │                  │              │
│         └──────────────────┴──────────────────┘              │
│                           │                                  │
│         ┌─────────────────┴─────────────────┐               │
│         │                                     │               │
│    ┌────▼─────┐  ┌──────────┐  ┌───────────▼──┐           │
│    │PostgreSQL│  │ChromaDB  │  │  Model       │            │
│    │   ORM    │  │  (RAG)   │  │  Wrapper     │            │
│    └──────────┘  └──────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────┘
         │               │                  │
         │               │                  │
    ┌────▼────┐    ┌────▼────┐      ┌─────▼──────┐
    │PostgreSQL│   │ ChromaDB│      │  LLaMA 3.2 │
    │  Docker │    │  Local  │      │ Fine-tuned │
    └─────────┘    └─────────┘      │   Model    │
                                     └────────────┘
         │
         └─────────► Gemini API (Professor Summaries)
```

### Database Schema (9 Tables)
- **users** - Extended with username, password_hash, must_change_password
- **classes** - Stores class information and unique codes
- **class_enrollments** - Links students to classes
- **class_materials** - Stores uploaded course PDFs
- **class_material_chunks** - Indexed chunks for RAG
- **chats** - Extended with class_id for context
- **feedback** - Student ratings and comments
- **pdf_documents** - Personal student uploads
- **pdf_chunks** - Personal PDF chunks for RAG

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- Firebase project with Google Sign-In enabled
- Google Gemini API key
- Fine-tuned LLaMA model

### 1. Start Database
```bash
cd /Users/aniketpatel/Desktop/CS460/coursewiser
docker-compose up -d
```

### 2. Backend Setup
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

### 3. Frontend Setup
```bash
cd ../frontend

# Install dependencies
npm install

# Create .env file
cat > .env << EOF
VITE_FIREBASE_API_KEY=your_firebase_api_key
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

### 4. Create Professor Account
```bash
cd backend
source venv/bin/activate
python create_professor_credentials.py
```

**Professor Credentials:**
- Username: `prof1`
- Password: `temp123` (must change on first login)
- Email: `prof@university.edu`

### 5. Access Application
- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **pgAdmin**: http://localhost:5050 (admin@coursewiser.com / admin123)

## 🔧 Configuration

### Firebase Setup
1. Go to https://console.firebase.google.com
2. Create a new project or select existing
3. Enable Google Sign-In in Authentication > Sign-in method
4. Get Firebase config for frontend (.env variables)
5. Download service account key for backend (Project Settings > Service Accounts)

### Google Gemini API
1. Get API key from https://makersuite.google.com/app/apikey
2. Add to backend `.env` as `GEMINI_API_KEY`

## 📖 Usage Guide

### For Professors

#### 1. Login and Setup
1. Go to http://localhost:5173/login
2. Toggle to "Professor" tab
3. Login with `prof1` / `temp123`
4. Change password (required on first login)

#### 2. Create Classes
1. Navigate to "Class Management" tab
2. Click "Create New Class"
3. Enter class name and description
4. **Copy the generated class code** (e.g., `TWEB9GY3`)
5. Share code with students

#### 3. Upload Materials
1. Click "Materials" on class card
2. Upload PDF files
3. Materials are automatically indexed for AI context
4. View/delete materials as needed

#### 4. Monitor Analytics
1. Go to "Analytics Dashboard" tab
2. Select class from dropdown
3. View feedback statistics
4. Generate AI insights with Gemini
5. Export data as CSV

### For Students

#### 1. Login and Join Class
1. Go to http://localhost:5173/login
2. Click "Sign in with Google"
3. Click "Join Your First Class"
4. Enter class code from professor
5. Select active class from dropdown

#### 2. Chat with AI
1. Ask questions in the ChatGPT-style interface
2. AI uses professor's materials + your notes
3. Provide feedback (👍/👎) on responses
4. View source attribution

#### 3. Upload Personal Notes
1. Click 📎 button in chat input
2. Upload supplementary PDFs
3. Notes supplement class materials
4. View in "📄 PDFs" tab

#### 4. Join Additional Classes
1. Click "+ Join Class" button
2. Enter new class code
3. Switch between classes using dropdown

## 🎨 UI/UX Features

### Modern Design Elements
- **Gradient Backgrounds** - Indigo/Purple/Pink color scheme
- **Glassmorphism Effects** - Frosted glass with backdrop blur
- **Smooth Animations** - Hover effects, scale transitions
- **ChatGPT-Style Chat** - One-page layout, fixed input area
- **Responsive Design** - Works on desktop and mobile

### Interactive Elements
- **Animated Blobs** - Organic floating backgrounds
- **Hover Effects** - Scale animations on buttons and cards
- **Success Feedback** - Bounce animations on copy actions
- **Loading States** - Spinners and disabled states
- **Error Handling** - User-friendly error messages

## 🔧 Technical Stack

### Backend
- **FastAPI** - Modern Python web framework
- **PostgreSQL** - Relational database with Docker
- **ChromaDB** - Vector database for embeddings
- **SQLAlchemy** - ORM for database operations
- **PyTorch + Transformers** - Model inference
- **sentence-transformers** - Embedding generation
- **Firebase Admin SDK** - Authentication
- **Google Gemini API** - AI insights
- **JWT + bcrypt** - Security

### Frontend
- **React 18 + TypeScript** - Modern UI framework
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **React Router** - Client-side routing
- **Firebase SDK** - Authentication and storage
- **Axios** - HTTP client
- **Lucide React** - Icon library

## 📊 API Endpoints

### Authentication
- `POST /api/auth/verify` - Verify Firebase token (students)
- `POST /api/auth/professor/login` - Professor login (JWT)
- `POST /api/auth/professor/change_password` - Password update

### Class Management
- `POST /api/classes` - Create class (professor)
- `GET /api/classes/my_classes` - List professor's classes
- `PUT /api/classes/{id}` - Update class
- `DELETE /api/classes/{id}` - Delete class
- `POST /api/classes/join` - Join class with code (student)
- `GET /api/classes/enrolled` - List enrolled classes (student)

### Materials
- `POST /api/classes/{id}/materials` - Upload class material
- `GET /api/classes/{id}/materials` - List materials
- `DELETE /api/classes/{id}/materials/{mid}` - Delete material
- `POST /api/pdf/upload_file` - Upload personal PDF (student)

### Chat & Analytics
- `POST /api/chat` - Send message with RAG
- `GET /api/history` - Get chat history
- `GET /api/professor/stats` - Feedback statistics
- `GET /api/professor/summary` - AI insights
- `GET /api/professor/export_csv` - Export data

## 🧪 Testing

### Test Professor Flow
1. Login with `prof1` / `temp123`
2. Change password (required)
3. Create a class and get class code
4. Upload materials
5. View analytics

### Test Student Flow
1. Login with Google
2. Join class with code
3. Select active class
4. Chat with AI
5. Upload personal notes
6. Provide feedback

### Test Multi-Class Support
1. Join multiple classes
2. Switch between classes
3. Verify class-specific context
4. Upload class-specific notes

## 🚀 Deployment

### Production Setup
1. Set `JWT_SECRET_KEY` environment variable
2. Configure Firebase credentials
3. Set `GEMINI_API_KEY`
4. Point to production database
5. Configure CORS for production domain
6. Set up SSL/HTTPS

### AWS Deployment
- **EC2**: Backend hosting (t2.xlarge or GPU instances)
- **RDS**: Managed PostgreSQL
- **S3**: Static frontend hosting
- **CloudFront**: CDN for frontend
- **ALB**: Load balancing

## 🐛 Troubleshooting

### Common Issues

**Backend won't start**
- Check `MERGED_MODEL_PATH` in .env
- Verify PostgreSQL is running: `docker-compose ps`
- Check Firebase credentials path

**Frontend won't start**
- Verify all Firebase config variables in .env
- Check if backend is running on port 8000
- Ensure `VITE_API_URL` is correct

**Model loading is slow**
- First request takes 30-60s while model loads
- Subsequent requests are much faster
- Use GPU if available for better performance

**PDF upload fails**
- Check file size (large PDFs may timeout)
- Verify file is valid PDF
- Check backend logs for errors

### Reset Everything
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

## 📈 Performance

### Build Metrics
- **Build time**: ~1.5 seconds
- **Bundle size**: 433 KB (gzipped: 115 KB)
- **CSS size**: 27 KB (gzipped: 5.3 KB)
- **Load time**: < 1 second

### Optimizations
- **Model Loading**: Lazy initialization, 4-bit quantization
- **Database**: Indexed columns, connection pooling
- **Vector Search**: Efficient similarity search with HNSW
- **Frontend**: Code splitting, lazy loading

## 🔮 Future Enhancements

### Planned Features
- **Multi-document chat**: Reference multiple PDFs simultaneously
- **Code execution**: Run code snippets safely
- **Video explanations**: Link to relevant video content
- **Progress tracking**: Student learning analytics
- **Mobile app**: Native iOS/Android apps

### Technical Improvements
- **Caching layer**: Redis for frequently asked questions
- **Streaming responses**: WebSocket for real-time generation
- **Auto-scaling**: Based on load metrics
- **CI/CD pipeline**: Automated testing and deployment

## 📚 File Structure

```
coursewiser/
├── backend/
│   ├── app/
│   │   ├── api/              # API endpoints
│   │   ├── services/         # Business logic
│   │   ├── models.py         # Database models
│   │   ├── database.py       # DB configuration
│   │   └── main.py           # FastAPI app
│   ├── requirements.txt      # Python dependencies
│   └── create_professor_credentials.py
├── frontend/
│   ├── src/
│   │   ├── components/       # Reusable components
│   │   ├── pages/            # Student & Professor pages
│   │   ├── services/         # API & Firebase services
│   │   ├── App.tsx           # Main app component
│   │   └── main.tsx          # Entry point
│   ├── package.json
│   └── vite.config.ts
├── docker-compose.yml         # PostgreSQL container
└── README.md                  # This file
```

## 📞 Troubleshoot

For issues or questions:
1. Check terminal logs for errors
2. Verify all environment variables are set
3. Ensure all services are running
4. Review this README for troubleshooting

## 📄 License

MIT License

## 👥 Contributors

- Aniket Patel

---

**CourseWiser is production-ready and fully functional!** 🚀

The application provides a complete learning management system with intelligent AI assistance, role-based authentication, and beautiful modern UI. Students can join multiple classes, chat with context-aware AI, and upload supplementary materials. Professors can create classes, upload materials, monitor analytics, and track student engagement.

**Ready for educational institutions to deploy and use!** 🎓