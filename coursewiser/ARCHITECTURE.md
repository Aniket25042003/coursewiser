# CourseWiser Architecture Documentation

## System Overview

CourseWiser is a full-stack web application that provides an intelligent DSA (Data Structures & Algorithms) tutoring system powered by a fine-tuned LLaMA model with RAG (Retrieval-Augmented Generation) capabilities.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         FRONTEND                             │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Login    │  │   Student    │  │  Professor   │        │
│  │  (Google)  │  │     Page     │  │  Dashboard   │        │
│  └────────────┘  └──────────────┘  └──────────────┘        │
│         │                │                  │                │
│         └────────────────┴──────────────────┘                │
│                          │                                   │
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
│  ┌──────────────┐  ┌──────────────┐                        │
│  │   Feedback   │  │   Professor  │                         │
│  │   Endpoints  │  │   Endpoints  │                         │
│  └──────────────┘  └──────────────┘                         │
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

## Component Details

### Frontend (React + TypeScript)

#### Pages
- **Login**: Google OAuth via Firebase, role selection
- **Student**: Chat interface, PDF upload, history sidebar
- **Professor**: Analytics dashboard, Gemini insights, CSV export

#### Services
- **firebase.ts**: Authentication, token management
- **api.ts**: Axios client with token interceptor

#### Components
- **ChatBox**: Real-time chat with model, displays sources
- **FeedbackWidget**: Thumbs up/down with optional comments
- **PdfUploader**: Drag-and-drop PDF upload with progress
- **HistoryList**: Tabbed view of chat history and PDFs

### Backend (FastAPI)

#### API Layer (`app/api/`)

**auth.py**
- `POST /api/auth/verify`: Verify Firebase token, create/retrieve user
- Dependency: `get_current_user()` for protected routes

**chat.py**
- `POST /api/chat`: Main chat endpoint with RAG
  - Retrieves relevant chunks from ChromaDB
  - Builds context-aware prompt
  - Calls model for inference
  - Returns response + sources
- `GET /api/history`: Paginated chat history
- `DELETE /api/history/clear`: Clear all history

**pdf.py**
- `POST /api/pdf/upload_file`: Upload and process PDF
  - Extract text with PyMuPDF
  - Chunk with LangChain
  - Generate embeddings
  - Index in ChromaDB
- `GET /api/pdf/user_pdfs`: List user's PDFs
- `DELETE /api/pdf/pdf/{id}`: Delete PDF and chunks

**feedback.py**
- `POST /api/feedback/submit`: Submit feedback (rating + comment)
- `GET /api/feedback/chat/{id}`: Get feedback for chat

**professor.py**
- `GET /api/professor/stats`: Aggregate feedback statistics
- `GET /api/professor/low_rated`: Low-rated Q&A pairs
- `GET /api/professor/summary`: Gemini AI insights
- `GET /api/professor/export_csv`: Export feedback data

#### Service Layer (`app/services/`)

**model_loader.py**
- Singleton pattern for model instance
- 4-bit quantization with BitsAndBytes
- Lazy loading to save memory
- Thread-safe generation

**rag.py**
- PDF text extraction (PyMuPDF)
- Text chunking (LangChain RecursiveCharacterTextSplitter)
- Embedding generation (sentence-transformers)
- ChromaDB indexing and retrieval
- Metadata management

**inference.py**
- Prompt building with system instructions
- Context injection from RAG
- Conversation history management
- Safety checks (pre and post processing)
- Response cleaning

**gemini.py**
- Gemini API integration
- Prompt engineering for summaries
- Error handling and retries

#### Data Layer (`app/models.py`, `app/database.py`)

**Database Models (SQLAlchemy)**
- User: Authentication and role management
- Chat: Q&A pairs with sources
- Feedback: Satisfaction ratings
- PdfDocument: Uploaded file metadata
- PdfChunk: Text chunks with indexes

### Data Flow

#### Chat Flow
1. User sends message via ChatBox
2. Frontend calls `POST /api/chat` with Bearer token
3. Backend verifies token, identifies user
4. RAG service queries ChromaDB for relevant chunks
5. Inference service builds prompt with context + history
6. Model generates response
7. Response saved to PostgreSQL
8. Sources and response returned to frontend
9. FeedbackWidget allows rating

#### PDF Upload Flow
1. User drops PDF in PdfUploader
2. Frontend uploads to `POST /api/pdf/upload_file`
3. Backend extracts text with PyMuPDF
4. LangChain splits text into chunks (800 chars, 150 overlap)
5. sentence-transformers generates embeddings
6. ChromaDB indexes embeddings with metadata
7. PostgreSQL stores document and chunk records
8. Success message with chunk count returned

#### Professor Dashboard Flow
1. Professor accesses dashboard
2. Frontend fetches stats from `/api/professor/stats`
3. Displays feedback metrics in cards
4. Professor clicks "Generate Summary"
5. Backend aggregates low-rated chats
6. Sends batch to Gemini API with analysis prompt
7. Gemini returns structured insights
8. Frontend displays summary with markdown
9. Professor can export CSV for further analysis

## Technology Stack Details

### Backend Dependencies

**Core Framework**
- FastAPI: Async web framework
- Uvicorn: ASGI server

**Database**
- PostgreSQL: Relational data
- SQLAlchemy: ORM
- psycopg2: PostgreSQL driver

**Vector Database**
- ChromaDB: Embedding storage and similarity search

**ML/AI**
- torch: PyTorch framework
- transformers: Hugging Face transformers
- bitsandbytes: 4-bit quantization
- peft: Parameter-efficient fine-tuning
- sentence-transformers: Embedding generation

**Text Processing**
- LangChain: Text splitting utilities
- PyMuPDF: PDF text extraction

**Authentication**
- firebase-admin: Token verification

**API Integration**
- httpx: Async HTTP client for Gemini API

### Frontend Dependencies

**Core Framework**
- React 18: UI library
- TypeScript: Type safety
- Vite: Build tool and dev server

**Routing & State**
- React Router DOM: Client-side routing

**Styling**
- Tailwind CSS: Utility-first CSS
- PostCSS + Autoprefixer: CSS processing

**UI Components**
- Lucide React: Icon library

**Firebase**
- Firebase SDK: Authentication and Storage

**HTTP Client**
- Axios: Promise-based HTTP client

## Security Considerations

### Authentication
- Firebase handles OAuth securely
- JWT tokens verified on every request
- Tokens refresh automatically
- Role-based access control (RBAC)

### Data Protection
- SQL injection prevented by SQLAlchemy ORM
- CORS configured for specific origins
- Passwords never stored (OAuth only)
- API keys stored in environment variables

### Model Safety
- Input sanitization for harmful content
- Output filtering and cleaning
- Rate limiting on expensive operations
- System prompt enforces topic boundaries

## Performance Optimizations

### Model Loading
- Lazy initialization (loaded on first request)
- 4-bit quantization reduces memory by 4x
- Singleton pattern prevents multiple instances
- GPU utilization when available

### Database
- Indexed foreign keys and lookup columns
- Connection pooling in SQLAlchemy
- Lazy loading of relationships
- Pagination for large result sets

### Vector Search
- ChromaDB persistent storage
- Efficient similarity search with HNSW
- Metadata filtering before embedding search
- Batch embedding generation

### Frontend
- Code splitting with Vite
- Lazy component loading
- Optimistic UI updates
- Debounced API calls

## Scalability Considerations

### Horizontal Scaling
- Stateless backend (can run multiple instances)
- Load balancer distributes traffic
- Database connection pooling
- Shared ChromaDB volume or separate service

### Vertical Scaling
- Model requires GPU for production
- PostgreSQL can scale with read replicas
- ChromaDB benefits from more RAM
- Frontend can be served from CDN

### Caching
- Model loaded once per instance
- Browser caching for static assets
- Database query result caching (optional)
- Embeddings cached in ChromaDB

## Deployment Architecture (AWS)

```
Internet
   │
   ├─► CloudFront (Frontend CDN)
   │      │
   │      └─► S3 Bucket (React build)
   │
   └─► Application Load Balancer
          │
          ├─► EC2 Instance 1 (Backend + Model)
          ├─► EC2 Instance 2 (Backend + Model)
          └─► EC2 Instance N
                │
                ├─► RDS PostgreSQL (Multi-AZ)
                └─► ChromaDB (EBS Volume or Service)
```

### AWS Services
- **EC2**: Backend hosting (t2.xlarge or GPU instances)
- **RDS**: Managed PostgreSQL
- **S3**: Static frontend hosting
- **CloudFront**: CDN for frontend
- **ALB**: Load balancing
- **EBS**: Persistent volumes for ChromaDB
- **Secrets Manager**: Secure credential storage
- **CloudWatch**: Logging and monitoring

## Monitoring & Logging

### Backend Logs
- Request/response logging
- Error tracking with stack traces
- Model inference latency
- Database query performance

### Frontend Logs
- Error boundaries for React errors
- API call failures
- User interaction tracking (optional)

### Metrics
- Response times (p50, p95, p99)
- Request volume per endpoint
- Model inference time
- Database connection pool usage
- Feedback sentiment distribution

## Future Enhancements

### Planned Features
1. **Multi-document chat**: Reference multiple PDFs simultaneously
2. **Conversation threads**: Organize related chats
3. **Code execution**: Run DSA code snippets safely
4. **Video explanations**: Link to relevant video content
5. **Progress tracking**: Student learning analytics
6. **Collaborative features**: Share PDFs between students
7. **Mobile app**: Native iOS/Android apps
8. **Offline mode**: Cached responses for common questions

### Technical Improvements
1. **Caching layer**: Redis for frequently asked questions
2. **Streaming responses**: WebSocket for real-time generation
3. **Better embeddings**: Fine-tune embedding model on DSA content
4. **Model quantization**: Further optimize for edge devices
5. **A/B testing**: Compare different prompts and models
6. **Auto-scaling**: Based on load metrics
7. **Backup automation**: Scheduled database backups
8. **CI/CD pipeline**: Automated testing and deployment

