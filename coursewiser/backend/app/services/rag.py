"""
RAG service for PDF processing, embedding generation, and retrieval using ChromaDB
"""
import os
import json
from typing import List, Dict, Optional
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Set Hugging Face to use local cache and not require authentication for public models
os.environ['HF_HUB_OFFLINE'] = '0'  # Allow downloads but don't require auth
os.environ['TRANSFORMERS_OFFLINE'] = '0'


class RAGService:
    """
    Service for handling PDF processing, embedding generation, and retrieval
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not RAGService._initialized:
            # Initialize embedding model
            print("🔹 Loading sentence transformer model...")
            # Use a simple, reliable model name and explicitly set token=False (no HF token)
            try:
                self.embedding_model = SentenceTransformer(
                    'all-MiniLM-L6-v2',
                    token=False  # Explicitly no token for public models
                )
            except TypeError:
                # Fallback if token parameter not supported in this version
                print("⚠️  'token' parameter not supported, trying without...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize ChromaDB
            chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
            os.makedirs(chroma_dir, exist_ok=True)
            
            print(f"🔹 Initializing ChromaDB at: {chroma_dir}")
            self.chroma_client = chromadb.PersistentClient(
                path=chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collections
            self.collection = self.chroma_client.get_or_create_collection(
                name="pdf_chunks",
                metadata={"description": "PDF document chunks for RAG"}
            )
            
            self.class_material_collection = self.chroma_client.get_or_create_collection(
                name="class_material_chunks",
                metadata={"description": "Class material chunks for RAG"}
            )
            
            # Text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            RAGService._initialized = True
            print("✅ RAG service initialized")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using PyMuPDF
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        text = ""
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            raise
        
        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks using RecursiveCharacterTextSplitter
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        chunks = self.text_splitter.split_text(text)
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()

    def index_pdf_chunks(
        self,
        chunks: List[str],
        pdf_id: int,
        chunk_ids: List[int],
        user_id: int,
        filename: str
    ) -> None:
        """
        Index PDF chunks into ChromaDB
        
        Args:
            chunks: List of text chunks
            pdf_id: PDF document ID from database
            chunk_ids: List of chunk IDs from database
            user_id: User ID who uploaded the PDF
            filename: Original filename
        """
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Prepare metadata
        metadatas = []
        ids = []
        for i, chunk_id in enumerate(chunk_ids):
            metadatas.append({
                "pdf_id": str(pdf_id),
                "chunk_id": str(chunk_id),
                "chunk_index": str(i),
                "user_id": str(user_id),
                "filename": filename
            })
            ids.append(f"chunk_{chunk_id}")
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(chunks)} chunks from {filename}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[int] = None,
        pdf_ids: Optional[List[int]] = None,
        class_id: Optional[int] = None
    ) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            user_id: Filter by user ID (optional)
            pdf_ids: Filter by specific PDF IDs (optional)
            class_id: Filter by class ID for class materials (optional)
            
        Returns:
            List of relevant chunks with metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        all_chunks = []
        
        # Search personal PDFs
        if user_id is not None or pdf_ids is not None:
            # Build where filter for personal PDFs
            where_filter = None
            if user_id is not None and pdf_ids is not None:
                where_filter = {
                    "$and": [
                        {"user_id": str(user_id)},
                        {"pdf_id": {"$in": [str(pid) for pid in pdf_ids]}}
                    ]
                }
            elif user_id is not None:
                where_filter = {"user_id": str(user_id)}
            elif pdf_ids is not None:
                where_filter = {"pdf_id": {"$in": [str(pid) for pid in pdf_ids]}}
            
            # Query personal PDFs collection
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter if where_filter else None
                )
                
                if results and results['documents'] and len(results['documents']) > 0:
                    for i in range(len(results['documents'][0])):
                        metadata = results['metadatas'][0][i]
                        metadata['source_type'] = 'personal_pdf'
                        all_chunks.append({
                            'text': results['documents'][0][i],
                            'metadata': metadata,
                            'distance': results['distances'][0][i] if 'distances' in results else None
                        })
            except Exception as e:
                print(f"Error during personal PDF search: {e}")
        
        # Search class materials
        if class_id is not None:
            where_filter = {"class_id": str(class_id)}
            
            try:
                results = self.class_material_collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where_filter
                )
                
                if results and results['documents'] and len(results['documents']) > 0:
                    for i in range(len(results['documents'][0])):
                        metadata = results['metadatas'][0][i]
                        metadata['source_type'] = 'class_material'
                        all_chunks.append({
                            'text': results['documents'][0][i],
                            'metadata': metadata,
                            'distance': results['distances'][0][i] if 'distances' in results else None
                        })
            except Exception as e:
                print(f"Error during class material search: {e}")
        
        # Sort all chunks by distance and return top_k
        all_chunks.sort(key=lambda x: x['distance'] if x['distance'] is not None else float('inf'))
        return all_chunks[:top_k]

    def delete_pdf_chunks(self, chunk_ids: List[int]) -> None:
        """
        Delete chunks from ChromaDB
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        ids_to_delete = [f"chunk_{cid}" for cid in chunk_ids]
        try:
            self.collection.delete(ids=ids_to_delete)
            print(f"✅ Deleted {len(ids_to_delete)} chunks from ChromaDB")
        except Exception as e:
            print(f"Error deleting chunks: {e}")

    def index_class_material_chunks(
        self,
        chunks: List[str],
        material_id: int,
        chunk_ids: List[int],
        class_id: int,
        filename: str
    ) -> None:
        """
        Index class material chunks into ChromaDB
        
        Args:
            chunks: List of text chunks
            material_id: Class material ID from database
            chunk_ids: List of chunk IDs from database
            class_id: Class ID
            filename: Original filename
        """
        # Generate embeddings
        embeddings = self.generate_embeddings(chunks)
        
        # Prepare metadata
        metadatas = []
        ids = []
        for i, chunk_id in enumerate(chunk_ids):
            metadatas.append({
                "material_id": str(material_id),
                "chunk_id": str(chunk_id),
                "chunk_index": str(i),
                "class_id": str(class_id),
                "filename": filename
            })
            ids.append(f"class_chunk_{chunk_id}")
        
        # Add to ChromaDB
        self.class_material_collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Indexed {len(chunks)} class material chunks from {filename}")

    def delete_class_material_chunks(self, chunk_ids: List[int]) -> None:
        """
        Delete class material chunks from ChromaDB
        
        Args:
            chunk_ids: List of chunk IDs to delete
        """
        ids_to_delete = [f"class_chunk_{cid}" for cid in chunk_ids]
        try:
            self.class_material_collection.delete(ids=ids_to_delete)
            print(f"✅ Deleted {len(ids_to_delete)} class material chunks from ChromaDB")
        except Exception as e:
            print(f"Error deleting class material chunks: {e}")


# Global singleton instance
rag_service = None


def get_rag_service() -> RAGService:
    """
    Get the global RAG service instance
    """
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service

