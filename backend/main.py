import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from datetime import datetime

# Add parent directory to path để import GraphRAG modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.graphs import Neo4jGraph
from graph_rag_query import GraphRAGQuery
from graph_rag_embeddings import EntityEmbeddings
from graph_rag_hybrid import HybridRetriever
from graph_rag_context import ContextBuilder
from graph_rag_gemini import GeminiRAG
from context_rag import ContextRAG

app = FastAPI(title="Graph RAG API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j configuration - get from environment variables or use defaults
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j+s://41ab799a.databases.neo4j.io")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "xmriUzmvo9dSAyc10u9mpB7nzyQHMZFooKqH5yBP2d4")

# Global instances (khởi tạo 1 lần khi start server)
graph = None
gemini_rag = None
context_rag = None  # Traditional RAG với ChromaDB
sessions = {}  # Lưu trữ sessions {session_id: {history: [], created_at: datetime}}
shared_embedding_model = None  # Shared embedding model cho cả 2 RAG systems

@app.on_event("startup")
async def startup_event():
    """Khởi tạo GraphRAG và Context RAG khi server start"""
    global graph, gemini_rag, context_rag, shared_embedding_model
    
    print("Initializing Graph RAG system...")
    
    # Connect to Neo4j
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    # Initialize components (EntityEmbeddings tự động load model)
    graph_query = GraphRAGQuery(graph)
    embeddings = EntityEmbeddings(graph)
    hybrid = HybridRetriever(graph_query, embeddings)
    builder = ContextBuilder()
    
    # Save shared embedding model từ Graph RAG
    shared_embedding_model = embeddings.model 
    
    # Initialize Gemini RAG
    gemini_rag = GeminiRAG(
        hybrid_retriever=hybrid,
        context_builder=builder,
        model_name="gemini-2.5-flash"
    )
    
    print("Graph RAG system initialized successfully!")
    
    # Initialize Context RAG (Traditional RAG) - KHÔNG load embedding model mới
    print("Initializing Context RAG system...")
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("Warning: GOOGLE_API_KEY not set, Context RAG will not work")
    
    # Pass shared embedding model thay vì để ContextRAG tự load
    context_rag = ContextRAG(
        chroma_dir="../chroma_db",
        collection_name="vn_traffic_law",
        gemini_api_key=gemini_key,
        gemini_model="gemini-2.5-flash",
        use_rerank=False,
        embedding_model=shared_embedding_model  # Dùng chung model đã load
    )
    
    print("Context RAG system initialized successfully!")

@app.get("/")
def read_root():
    return {
        "message": "Graph RAG API",
        "status": "running",
        "endpoints": {
            "create_session": "/api/create_new_session",
            "graphrag": "/api/graphrag",
            "contextrag": "/api/rag"
        }
    }

# Models
class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message: str

class GraphRAGMessage(BaseModel):
    session_id: Optional[str] = None
    question: str

class RAGMessage(BaseModel):
    session_id: Optional[str] = None
    question: str
    
class GraphRAGResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    metadata: dict


class RAGResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    metadata: dict

@app.post("/api/create_new_session", response_model=SessionResponse)
def create_new_session():
    """Tạo mới một session hội thoại"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "history": [],
        "created_at": datetime.now()
    }
    
    return SessionResponse(
        session_id=session_id,
        created_at=sessions[session_id]["created_at"].isoformat(),
        message="New session created successfully"
    )

@app.post("/api/graphrag", response_model=GraphRAGResponse)
def ask_question(message: GraphRAGMessage):
    """
    Gọi GraphRAG Gemini để trả lời câu hỏi
    
    Args:
        message: Chứa session_id (optional) và question
        
    Returns:
        Câu trả lời từ GraphRAG
    """
    if gemini_rag is None:
        raise HTTPException(status_code=503, detail="GraphRAG system not initialized")
    
    # Tạo session mới nếu chưa có
    session_id = message.session_id
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "history": [],
            "created_at": datetime.now()
        }
    
    try:
        # Generate answer using GraphRAG
        result = gemini_rag.generate_answer(
            question=message.question,
            prompt_type="qa",
            max_tokens=8192, include_law_content=True, tranditional_rag_context=context_rag
        )
        
        # Save to session history
        sessions[session_id]["history"].append({
            "question": message.question,
            "answer": result["answer"],
            "timestamp": datetime.now().isoformat()
        })
        
        return GraphRAGResponse(
            session_id=session_id,
            question=message.question,
            answer=result["answer"],
            metadata={
                "retrieval_time": result["metadata"]["retrieval_time"],
                "generation_time": result["metadata"]["generation_time"],
                "total_time": result["metadata"]["total_time"],
                "entities_used": result["metadata"]["entities_used"],
                "relationships_used": result["metadata"]["relationships_used"]
            }
        )
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.get("/api/session/{session_id}")
def get_session_history(session_id: str):
    """Lấy lịch sử hội thoại của session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "created_at": sessions[session_id]["created_at"].isoformat(),
        "history": sessions[session_id]["history"]
    }
@app.post("/api/rag", response_model=RAGResponse)
def ask_question_rag(message: RAGMessage):
    """
    Gọi Context RAG (Traditional RAG) để trả lời câu hỏi
    
    Args:
        message: Chứa session_id (optional) và question
        
    Returns:
        Câu trả lời từ Context RAG
    """
    if context_rag is None:
        raise HTTPException(status_code=503, detail="Context RAG system not initialized")
    
    # Tạo session mới nếu chưa có
    session_id = message.session_id
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {
            "history": [],
            "created_at": datetime.now()
        }
    
    try:
        import time
        start_time = time.time()
        
        # Generate answer using Context RAG
        result = context_rag.rag_qa(
            question=message.question,
            top_k_dense=30,
            top_n_final=10,
            enable_rewrite=False
        )
        
        total_time = time.time() - start_time
        
        # Save to session history
        sessions[session_id]["history"].append({
            "question": message.question,
            "answer": result["answer"],
            "timestamp": datetime.now().isoformat()
        })
        
        return RAGResponse(
            session_id=session_id,
            question=message.question,
            answer=result["answer"],
            metadata={
                "total_time": total_time,
                "entities_used": len(result["hits"]),
                "relationships_used": 0,  # Traditional RAG không có relationships
                "normalized_query": result["normalized_query"]
            }
        )
        
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    """Xóa session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "graphrag_initialized": gemini_rag is not None,
        "contextrag_initialized": context_rag is not None,
        "active_sessions": len(sessions)
    }