# app.py - Main FastAPI Application for LFM2 Model
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from llama_cpp import Llama
import os
import time

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "./model"  # Local path where model will be saved
MODEL_FILE = "LFM2-1.2B-Q4_0.gguf"  # The model file name
DEFAULT_SYSTEM_PROMPT = "You're a helpful assistant. Be concise and accurate."

# ============================================
# DATA MODELS
# ============================================

class ChatMessage(BaseModel):
    """A single message in conversation"""
    role: str = Field(..., description="Either 'user' or 'assistant'")
    content: str = Field(..., description="The message text")

class ChatRequest(BaseModel):
    """Request format for chat endpoint"""
    message: str = Field(..., description="Your question or prompt")
    system_prompt: Optional[str] = Field(None, description="Custom AI instructions")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Previous messages")
    temperature: float = Field(0.0, description="0 = consistent, higher = creative")
    max_tokens: int = Field(512, description="Maximum response length")

class ChatResponse(BaseModel):
    """Response format from chat endpoint"""
    response: str = Field(..., description="AI's answer")
    prompt_tokens: int = Field(..., description="Input length")
    completion_tokens: int = Field(..., description="Output length")
    time_taken: float = Field(..., description="Response time in seconds")

class RAGRequest(BaseModel):
    """Request format for RAG (document-based) chat"""
    message: str = Field(..., description="Your question")
    documents: List[str] = Field(..., description="Context documents")
    system_prompt: Optional[str] = Field(None, description="Custom AI instructions")
    temperature: float = Field(0.0)
    max_tokens: int = Field(512)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_path: str
    uptime_seconds: float

# ============================================
# INITIALIZE FASTAPI APP
# ============================================

app = FastAPI(
    title="LFM2-1.2B-RAG Chat API",
    description="Production-ready API for LFM2 model with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
llm = None
startup_time = time.time()

# ============================================
# STARTUP: LOAD MODEL
# ============================================

@app.on_event("startup")
async def load_model():
    """Load the LFM2 model on startup"""
    global llm
    
    model_file_path = os.path.join(MODEL_PATH, MODEL_FILE)
    
    print(f"🔍 Looking for model at: {model_file_path}")
    
    if not os.path.exists(model_file_path):
        print(f"❌ Model file not found at {model_file_path}")
        print("⚠️  Server will start but model endpoints will fail")
        return
    
    try:
        print(f"🔄 Loading LFM2 model...")
        llm = Llama(
            model_path=model_file_path,
            n_ctx=2048,        # Context window
            n_threads=4,        # CPU threads (adjust based on server)
            n_batch=512,        # Batch size for prompt processing
            verbose=False
        )
        print("✅ Model loaded successfully!")
        print(f"📊 Context size: 2048 tokens")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        llm = None

# ============================================
# HEALTH CHECK ENDPOINTS
# ============================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "LFM2 Chat API is running",
        "model": "LFM2-1.2B-RAG",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "/chat",
            "rag_chat": "/rag-chat"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status"""
    uptime = time.time() - startup_time
    
    return HealthResponse(
        status="healthy" if llm else "model_not_loaded",
        model_loaded=llm is not None,
        model_path=MODEL_FILE,
        uptime_seconds=round(uptime, 2)
    )

# ============================================
# CHAT ENDPOINTS
# ============================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Basic chat endpoint
    
    Send a message and get AI response. Supports conversation history.
    
    Example:
    ```json
    {
        "message": "What is machine learning?",
        "system_prompt": "Be concise",
        "history": [],
        "temperature": 0.0,
        "max_tokens": 200
    }
    ```
    """
    
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs or contact admin."
        )
    
    start_time = time.time()
    
    try:
        # Build conversation messages
        messages = []
        
        # Add system prompt
        system_msg = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_msg})
        
        # Add conversation history
        for msg in request.history:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
        # Generate response
        output = llm.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Extract results
        response_text = output["choices"][0]["message"]["content"]
        prompt_tokens = output["usage"]["prompt_tokens"]
        completion_tokens = output["usage"]["completion_tokens"]
        
        time_taken = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            time_taken=round(time_taken, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )

@app.post("/rag-chat", response_model=ChatResponse, tags=["RAG"])
async def rag_chat(request: RAGRequest):
    """
    RAG (Retrieval Augmented Generation) chat endpoint
    
    Ask questions based on provided documents/context.
    
    Example:
    ```json
    {
        "message": "What technologies are mentioned?",
        "documents": [
            "Python is great for AI development.",
            "FastAPI is a modern web framework."
        ],
        "temperature": 0.0,
        "max_tokens": 300
    }
    ```
    """
    
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    start_time = time.time()
    
    try:
        # Format documents with special tags (as per LFM2 docs)
        formatted_docs = ""
        for i, doc in enumerate(request.documents, 1):
            formatted_docs += f"<document{i}>{doc}</document{i}>"
        
        # Build messages
        messages = []
        system_msg = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_msg})
        
        # Add documents + question in user message
        user_content = f"{formatted_docs}\n\nQuestion: {request.message}"
        messages.append({"role": "user", "content": user_content})
        
        # Generate response
        output = llm.create_chat_completion(
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        response_text = output["choices"][0]["message"]["content"]
        prompt_tokens = output["usage"]["prompt_tokens"]
        completion_tokens = output["usage"]["completion_tokens"]
        
        time_taken = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            time_taken=round(time_taken, 2)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating RAG response: {str(e)}"
        )

# ============================================
# RUN SERVER (for local testing)
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting LFM2 API Server...")
    print("📖 API Docs: http://localhost:8080/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
