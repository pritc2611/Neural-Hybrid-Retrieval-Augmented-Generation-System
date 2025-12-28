import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import pickle
import io
import traceback
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
llm = None
chat_llm = None
embed_llm = None
pc_retriver = None
pc_index = None
text_splitter = None
sparser_encode = None  # Global BM25 encoder
startup_time = 0
query_cache = {}

# MongoDB & Redis
mongo_client = None
db = None
collection = None
redis_client = None

# Configuration
MAX_SHORT_MEMORY = 5        # Last N messages in Redis (fast access)
MAX_HISTORY_CONTEXT = 4    # Messages to include in LLM context
CHUNK_SIZE = 800
CHUNK_OVERLAP = 20
TOP_K_RETRIEVAL = 4

HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
Mongo_Url = os.getenv("MONGO_URL")
EMBED_MODEL_PATH = os.getenv("EMBEDMODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# =============================================================================
# MONGODB & REDIS SETUP
# =============================================================================
def init_storage():
    """Initialize MongoDB and Redis connections."""
    global mongo_client, db, collection, redis_client
    
    try:
        from pymongo import MongoClient
        import redis
        
        # MongoDB
        mongo_client = MongoClient(Mongo_Url)
        db = mongo_client["chatdb"]
        collection = db["messages"]
        print("✅ MongoDB connected")
        
        # Redis
        redis_client = redis.Redis(host="redis_db", port=6379, db=0, decode_responses=True)
        redis_client.ping()  # Test connection
        print("✅ Redis connected")
        
    except Exception as e:
        print(f"⚠️ Storage connection failed: {e}")
        print("   Falling back to in-memory storage")
        # Fallback to in-memory
        collection = None
        redis_client = None

def extract_answer(user_query, model_output):
    uq = user_query.strip().lower()
    out = model_output.strip()
    if uq in out.lower():
        idx = out.lower().find(uq) + len(uq) + 1
        out = out[idx:].strip()

    return out

# =============================================================================
# STORAGE HELPER FUNCTIONS
# =============================================================================
def save_message_mongo(session_id: str, role: str, content: str):
    """Save message to MongoDB (long-term storage)."""
    if collection is not None:
        try:
            collection.insert_one({
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
        except Exception as e:
            print(f"⚠️ MongoDB save failed: {e}")

def save_message_redis(session_id: str, role: str, content: str):
    """Save message to Redis (short-term memory)."""
    if redis_client is not None:
        try:
            key = f"chat:{session_id}"
            msg = json.dumps({"role": role, "content": content})
            redis_client.rpush(key, msg)
            # Keep only last MAX_SHORT_MEMORY messages
            redis_client.ltrim(key, -MAX_SHORT_MEMORY, -1)
        except Exception as e:
            print(f"⚠️ Redis save failed: {e}")

def get_short_memory(session_id: str) -> list:
    """Get recent messages from Redis."""
    if redis_client is None:
        return []
    
    try:
        key = f"chat:{session_id}"
        raw_msgs = redis_client.lrange(key, 0, -1)
        return [json.loads(m) for m in raw_msgs]
    except Exception as e:
        print(f"⚠️ Redis read failed: {e}")
        return []

def get_conversation_context(session_id: str, include_mongo: bool = False, mongo_limit: int = 50) -> list:
    """
    Fetch conversation context for a session.
    Returns: list of dicts [{"role": "user"/"bot", "content": "..."}]
    """
    # Get recent messages from Redis
    context = get_short_memory(session_id)
    
    if include_mongo and collection is not None:
        try:
            # Get older messages from MongoDB
            mongo_msgs = list(collection.find(
                {"session_id": session_id},
                {"_id": 0, "role": 1, "content": 1, "timestamp": 1}
            ).sort("timestamp", -1).limit(mongo_limit))
            
            # Reverse to get chronological order
            mongo_msgs.reverse()
            
            # Only add messages not in Redis
            redis_content = {(m['role'], m['content']) for m in context}
            for msg in mongo_msgs:
                if (msg['role'], msg['content']) not in redis_content:
                    context.insert(0, {"role": msg['role'], "content": msg['content']})
        except Exception as e:
            print(f"⚠️ MongoDB read failed: {e}")
    
    # Return only last MAX_HISTORY_CONTEXT messages for LLM
    return context[-MAX_HISTORY_CONTEXT:]

def format_context_for_model(messages: list) -> str:
    """Convert message list to text format for LLM."""
    if not messages:
        return ""
    
    context_text = "Previous conversation:\n"
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = re.sub('<[^<]+?>', '', msg['content'])
        context_text += f"{role}: {content}\n"
    
    return context_text + "\n"

# =============================================================================
# TEXT PROCESSING FUNCTIONS
# =============================================================================
def extract_text_from_file(filename: str, data: bytes) -> str:
    """Extract text from uploaded files (PDF or text-based files)."""

    if filename.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # avoid None
                text += page_text + "\n"

        return text

    # handle ANY text-based file, not just .txt
    elif filename.lower().endswith((".txt", ".md", ".csv", ".json", ".log", ".xml")):
        return data.decode("utf-8")

    else:
        raise ValueError(f"Unsupported file type: {filename}")

def get_chunks_format_text(text):
    """Split text into chunks. Handles both Document objects and plain text."""
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    if isinstance(text, list):
        # List of Document objects
        context = " ".join(doc.page_content for doc in text)
    else:
        # Plain text string
        context = text
    
    return text_splitter.split_text(context)

# =============================================================================
# MODEL INFERENCE
# =============================================================================
async def async_model(question: str, session_id: str = "default") -> str:
    print(f"🔍 Processing: {question[:50]}...")

    conversation_history = get_conversation_context(session_id, include_mongo=True)
    context_text = format_context_for_model(conversation_history)

    cache_key = question.strip()
    if cache_key in query_cache:
        print("✅ Cache hit")
        return query_cache[cache_key]

    try:
        start = time.time()
        response = await run_in_threadpool(
            final_chain.invoke,
            {"question": question, "chat_history": context_text}
        )
        print(f"✅ Response from model in {time.time() - start:.2f}s")

        answer = extract_answer(question, response)

        # GUARANTEE string output
        if not isinstance(answer, str):
            answer = str(answer)

        query_cache[cache_key] = answer
        return answer

    except Exception as e:
        print("❌ /chat error:")
        traceback.print_exc()

        # ✅ ALWAYS return a STRING
        return f"An error occurred while generating the response."



# =============================================================================
# CUSTOM PINECONE RETRIEVER
# =============================================================================

from langchain_classic.retrievers import PineconeHybridSearchRetriever
from langchain_core.documents import Document
class CPineconeHybridRetriever(PineconeHybridSearchRetriever):
    def _get_relevant_documents(self, query, run_manager=None, **kwargs):
        if isinstance(query, dict):
            # When coming from RunnableParallel, query contains the full context
            actual_query = query.get("question", str(query))
        else:
            actual_query = str(query)
        
        print(f"🔍 Query type: {type(query)}, Extracted: {actual_query[:100]}")

        dense_vec = self.embeddings.embed_query(actual_query)
        

        # 🔥 FIX: If sparse vector is empty, remove it
        try:
            sparse_vec = self.sparse_encoder.encode_queries(actual_query)
            
            # FIX 2: Handle empty sparse vectors properly
            if sparse_vec and len(sparse_vec.get("indices", [])) == 0:
                print("⚠️ Empty sparse vector, using dense only")
                sparse_vec = None
        except Exception as e:
            print(f"⚠️ Sparse encoding failed: {e}, falling back to dense only")
            sparse_vec = None


        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
            namespace= new_namespace or "jack-story",)

        docs = []
        for m in result.matches:
            docs.append(Document(page_content=m.metadata.get("context",""),metadata={"score":m.score}))
        return docs

def extract_question(input_dict):
    """Extract just the question from the input"""
    if isinstance(input_dict, dict):
      return input_dict.get("question", str(input_dict))
    return str(input_dict)


# =============================================================================
# LIFESPAN - STARTUP
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all resources on startup."""
    global llm, chat_llm, embed_llm, pc_retriver, pc_index, final_chain, sparser_encode, startup_time , parellal_runnable
    
    start = time.time()
    print("\n" + "="*70)
    print("🚀 RAG CHATBOT STARTUP")
    print("="*70)
    
    # Initialize storage
    print("\n⏱️ Initializing storage...")
    init_storage()
    
    # Initialize text splitter
    # from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP
    # )
    
    # Load data
    print("\n⏱️ Loading data...")
    from pinecone_text.sparse import BM25Encoder
    from langchain_classic.document_loaders import TextLoader
    
    step_start = time.time()
    text_loader = TextLoader("texts.text")
    text_loaded = text_loader.load()
    chunks = get_chunks_format_text(text_loaded)
    print(f"✅ Loaded {len(chunks)} chunks in {time.time() - step_start:.2f}s")
    
    # Fit BM25 encoder
    print("⏱️ Fitting BM25 encoder...")
    step_start = time.time()
    sparser_encode = BM25Encoder().fit(chunks)
    print(f"✅ BM25 fitted in {time.time() - step_start:.2f}s")
    
    # Import libraries
    print("\n⏱️ Importing libraries...")
    step_start = time.time()
    from pinecone import Pinecone
    from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
    from transformers import AutoTokenizer , AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    print(f"✅ Libraries imported in {time.time() - step_start:.2f}s")
    
    # Setup HuggingFace
    print("\n⏱️ loading models ...")
    step_start = time.time()
    
    from langchain.embeddings.base import Embeddings
    from transformers import pipeline
    from langchain_huggingface import HuggingFacePipeline
    class SentenceTransformerEmbeddings(Embeddings):
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            return self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
                ).tolist()

        def embed_query(self, text):
            return self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=True
                )[0].tolist()

    
    embed_llm = SentenceTransformer(EMBED_MODEL_PATH)
    print(embed_llm.get_sentence_embedding_dimension() > 0)
    embedder = SentenceTransformerEmbeddings(model=embed_llm)
    print("✅ embedding model loaded")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=GOOGLE_API_KEY)
    print("✅ text model loaded")
    print(f"✅ text model ready in {time.time() - step_start:.2f}s")

    print("\n⏱️ Setting up Pinecone...")
    step_start = time.time()
    
    pinecone_storage = Pinecone(api_key=PINECONE_KEY)
    pc_index = pinecone_storage.Index("rag-endtoend")
    
    pc_retriver = CPineconeHybridRetriever(
        embeddings=embedder,
        index=pc_index,
        top_k=TOP_K_RETRIEVAL,
        sparse_encoder=sparser_encode
    )
    print(f"✅ Pinecone ready in {time.time() - step_start:.2f}s")
    
    # Build RAG chain
    print("\n⏱️ Building RAG chain...")
    step_start = time.time()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent RAG assistant with access to a knowledge base.
         CORE PRINCIPLES:
         1. Use retrieved context when relevant to answer the question
         2. If context is insufficient or irrelevant, use your general knowledge
         3. NEVER say "the context doesn't mention" - just answer naturally
         4. Maintain conversation continuity by considering previous messages
         5. Be concise, accurate, and helpful
         
         RESPONSE STYLE:
         - Answer directly without repeating the question
         - Use clear structure (paragraphs, lists when appropriate)
         - For long answers (>200 words), add a brief summary
         - Cite specific information from context when using it"""),
         
          ("""Previous Conversation:
             {chat_history}

             Retrieved Context:
                {rag_context}

             Question:
               {question}""")])

    
    parellal_runnable = RunnableParallel({
    "rag_context": RunnableLambda(extract_question) | pc_retriver | RunnableLambda(get_chunks_format_text),
    "chat_history": RunnablePassthrough(),
    "question": RunnablePassthrough()})
    parser = StrOutputParser()
    final_chain = parellal_runnable | prompt | llm | parser
    
    print(f"✅ RAG chain ready in {time.time() - step_start:.2f}s")
    
    startup_time = time.time() - start
    print("\n" + "="*70)
    print(f"✅ APPLICATION READY! Startup: {startup_time:.2f}s")
    print("="*70 + "\n")
    
    yield
    
    # Cleanup
    print("\n🛑 Shutting down...")
    # query_cache.clear()
    # if mongo_client:
    #     mongo_client.close()
    # if redis_client:
    #     redis_client.close()

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================
app = FastAPI(
    title="RAG Chatbot with Memory",
    description="RAG chatbot with MongoDB + Redis conversation memory",
    version="2.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =============================================================================
# MAIN ROUTES
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = "default"

    # Load history (from Redis + MongoDB)
    raw_history = get_conversation_context(session_id, include_mongo=True)

    # Convert to frontend-friendly format
    chat_history = []
    for msg in raw_history:
        chat_history.append({
            "type": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"]
        })

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": chat_history,
            "latency": None,
            "startup_time": f"{startup_time:.2f}s"
        }
    )

@app.post("/chat")
async def chat(request: Request):
    """Handle chat with MongoDB + Redis storage."""
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        
        if not question:
            return JSONResponse(
                {"error": "Please enter a question"},
                status_code=400
            )
        
        request_start = time.time()
        
        # Generate response with conversation context
        answer = await async_model(question, session_id)
        
        # Convert to HTML
        answer_html = markdown.markdown(
            answer,
            extensions=[TableExtension(), FencedCodeExtension()]
        )
        
        # Calculate latency
        latency_ms = int((time.time() - request_start) * 1000)
        
        # Save to storage
        save_message_redis(session_id, "user", question)
        save_message_redis(session_id, "bot", answer)
        save_message_mongo(session_id, "user", question)
        save_message_mongo(session_id, "bot", answer_html)
        
        return JSONResponse({
            "answer": answer,
            "answer_html": answer_html,
            "latency": latency_ms,
            "session_id": session_id
        })
        
    except Exception as e:
        print("❌ /chat error:")
        traceback.print_exc()
        return JSONResponse(
            {"error": f"An error occurred: {str(e)}"},
            status_code=500
        )

# =============================================================================
# FILE UPLOAD
# =============================================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process documents."""
    global sparser_encode , new_namespace
    
    try:
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.text')):
            return JSONResponse(
                {"error": "Only PDF and TXT files are supported"},
                status_code=400
            )
        
        print(f"\n📄 Processing: {file.filename}")
        start = time.time()
        
        # Extract text
        contents = await file.read()
        text = extract_text_from_file(file.filename, contents)
        print(f"✅ Extracted {len(text)} characters")
        
        # Chunk text
        chunks = get_chunks_format_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        
        name = os.path.splitext(file.filename)[0]
        new_namespace = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        pc_retriver.add_texts(
               texts=chunks,
               namespace=new_namespace)

        # Re-fit BM25 with new chunks
        from pinecone_text.sparse import BM25Encoder
        all_chunks = chunks 
        sparser_encode = BM25Encoder().fit(all_chunks)
        pc_retriver.sparse_encoder = sparser_encode
        print(f"✅ BM25 encoder updated")
        
        elapsed = time.time() - start
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "processing_time": f"{elapsed:.2f}s"
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# CONVERSATION HISTORY ENDPOINTS
# =============================================================================
@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 50):
    """Get conversation history for a session."""
    try:
        history = get_conversation_context(session_id, include_mongo=True, mongo_limit=limit)
        return JSONResponse({
            "session_id": session_id,
            "messages": history,
            "count": len(history)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/history/{session_id}")
async def clear_session_history(session_id: str):
    """Clear history for a specific session."""
    try:
        # Clear Redis
        if redis_client:
            redis_client.delete(f"chat:{session_id}")
        
        # Clear MongoDB
        if collection:
            result = collection.delete_many({"session_id": session_id})
            deleted = result.deleted_count
        else:
            deleted = 0
        
        return JSONResponse({
            "message": f"Cleared history for session {session_id}",
            "deleted_messages": deleted
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================
@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "startup_time": f"{startup_time:.2f}s",
        "mongodb": collection is not None,
        "redis": redis_client is not None,
        "query_cache_size": len(query_cache)
    }

@app.get("/stats")
async def stats():
    """Get statistics."""
    mongo_count = 0
    if collection:
        try:
            mongo_count = collection.count_documents({})
        except:
            pass
    
    return {
        "startup_time": f"{startup_time:.2f}s",
        "mongodb_messages": mongo_count,
        "redis_connected": redis_client is not None,
        "query_cache": {
            "size": len(query_cache),
            "cached_queries": list(query_cache.keys())[:5]
        }
    }

@app.post("/clear-cache")
async def clear_cache():
    """Clear query cache."""
    query_cache.clear()
    return {"message": "Cache cleared"}
    
