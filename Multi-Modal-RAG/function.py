import os
import time
import pickle
import io
import traceback
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request, UploadFile, File 
import numpy as np
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
query_cache = {}
# mongo_clients = mongo_client
# redis_clients = redis_client
db = None
collection = None


# Configuration
MAX_SHORT_MEMORY = 5       
MAX_HISTORY_CONTEXT = 4   
CHUNK_SIZE = 800
CHUNK_OVERLAP = 20
TOP_K_RETRIEVAL = 4

HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
Mongo_Url = os.getenv("MONGO_URL")
EMBED_MODEL_PATH = os.getenv("EMBEDMODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

all_docs = []
all_embeddings = []
image_data_store = {}





# =============================================================================
# MONGODB & REDIS SETUP
# =============================================================================
def init_storage():
    """Initialize MongoDB and Redis connections."""
    global mongo_client, db, collection, redis_client , mongo_client
    
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
    fetching conversation context for a session.
    this will returns: list of dicts [{"role": "user"/"bot", "content": "..."}]
    """
    # Get recent messages from Redis
    print("getting privious converstion. . . ")
    start_time = time.time()
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
    print(f"returning converstion in {start_time - time.time()}")
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

def fit_bmencoder(text):
    from pinecone_text.sparse import BM25Encoder
    sparser_encode = BM25Encoder().fit(text)
    pc_retriver.sparse_encoder = sparser_encode

# =============================================================================
# TEXT PROCESSING FUNCTIONS
# =============================================================================

def extract_and_store(filename: str, data):
    """Extract text and images from uploaded files."""
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    from PIL import Image
    import base64
    import io
    global new_namespace
    import pymupdf

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    print("extracting and storing from Docs. . .")
    start_time = time.time()


    if filename.lower().endswith(".pdf"):
        pdf_bytes = data.file.read()

        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                temp_doc = Document(
                    page_content=text,
                    metadata={"page": i, "type": "text"}
                )
                text_chunks = splitter.split_documents([temp_doc])

                for chunk in text_chunks:
                    embedding = clip_embedder.embed_query(chunk.page_content)
                    all_embeddings.append(embedding)
                    all_docs.append(chunk)

            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                    image_id = f"page_{i}_img_{img_index}"

                    buffered = io.BytesIO()
                    pil_image.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    image_data_store[image_id] = img_base64

                    embedding = clip_embedder.embed_image(pil_image)
                    all_embeddings.append(embedding)

                    image_doc = Document(
                        page_content=f"[Image: {image_id}]",
                        metadata={
                            "page": i,
                            "type": "image",
                            "image_id": image_id
                        }
                    )
                    all_docs.append(image_doc)

                except Exception as e:
                    print(f"Error processing image {img_index} on page {i}: {e}")

    elif filename.lower().endswith((".txt", ".md", ".csv", ".json", ".log", ".xml")):
        return data.file.read().decode("utf-8")

    else:
        raise ValueError(f"Unsupported file type: {filename}")
    
    import numpy as np
    embeddings_array = np.array(all_embeddings)
    vectors = []
    for i, (doc, emb) in enumerate(zip(all_docs, embeddings_array)):
        vectors.append(
            {
                "id": f"doc-{i}",
                "values": emb.tolist(),
                "metadata": {
                    "context": doc.page_content,
                    **doc.metadata,
                },
            }
        )
    
    name = os.path.splitext(filename)[0]
    new_namespace = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    

    pc_index.upsert(
        vectors=vectors,
        namespace=new_namespace)
    print(f"stored Docs in about {start_time - time.time()}")
    return all_docs

def create_multimodal_message(retrieved_docs):
    content = []

    # Add the query

    # Separate text and image documents
    start = time.time()
    print("creating multimodal input message")
    text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
    image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]

    # Add text context
    if text_docs:
        text_context = "\n\n".join(
            [f"[Page content: {doc.page_content}" for doc in text_docs]
        )
        content.append({"type": "text", "text": f"Text excerpts:\n{text_context}\n"})

    if image_docs is not None:
        for doc in image_docs:
            image_id = doc.metadata.get("image_id")
            if image_id and image_id in image_data_store:
                content.append(
                    {
                        "type": "text",
                        "text": f"\n[Image from page {doc.metadata['page']}]:\n",
                    }
                                )
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data_store[image_id]}"
                                    },
                    }
                                )
    print(f"returning message in {start - time.time()}")
    return content

def content_to_documents(contents, base_metadata=None):
    content = create_multimodal_message(contents)
    docs = []
    base_metadata = base_metadata or {}


    for i, item in enumerate(content):
        if item.get("type") == "text":
            docs.append(
                Document(
                    page_content=item.get("text", ""),
                    metadata={
                        **base_metadata,
                        "chunk_id": i,
                        "type": "text",
                    },
                )
            )

    return docs

def get_chunks_format_text(texts):
    """Split text into chunks. Handles both Document objects and plain text."""
    from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    start = time.time()
    text = content_to_documents(texts)
    if isinstance(text, list):
        # List of Document objects
        context = " ".join(doc.page_content for doc in text)
    else:
        # Plain text string
        context = text
    
    print(f"returned formated chunks in about {start - time.time()}")    
    return text_splitter.split_text(context)

def get_format_text(text):
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
from langchain_core.embeddings import Embeddings
from PIL import Image
class CLIPMultimodalEmbeddings(Embeddings):
    """
    CLIP embeddings model for text and images
    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.model.eval()

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x)

    def embed_query(self, text: str):
        return self._embed_text(text)

    def embed_documents(self, texts):
        return [self._embed_text(t) for t in texts]

    def embed_image(self, image):
        return self._embed_image(image)

    def _embed_text(self, text: str) -> list[float]:
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        features = self.model.get_text_features(**inputs)

        vec = features[0].detach().cpu().numpy()   # ✅ (512,)
        vec = self._normalize(vec)

        return vec.astype(float).tolist()

    def _embed_image(self, image) -> list[float]:
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")

        features = self.model.get_image_features(**inputs)

        vec = features[0].detach().cpu().numpy()   # ✅ (512,)
        vec = self._normalize(vec)

        return vec.astype(float).tolist()

from langchain_classic.retrievers import PineconeHybridSearchRetriever
from langchain_core.documents import Document
class CustomPineconeHybridRetriever(PineconeHybridSearchRetriever):
    def _normalize_text(self, text):
        # LangChain Documents
        if isinstance(text, list):
            if hasattr(text[0], "page_content"):
                return "\n".join(d.page_content for d in text)
            return "\n".join(map(str, text))

        # Single Document
        if hasattr(text, "page_content"):
            return text.page_content

        # Already string
        if isinstance(text, str):
            return text

        raise ValueError("Text must be str, Document, or list of Documents")

    def _get_relevant_documents(self, query, run_manager=None, **kwargs):
        if isinstance(query, dict):
            text = query.get("text")
            image = query.get("image")

            if image is not None and text is not None:

                text = self._normalize_text(text)
                image_vec = np.array(self.embeddings.embed_image(image))
                text_vec = np.array(self.embeddings.embed_query(text))

                dense_vec = 0.6 * image_vec + 0.4 * text_vec

                sparse_vec = self.sparse_encoder.encode_queries(text)

        elif isinstance(query, Image.Image):
            dense_vec = self.embeddings.embed_image(query)
            sparse_vec = None

        elif isinstance(query, str):
            dense_vec = self.embeddings.embed_query(query)
            sparse_vec = self.sparse_encoder.encode_queries(query)

        else:
            raise ValueError("Unsupported query type")

        if dense_vec is not None:
            dense_vec = dense_vec

        if len(sparse_vec["indices"]) == 0:
            sparse_vec = None

        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=self.top_k,
            include_metadata=True,
            namespace=new_namespace,
        )

        return [
            Document(
                page_content=m.metadata.get("context", ""),
                metadata={
                    "score": m.score,
                    "type": m.metadata.get("modality", "text"),
                    "source": m.metadata.get("source"),
                    "page": m.metadata.get("page"),
                },
            )
            for m in result.matches
        ]
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
    global llm, clip_model,clip_processor, pc_retriver, pc_index, final_chain, sparser_encode, startup_time , parellal_runnable , clip_embedder
    
    start = time.time()
    print("\n" + "="*70)
    print("🚀 RAG CHATBOT STARTUP")
    print("="*70)
    
    # Initialize storage
    print("\n⏱️ Initializing storage...")
    init_storage()
    
    # Load data
    print("\n⏱️ Loading data...")
    from pinecone_text.sparse import BM25Encoder
    from langchain_classic.document_loaders import TextLoader
    
    step_start = time.time()
    text_loader = TextLoader("texts.text")
    text_loaded = text_loader.load()
    chunks = get_format_text(text_loaded)
    print(f"✅ Loaded {len(chunks)} chunks in {time.time() - step_start:.2f}s")
    
    # Fit BM25 encoder
    print("⏱️ Fitting BM25 encoder...")
    sparser_encode = BM25Encoder().fit(chunks)
    
    # Import libraries
    print("\n⏱️ Importing libraries...")
    step_start = time.time()
    from pinecone import Pinecone
    from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    # from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEndpointEmbeddings
    # from transformers import AutoTokenizer , AutoModelForCausalLM
    # from sentence_transformers import SentenceTransformer
    print(f"✅ Libraries imported in {time.time() - step_start:.2f}s")
    
    # Setup HuggingFace
    print("\n⏱️ loading models ...")
    step_start = time.time()
    
    # from langchain.embeddings.base import Embeddings
    # from transformers import pipeline
    from transformers import CLIPModel , CLIPProcessor


    clip_model = CLIPModel.from_pretrained(EMBED_MODEL_PATH)
    clip_processor = CLIPProcessor.from_pretrained(EMBED_MODEL_PATH)
    clip_embedder = CLIPMultimodalEmbeddings(
    model=clip_model,
    processor=clip_processor,)

    # print(embed_llm.get_sentence_embedding_dimension() > 0)
    # embedder = SentenceTransformerEmbeddings(model=embed_llm)

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=GOOGLE_API_KEY)
    print(f"✅ text model ready in {time.time() - step_start:.2f}s")

    print("\n⏱️ Setting up Pinecone...")
    step_start = time.time()
    
    pinecone_storage = Pinecone(api_key=PINECONE_KEY)
    pc_index = pinecone_storage.Index("rag-img-text-db")
    
    pc_retriver = CustomPineconeHybridRetriever(
        embeddings=clip_embedder,
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
    
    
    app.state.startup_time = time.time() - start
    print("\n" + "="*70)
    print(f"✅ APPLICATION READY! Startup: {app.state.startup_time:.2f}s")
    print("="*70 + "\n")
    
    yield
    
    # Cleanup
    print("\n🛑 Shutting down...")