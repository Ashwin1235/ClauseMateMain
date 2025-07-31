import os
import hashlib
import re
import traceback
import time
from io import BytesIO
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import asyncio
import aiohttp
import requests
import numpy as np

from dotenv import load_dotenv
from docx import Document
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import urllib.parse

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# -------------------------------
# Load Environment Variables & Config
# -------------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "2eae629d76beaa92a8afaacdb945eeecec9d7473f961178e9933101fd3c6a4d8")

# Model Configuration
MODEL_NAME = "BAAI/bge-m3"
PINECONE_DIMENSION = 1024
PINECONE_INDEX_NAME = "bajajhackathon-final"

# Railway-specific port configuration
PORT = int(os.getenv("PORT", 8000))

# -------------------------------
# Initialize Models and Services
# -------------------------------
@lru_cache(maxsize=1)
def get_embedding_model():
    """Cache the model to avoid reloading on each request"""
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Embedding model loaded successfully")
    return model

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
except Exception as e:
    print(f"❌ Pinecone initialization error: {e}")
    index = None

# -------------------------------
# FastAPI Setup
# -------------------------------
app = FastAPI(
    title="Optimized RAG System for Hackathon",
    description="RAG system deployed on Railway",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# -------------------------------
# Health Check Endpoint
# -------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "pinecone_connected": index is not None,
        "openrouter_key_set": bool(OPENROUTER_API_KEY)
    }

@app.get("/")
async def root():
    return {
        "message": "RAG System API is running!",
        "endpoints": {
            "health": "/health",
            "main_endpoint": "/api/v1/hackrx/run"
        }
    }

# -------------------------------
# Pydantic Models & Caching
# -------------------------------
class QueryPayload(BaseModel):
    documents: str
    questions: List[str]

processed_docs_cache = set()

@lru_cache(maxsize=4096)
def get_cached_embedding(text: str) -> tuple:
    embed_model = get_embedding_model()
    embedding = embed_model.encode(text, normalize_embeddings=True)
    return tuple(embedding.tolist())

def generate_doc_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

# -------------------------------
# Core Logic Functions
# -------------------------------
def load_and_extract_text(url: str) -> str:
    try:
        base_url = urllib.parse.urlsplit(url).path
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        file_stream = BytesIO(response.content)
        
        if base_url.lower().endswith('.pdf'):
            reader = PdfReader(file_stream)
            return "\n\n".join(f"[Page {i+1}]\n{page.extract_text()}" 
                             for i, page in enumerate(reader.pages) 
                             if page.extract_text())
        elif base_url.lower().endswith('.docx'):
            doc = Document(file_stream)
            return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        else:
            raise ValueError("Unsupported file type.")
    except Exception as e:
        raise Exception(f"Failed to extract text from {url}: {str(e)}")

def get_enhanced_text_chunks(text: str, doc_url: str) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    enhanced_chunks = []
    current_page = 1
    
    for i, chunk in enumerate(chunks):
        page_match = re.search(r'\[Page (\d+)\]', chunk)
        if page_match:
            current_page = int(page_match.group(1))
        
        clean_chunk = re.sub(r'\[Page \d+\]\n', '', chunk).strip()
        if len(clean_chunk) > 50:
            enhanced_chunks.append({
                'text': clean_chunk,
                'chunk_id': i,
                'page_number': current_page,
                'source_url': doc_url
            })
    
    return enhanced_chunks

async def batch_upsert_vectors(chunks: List[Dict], doc_url: str):
    if not index:
        raise Exception("Pinecone index not initialized")
    
    batch_size = 100
    doc_hash = generate_doc_hash(doc_url)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        vectors_to_upsert = []
        
        for chunk in batch:
            embedding = list(get_cached_embedding(chunk['text']))
            vector_id = f"{doc_hash}-{chunk['chunk_id']}"
            metadata = {k: v for k, v in chunk.items()}
            vectors_to_upsert.append((vector_id, embedding, metadata))
        
        try:
            index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            print(f"Batch upsert error: {e}")

async def get_rag_answer_async(prompt: str, max_retries: int = 3) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistralai/mistral-7b-instruct:free",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.1
    }
    
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        error_text = await response.text()
                        if attempt == max_retries - 1:
                            raise Exception(f"OpenRouter API error: {response.status} {error_text}")
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise Exception("Request timeout after multiple attempts")
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
        
        # Wait before retry
        await asyncio.sleep(2 ** attempt)

async def answer_question(question: str, doc_url: str) -> str:
    try:
        if not index:
            return "Pinecone index not available"
        
        q_embedding = list(get_cached_embedding(question))
        
        query_result = index.query(
            vector=q_embedding,
            top_k=15,
            include_metadata=True,
            include_scores=True,
            filter={"source_url": doc_url}
        )
        
        if not query_result['matches']:
            return "No relevant information found in the document."

        top_results = [m['metadata'] for m in query_result['matches']]
        
        context_parts = []
        max_context_chunks = 7
        for result in top_results[:max_context_chunks]:
            context_parts.append(f"Source (Page {result.get('page_number', 'N/A')}):\n{result['text']}")
        
        combined_context = "\n\n---\n\n".join(context_parts)
        
        prompt = f"""You are a precise policy document expert. Your task is to answer the question based ONLY on the provided context.

Context from the document:
{combined_context}

Question: {question}

CRITICAL INSTRUCTIONS:
1. Analyze the context carefully. Your answer MUST be based exclusively on the text provided.
2. Start with a DIRECT answer to the question.
3. Extract EXACT numbers, percentages, and timeframes (e.g., "30 days", "36 months", "5%").
4. If, and only if, the answer is not present in the context, you must state: "The answer to this question is not available in the provided document excerpts." Do not try to infer or guess.

Answer:"""
        
        return await get_rag_answer_async(prompt)
    except Exception as e:
        print(f"Error in answer_question: {traceback.format_exc()}")
        return f"Error processing question: {str(e)}"

# -------------------------------
# Main Hackathon API Endpoint
# -------------------------------
@app.post("/api/v1/hackrx/run")
async def run_rag_endpoint(payload: QueryPayload, authorization: Optional[str] = Header(None)):
    if authorization != f"Bearer {AUTH_TOKEN}":
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        doc_url = payload.documents
        questions = payload.questions
        doc_hash = generate_doc_hash(doc_url)

        if doc_hash not in processed_docs_cache:
            print(f"New document: {doc_url}. Starting ingestion...")
            text = load_and_extract_text(doc_url)
            chunks = get_enhanced_text_chunks(text, doc_url)
            await batch_upsert_vectors(chunks, doc_url)
            processed_docs_cache.add(doc_hash)
            print(f"✅ Document ingestion complete.")
        else:
            print(f"Using cached document: {doc_url}")

        tasks = [answer_question(q, doc_url) for q in questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        processed_answers = [str(a) if isinstance(a, Exception) else a for a in answers]
        
        return {"answers": processed_answers}
    except Exception as e:
        traceback_str = traceback.format_exc()
        print(f"Error in run_rag_endpoint: {traceback_str}")
        raise HTTPException(status_code=500, detail=str(e))

