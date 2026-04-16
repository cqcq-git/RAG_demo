import os
import uuid
import re  # [NEW] For BM25 tokenization
from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi  # [NEW] BM25 Library
import ollama

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
LLM_MODE = os.getenv("LLM_MODE", "gemini").lower()

embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", local_files_only=True)
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2', local_files_only=True)

google_client = genai.Client()

# [NEW] In-memory storage for BM25
bm25_index = None
all_chunks = []

db_path = os.path.join(os.getcwd(), "chroma_storage")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(name="RAG_demo_collection")

# --- 2. THE INGESTION ENGINE ---
def initialize_database():
    global bm25_index, all_chunks  # [NEW] Access globals
    
    if os.path.exists("doc.md"):
        with open("doc.md", 'r') as file:
            content = file.read()
        all_chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
        
        # [NEW] Initialize BM25 Index
        tokenized_corpus = [re.sub(r'[^\w\s]', '', c.lower()).split() for c in all_chunks]
        bm25_index = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 Index built with {len(all_chunks)} chunks.")

        if collection.count() == 0:
            print("🔍 ChromaDB empty. Starting vector ingestion...")
            embeddings = embed_model.encode(all_chunks, normalize_embeddings=True).tolist()
            ids = [str(uuid.uuid4()) for _ in range(len(all_chunks))]
            collection.add(documents=all_chunks, embeddings=embeddings, ids=ids)
            print(f"✅ ChromaDB ingestion complete.")
    else:
        print("⚠️ Error: doc.md not found.")

# --- 3. DATA MODELS ---
class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# --- 4. CORE RAG LOGIC ---
def get_answer(query: str, top_k: int):
    # --- 4a. Hybrid Retrieval ---
    # 1. Vector Search
    query_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    vector_results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    vector_chunks = vector_results['documents'][0]
    
    # 2. [NEW] BM25 Search
    tokenized_query = re.sub(r'[^\w\s]', '', query.lower()).split()
    bm25_chunks = bm25_index.get_top_n(tokenized_query, all_chunks, n=top_k)
    
    # 3. [NEW] Combine and Deduplicate
    combined_chunks = list(set(vector_chunks + bm25_chunks))
    
    if not combined_chunks:
        return "I don't have any context in my database to answer that.", []

    # --- 4b. Reranking ---
    pairs = [(query, chunk) for chunk in combined_chunks]
    scores = rerank_model.predict(pairs)
    scored_chunks = sorted(zip(combined_chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, s in scored_chunks[:3]]

    # --- 4c. The Prompts ---
    context_text = "\n\n".join([f"[Source {i+1}]: {c}" for i, c in enumerate(top_chunks)])
    system_prompt = "You are a helpful assistant. Answer strictly using the provided context. If the answer is not in the context, say you don't know."
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    # --- 4d. THE SWITCH ---
    if LLM_MODE == "ollama":
        print(f"🤖 Using Local model (Ollama) with context length: {len(context_text)}")
        try:
            response = ollama.chat(
                model='qwen3:4b',
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
            )
            answer = response['message']['content']
        except Exception as e:
            answer = f"Ollama Error: {str(e)}"
    else:
        print("☁️ Using Google Gemini")
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = google_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=full_prompt
        )
        answer = response.text

    return answer, top_chunks

# --- 5. LIFESPAN & APP SETUP ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Project Engine starting...")
    initialize_database() 
    yield
    print("🛑 Engine shutting down.")
    
app = FastAPI(title="RAG demo API", lifespan=lifespan)

# --- 6. API ENDPOINTS ---
@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        answer, sources = get_answer(request.prompt, request.top_k)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")