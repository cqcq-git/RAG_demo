import os
import uuid
import re
import json
import time
from typing import Iterator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import ollama

# --- 1. SETUP & CONFIGURATION ---
load_dotenv()
LLM_MODE = os.getenv("LLM_MODE", "gemini").lower()

embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5", local_files_only=True)
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L12-v2', local_files_only=True)

google_client = genai.Client()

bm25_index = None
all_chunks = []

db_path = os.path.join(os.getcwd(), "chroma_storage")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(name="RAG_demo_collection")

# --- 2. THE INGESTION ENGINE ---
def initialize_database():
    global bm25_index, all_chunks
    
    if os.path.exists("doc.md"):
        with open("doc.md", 'r') as file:
            content = file.read()
        all_chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
        
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

initialize_database()

# --- 3. DATA MODELS ---
class QueryRequest(BaseModel):
    prompt: str
    top_k: int = 5

# --- 4. CORE RAG LOGIC ---
def build_rag_prompt(query: str, top_k: int) -> tuple[str, str, list[str]]:
    # --- 4a. Hybrid Retrieval ---
    # 1. Vector Search
    query_vec = embed_model.encode(query, normalize_embeddings=True).tolist()
    vector_results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    vector_chunks = vector_results['documents'][0]
    
    # 2. BM25 Search
    tokenized_query = re.sub(r'[^\w\s]', '', query.lower()).split()
    bm25_chunks = bm25_index.get_top_n(tokenized_query, all_chunks, n=top_k)
    
    # 3. Combine and Deduplicate
    combined_chunks = list(set(vector_chunks + bm25_chunks))
    
    if not combined_chunks:
        return "", "I don't have any context in my database to answer that.", []

    # --- 4b. Reranking ---
    pairs = [(query, chunk) for chunk in combined_chunks]
    scores = rerank_model.predict(pairs)
    scored_chunks = sorted(zip(combined_chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, s in scored_chunks[:3]]

    # --- 4c. The Prompts ---
    context_text = "\n\n".join([f"[Source {i+1}]: {c}" for i, c in enumerate(top_chunks)])
    system_prompt = "You are a helpful assistant. Answer strictly using the provided context. If the answer is not in the context, say you don't know."
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    return system_prompt, user_prompt, top_chunks


def stream_answer_chunks(system_prompt: str, user_prompt: str) -> Iterator[str]:
    if not system_prompt:
        yield user_prompt
        return

    if LLM_MODE == "ollama":
        print(f"🤖 Streaming from Local model (Ollama)")
        response = ollama.chat(
            model='qwen3:4b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            stream=True
        )
        for chunk in response:
            token = chunk.get('message', {}).get('content', '')
            if token:
                yield token
    else:
        print("☁️ Streaming from Google Gemini")
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = google_client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents=full_prompt
        )
        for chunk in response:
            token = getattr(chunk, "text", None)
            if token:
                yield token


def sse_event(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_rag_response(query: str, top_k: int) -> Iterator[str]:
    try:
        system_prompt, user_prompt, top_chunks = build_rag_prompt(query, top_k)
        yield sse_event("sources", top_chunks)

        output_text = ""
        started_at = time.perf_counter()
        for token in stream_answer_chunks(system_prompt, user_prompt):
            output_text += token
            yield sse_event("token", token)

        duration = time.perf_counter() - started_at
        output_tokens = len(output_text.split())
        yield sse_event("metrics", {
            "output_tokens_approx": output_tokens,
            "generation_seconds": round(duration, 3),
            "tokens_per_second_approx": round(output_tokens / duration, 2) if duration else None,
        })
        yield sse_event("done", True)
    except Exception as e:
        yield sse_event("error", f"RAG Error: {str(e)}")

app = FastAPI(title="RAG demo API")

# --- 6. API ENDPOINTS ---
@app.post("/ask")
async def ask_question(request: QueryRequest):
    return StreamingResponse(
        stream_rag_response(request.prompt, request.top_k),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
