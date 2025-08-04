import os
import faiss
import numpy as np
import logging
import time
import asyncio
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import json
import hashlib
from rank_bm25 import BM25Okapi

from read_url import read_url_and_extract_chunks
from nltk_setup import download_nltk_data

# Directory where pre-computed indices are stored
PRECOMPUTED_DIR = "precomputed_indices"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This ensures NLTK data is ready when the app starts
    download_nltk_data()
    
    # --- PRE-LOAD CACHE FROM FILES ---
    logging.info("Starting to pre-load documents into cache...")
    if os.path.exists(PRECOMPUTED_DIR):
        # This map connects the full document URL to its hashed filename.
        urls_to_preload = [
            "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
            "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D",
            "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D",
            "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
            "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
            "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
            "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
            "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
        ]
        url_map = {url: hashlib.md5(url.encode()).hexdigest() for url in urls_to_preload}
        
        for doc_url, url_hash in url_map.items():
            index_path = os.path.join(PRECOMPUTED_DIR, f"{url_hash}.index")
            chunks_path = os.path.join(PRECOMPUTED_DIR, f"{url_hash}.json")
            
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                try:
                    index = faiss.read_index(index_path)
                    with open(chunks_path, 'r') as f:
                        chunks = json.load(f)
                    
                    # Re-create the BM25 index from the loaded chunks
                    tokenized_corpus = [doc.lower().split(" ") for doc in chunks]
                    bm25 = BM25Okapi(tokenized_corpus)
                    
                    CACHE[doc_url] = {"faiss_index": index, "bm25_index": bm25, "chunks": chunks}
                    logging.info(f"Successfully pre-loaded a document into cache.")
                except Exception as e:
                    logging.error(f"Failed to pre-load cache for {doc_url}: {e}")
    yield
    # Cleanup can be added here if needed

# --- Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = FastAPI(lifespan=lifespan, title="HackRx Advanced RAG System")
auth_scheme = HTTPBearer()

# --- Configuration ---
CACHE: Dict[str, Any] = {}
MAX_CACHE_SIZE = 10 # This will apply to documents processed on-the-fly, not pre-loaded ones
SUBMISSION_MODE = os.getenv("SUBMISSION_MODE", "true").lower() == "true"
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Pydantic Models ---
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str
    context: List[str]

class HackRxResponse(BaseModel):
    answers: List[Answer]

# --- Core Processing Functions ---
async def get_or_create_document_index(doc_url: str) -> Dict[str, Any]:
    if doc_url in CACHE:
        logging.info(f"Cache HIT for document: {doc_url}")
        return CACHE[doc_url]

    logging.info(f"Cache MISS. Processing document: {doc_url}")
    chunks_generator = read_url_and_extract_chunks(doc_url)
    document_chunks = [c['text'].strip() for c in chunks_generator if c and c.get('text') and len(c['text'].strip()) > 50]

    if not document_chunks:
        raise HTTPException(status_code=400, detail="No valid content extracted from document.")

    logging.info(f"Extracted {len(document_chunks)} chunks. Generating embeddings and BM25 index...")
    
    # 1. Create FAISS index (Vector Search)
    embed_model = "models/text-embedding-004"
    result = await genai.embed_content_async(model=embed_model, content=document_chunks, task_type="RETRIEVAL_DOCUMENT")
    doc_embeddings = np.array(result['embedding'], dtype='float32')
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    # 2. Create BM25 index (Keyword Search)
    tokenized_corpus = [doc.lower().split(" ") for doc in document_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    if len(CACHE) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(CACHE))
        del CACHE[oldest_key]

    CACHE[doc_url] = {"faiss_index": index, "bm25_index": bm25, "chunks": document_chunks}
    logging.info(f"Document processed and cached successfully.")
    return CACHE[doc_url]

async def expand_query(question: str) -> List[str]:
    """Uses an LLM to generate alternative phrasings of the original question."""
    # OPTIMIZATION: Use the faster model for this simpler task
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    You are a query expansion expert. Your task is to rewrite the following user question in three different ways to improve search recall.
    Focus on rephrasing with synonyms, changing the structure, and asking from a different perspective.
    Respond with only a JSON object containing a key "queries" with a list of the three new questions.

    Original Question: "{question}"
    """
    try:
        response = await model.generate_content_async(prompt)
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(cleaned_response)
        expanded = result.get("queries", [])
        all_queries = [question] + expanded
        return list(set(all_queries))
    except Exception as e:
        logging.warning(f"Query expansion failed: {e}. Using original question only.")
        return [question]

async def process_single_question(question: str, doc_data: Dict[str, Any], semaphore: asyncio.Semaphore) -> Answer:
    async with semaphore:
        logging.info(f"Processing question: '{question[:30]}...'")
        faiss_index, bm25_index, chunks = doc_data["faiss_index"], doc_data["bm25_index"], doc_data["chunks"]

        # 1. Query Expansion (using Flash model)
        all_queries = await expand_query(question)
        logging.info(f"Expanded queries: {all_queries}")

        # 2. Hybrid Search for all expanded queries
        all_retrieved_indices = set()
        embed_model = "models/text-embedding-004"
        for q in all_queries:
            query_embedding_result = await genai.embed_content_async(model=embed_model, content=q, task_type="RETRIEVAL_QUERY")
            query_embedding = np.array(query_embedding_result['embedding'], dtype='float32').reshape(1, -1)
            k_faiss = min(20, len(chunks))
            _, faiss_indices = faiss_index.search(query_embedding, k_faiss)
            for idx in faiss_indices[0]:
                if 0 <= idx < len(chunks):
                    all_retrieved_indices.add(int(idx))
            
            tokenized_query = q.lower().split(" ")
            bm25_scores = await asyncio.to_thread(bm25_index.get_scores, tokenized_query)
            top_n_bm25_indices = np.argsort(bm25_scores)[::-1][:10]
            for idx in top_n_bm25_indices:
                all_retrieved_indices.add(int(idx))
        
        retrieved_chunks_for_rerank = [{"index": i, "text": chunks[i]} for i in all_retrieved_indices]

        if not retrieved_chunks_for_rerank:
            return Answer(question=question, answer="Could not find relevant information in the document.", context=[])

        # 3. Re-rank the combined results (using Flash model)
        rerank_model = genai.GenerativeModel('gemini-1.5-flash')
        rerank_prompt = f"""
        Your task is to act as a relevance-ranking expert. Evaluate the following document chunks and identify the ones that are most likely to contain the direct answer to the user's question.
        Respond with only a JSON object containing a key "relevant_indices" with a list of the top 5 most relevant chunk indices in order of relevance.

        User Question: "{question}"

        Document Chunks:
        {json.dumps(retrieved_chunks_for_rerank)}
        """
        try:
            rerank_response = await rerank_model.generate_content_async(rerank_prompt)
            cleaned_response = rerank_response.text.strip().replace("```json", "").replace("```", "")
            rerank_result = json.loads(cleaned_response)
            top_indices = rerank_result.get("relevant_indices", [])
            
            chunk_map = {chunk["index"]: chunk["text"] for chunk in retrieved_chunks_for_rerank}
            final_context = [chunk_map[i] for i in top_indices if i in chunk_map]

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logging.warning(f"Re-ranking with Gemini failed: {e}. Falling back to initial retrieval.")
            final_context = [chunk["text"] for chunk in retrieved_chunks_for_rerank[:5]]

        if not final_context:
            final_context = [chunk["text"] for chunk in retrieved_chunks_for_rerank[:5]]

        # 4. Generate the final answer (using the best Pro model)
        answer_model = genai.GenerativeModel('gemini-1.5-pro')
        answer_prompt = f'You are an expert AI assistant for analyzing policy documents. Your task is to answer the user\'s question based *only* on the provided context.\n\nContext:\n{"\n\n---\n\n".join(final_context)}\n\nQuestion: {question}\n\nInstructions:\n1. Analyze the context carefully.\n2. Provide a direct and concise answer.\n3. If the context has specific numbers or conditions, include them.\n4. If the context does not contain the answer, state: "The provided context does not contain the answer to this question."'
        
        answer_response = await answer_model.generate_content_async(answer_prompt)
        return Answer(question=question, answer=answer_response.text.strip(), context=final_context)

# --- API Endpoint ---
@app.post("/hackrx/run")
async def run_query_retrieval(request: HackRxRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")
    try:
        doc_data = await get_or_create_document_index(str(request.documents))
        semaphore = asyncio.Semaphore(5) # Increased semaphore slightly due to faster Flash model
        tasks = [process_single_question(q, doc_data, semaphore) for q in request.questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        answers = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logging.error(f"Error processing question '{request.questions[i]}': {res}", exc_info=True)
                answers.append(Answer(question=request.questions[i], answer="An error occurred while processing this question.", context=[]))
            else:
                answers.append(res)
        
        if SUBMISSION_MODE:
            return {"answers": [item.answer for item in answers]}
        else:
            return HackRxResponse(answers=answers)
            
    except Exception as e:
        logging.error(f"A critical error occurred in the main endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")