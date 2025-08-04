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
        
        # --- UPDATED: Includes all 8 unlocked document URLs ---
        url_map = {
            "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D": hashlib.md5("https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D".encode()).hexdigest(),
            "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D": hashlib.md5("https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D".encode()).hexdigest(),
            "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D": hashlib.md5("https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D".encode()).hexdigest(),
            "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D": hashlib.md5("https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D".encode()).hexdigest(),
            "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D": hashlib.md5("https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D".encode()).hexdigest(),
            "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D": hashlib.md5("https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D".encode()).hexdigest(),
            "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D": hashlib.md5("https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D".encode()).hexdigest(),
            "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D": hashlib.md5("https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D".encode()).hexdigest()
        }
        
        for doc_url, url_hash in url_map.items():
            index_path = os.path.join(PRECOMPUTED_DIR, f"{url_hash}.index")
            chunks_path = os.path.join(PRECOMPUTED_DIR, f"{url_hash}.json")
            
            if os.path.exists(index_path) and os.path.exists(chunks_path):
                try:
                    index = faiss.read_index(index_path)
                    with open(chunks_path, 'r') as f:
                        chunks = json.load(f)
                    
                    CACHE[doc_url] = {"index": index, "chunks": chunks}
                    logging.info(f"Successfully pre-loaded a document into cache.")
                except Exception as e:
                    logging.error(f"Failed to pre-load cache for {doc_url}: {e}")
    yield
    # Cleanup can be added here if needed

# --- Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(lifespan=lifespan, title="HackRx Gemini RAG System")
auth_scheme = HTTPBearer()

# --- Configuration ---
CACHE: Dict[str, Any] = {}
MAX_CACHE_SIZE = 10
SUBMISSION_MODE = os.getenv("SUBMISSION_MODE", "true").lower() == "true"

# Configure the Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

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
    """
    Processes a document by chunking, embedding, and creating a FAISS index.
    Results are cached in memory to avoid reprocessing.
    """
    if doc_url in CACHE:
        logging.info(f"Cache HIT for document: {doc_url}")
        return CACHE[doc_url]

    logging.info(f"Cache MISS. Processing document: {doc_url}")
    
    chunks_generator = read_url_and_extract_chunks(doc_url)
    document_chunks = [c['text'].strip() for c in chunks_generator if c and c.get('text') and len(c['text'].strip()) > 50]

    if not document_chunks:
        raise HTTPException(status_code=400, detail="No valid content extracted from document.")

    logging.info(f"Extracted {len(document_chunks)} chunks. Generating embeddings...")
    
    # Generate embeddings in batches for robustness
    embed_model = "models/text-embedding-004"
    result = await genai.embed_content_async(
        model=embed_model,
        content=document_chunks,
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    doc_embeddings = np.array(result['embedding'], dtype='float32')
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    if len(CACHE) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(CACHE))
        del CACHE[oldest_key]

    CACHE[doc_url] = {"index": index, "chunks": document_chunks}
    logging.info(f"Document processed and cached successfully.")
    return CACHE[doc_url]

async def process_single_question(question: str, doc_data: Dict[str, Any], semaphore: asyncio.Semaphore) -> Answer:
    """
    Processes a single question, but waits for the semaphore before starting.
    """
    async with semaphore:
        logging.info(f"Processing question: '{question[:30]}...'")
        index, chunks = doc_data["index"], doc_data["chunks"]

        # 1. Retrieve
        embed_model = "models/text-embedding-004"
        query_embedding_result = await genai.embed_content_async(
            model=embed_model,
            content=question,
            task_type="RETRIEVAL_QUERY"
        )
        query_embedding = np.array(query_embedding_result['embedding'], dtype='float32').reshape(1, -1)
        
        k = min(15, len(chunks))
        _, indices = index.search(query_embedding, k)
        
        retrieved_chunks = [{"index": int(i), "text": chunks[i]} for i in indices[0] if 0 <= i < len(chunks)]

        if not retrieved_chunks:
            return Answer(question=question, answer="Could not find relevant information in the document.", context=[])

        # 2. Re-rank
        rerank_model = genai.GenerativeModel('gemini-1.5-flash')
        rerank_prompt = f'Your task is to evaluate the relevance of the following document chunks to the user\'s question. Respond with only a JSON object containing a key "relevant_indices" with a list of the top 3 most relevant chunk indices in order of relevance.\n\nUser Question: "{question}"\n\nDocument Chunks:\n{json.dumps(retrieved_chunks)}'
        
        try:
            rerank_response = await rerank_model.generate_content_async(rerank_prompt)
            cleaned_response = rerank_response.text.strip().replace("```json", "").replace("```", "")
            rerank_result = json.loads(cleaned_response)
            top_indices = rerank_result.get("relevant_indices", [])
            
            chunk_map = {chunk["index"]: chunk["text"] for chunk in retrieved_chunks}
            final_context = [chunk_map[i] for i in top_indices if i in chunk_map]

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logging.warning(f"Re-ranking with Gemini failed: {e}. Falling back.")
            final_context = [chunk["text"] for chunk in retrieved_chunks[:3]]

        if not final_context:
            final_context = [chunk["text"] for chunk in retrieved_chunks[:3]]

        # 3. Generate
        answer_model = genai.GenerativeModel('gemini-1.5-flash')
        answer_prompt = f'Based on the provided context, answer the question accurately and concisely.\n\nContext:\n{"\n\n---\n\n".join(final_context)}\n\nQuestion: {question}\n\nInstructions:\n- Answer directly based on the context.\n- If the context does not contain the answer, state that clearly.'
        
        answer_response = await answer_model.generate_content_async(answer_prompt)
        return Answer(question=question, answer=answer_response.text.strip(), context=final_context)

# --- API Endpoint ---
@app.post("/hackrx/run")
async def run_query_retrieval(request: HackRxRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")

    try:
        doc_data = await get_or_create_document_index(str(request.documents))
        
        # Create a semaphore to limit concurrency to 5 questions at a time
        semaphore = asyncio.Semaphore(5)
        
        # Pass the semaphore to each task
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