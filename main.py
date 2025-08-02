import os
import cohere
import faiss
import numpy as np
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any
from dotenv import load_dotenv

from read_url import read_url_and_extract_chunks

# --- Initialization & Caching Setup ---
load_dotenv()

# IMPROVEMENT: Configure logging for robust monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# IMPROVEMENT: Add a simple size limit to the in-memory cache
CACHE: Dict[str, Any] = {}
MAX_CACHE_SIZE = 10  # Store up to 10 processed documents

# IMPROVEMENT: Add a submission mode flag controlled by an environment variable
SUBMISSION_MODE = os.getenv("SUBMISSION_MODE", "false").lower() == "true"
if SUBMISSION_MODE:
    logging.warning("Application is running in SUBMISSION MODE. API will return a simple list of answers.")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise RuntimeError("COHERE_API_KEY not found in .env file")
co = cohere.Client(COHERE_API_KEY)


# --- API Models ---
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str
    context: List[str]

class HackRxResponse(BaseModel):
    answers: List[Answer]


# --- Caching and Processing Logic ---
def get_or_create_document_index(doc_url: str):
    if doc_url in CACHE:
        logging.info(f"Cache HIT for document: {doc_url}")
        return CACHE[doc_url]

    logging.info(f"Cache MISS. Processing document: {doc_url}")

    chunks_generator = read_url_and_extract_chunks(doc_url)
    document_chunks = [chunk['text'] for chunk in chunks_generator if chunk and chunk.get('text')]
    if not document_chunks:
        raise ValueError("Could not extract text from the document.")

    response = co.embed(
        texts=document_chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    doc_embeddings = np.array(response.embeddings, dtype='float32')

    embedding_dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(doc_embeddings)

    # Enforce cache size limit
    if len(CACHE) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(CACHE))
        del CACHE[oldest_key]
        logging.info(f"Cache full. Removed oldest item: {oldest_key}")

    CACHE[doc_url] = {"index": index, "chunks": document_chunks}
    logging.info(f"Cached new document index: {doc_url}")

    return CACHE[doc_url]


# --- API Endpoint ---
@app.post("/hackrx/run")
async def run_query_retrieval(request: HackRxRequest):
    structured_answers = []
    try:
        doc_data = get_or_create_document_index(str(request.documents))
        index = doc_data["index"]
        document_chunks = doc_data["chunks"]

        for question in request.questions:
            response = co.embed(
                texts=[question],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            question_embedding = np.array(response.embeddings, dtype='float32')

            k_initial = 10
            distances, indices = index.search(question_embedding, k_initial)
            initial_chunks = [document_chunks[i] for i in indices[0]]

            rerank_response = co.rerank(
                model="rerank-english-v3.0",
                query=question,
                documents=initial_chunks,
                top_n=3
            )

            final_context_chunks = []
            if rerank_response and rerank_response.results:
                for result in rerank_response.results:
                    if result.document and result.document.get('text'):
                        final_context_chunks.append(result.document['text'])

            if not final_context_chunks:
                final_context_chunks = initial_chunks[:3]

            context_str = "\n\n---\n\n".join(final_context_chunks)
            prompt = f"""Please answer the user's question based on the following context.\n\nContext:\n{context_str}\n\nQuestion:\n{question}"""
            
            answer_response = co.chat(
                model="command-r-plus",
                message=prompt,
                temperature=0.1
            )

            structured_answers.append(
                Answer(
                    question=question,
                    answer=answer_response.text,
                    context=final_context_chunks
                )
            )

        # Handle different response formats based on submission mode
        if SUBMISSION_MODE:
            # Return the simple list of answers for the automated platform
            final_answers = [item.answer for item in structured_answers]
            return {"answers": final_answers}
        else:
            # Return the detailed object for judging and testing
            return HackRxResponse(answers=structured_answers)

    except Exception as e:
        # Use logging to capture the full traceback in your logs
        logging.error(f"An error occurred in run_query_retrieval: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")