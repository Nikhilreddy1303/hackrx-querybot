import os
import faiss
import numpy as np
import logging
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
from cachetools import LRUCache
from google.generativeai.types import GenerationConfig

from read_url import read_url_and_extract_chunks
from nltk_setup import download_nltk_data

PRECOMPUTED_DIR = "precomputed_indices"

# --- Initialization & Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CACHE: LRUCache[str, Any] = LRUCache(maxsize=10)
SUBMISSION_MODE = os.getenv("SUBMISSION_MODE", "true").lower() == "true"
JSON_GENERATION_CONFIG = GenerationConfig(response_mime_type="application/json")

@asynccontextmanager
async def lifespan(app: FastAPI):
    download_nltk_data()
    logging.info("Starting to pre-load documents into cache...")
    if os.path.exists(PRECOMPUTED_DIR):
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
                        chunks_with_meta = json.load(f)
                    
                    tokenized_corpus = [c['text'].lower().split(" ") for c in chunks_with_meta]
                    bm25 = BM25Okapi(tokenized_corpus)
                    
                    # Use the HASH as the cache key
                    CACHE[url_hash] = {"faiss_index": index, "bm25_index": bm25, "chunks_with_meta": chunks_with_meta}
                    logging.info(f"Successfully pre-loaded a document into cache (Hash: {url_hash[:8]}...).")
                except Exception as e:
                    logging.error(f"Failed to pre-load cache for {doc_url}: {e}")
    yield

app = FastAPI(lifespan=lifespan, title="HackRx Advanced RAG System")
auth_scheme = HTTPBearer()

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str
    context: List[str]

class HackRxResponse(BaseModel):
    answers: List[Answer]

async def get_or_create_document_index(doc_url: str) -> Dict[str, Any]:
    # Create the hash from the incoming URL to use as the canonical key.
    url_hash = hashlib.md5(doc_url.encode()).hexdigest()

    if url_hash in CACHE:
        logging.info(f"Cache HIT for document hash: {url_hash[:8]}...")
        return CACHE[url_hash]

    logging.info(f"Cache MISS. Processing document: {doc_url}")
    document_chunks_with_meta = list(read_url_and_extract_chunks(doc_url))
    
    if not document_chunks_with_meta:
        raise HTTPException(status_code=400, detail="No valid content extracted.")
    
    document_texts = [c['text'] for c in document_chunks_with_meta]
    logging.info(f"Extracted {len(document_texts)} chunks. Generating indices...")
    
    embed_model = "models/text-embedding-004"
    result = await genai.embed_content_async(model=embed_model, content=document_texts, task_type="RETRIEVAL_DOCUMENT")
    doc_embeddings = np.array(result['embedding'], dtype='float32')
    
    faiss_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    faiss_index.add(doc_embeddings)

    tokenized_corpus = [doc.lower().split(" ") for doc in document_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    # When storing the newly processed document, use the hash as the key.
    CACHE[url_hash] = {"faiss_index": faiss_index, "bm25_index": bm25_index, "chunks_with_meta": document_chunks_with_meta}
    logging.info(f"Document processed and cached successfully (Hash: {url_hash[:8]}...).")
    return CACHE[url_hash]

async def expand_query(question: str) -> List[str]:
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f'You are a query expansion expert for technical documents. Generate 3 rephrased versions of the question that maintain the exact original intent and entities. Only improve searchability through synonyms or minor sentence structure changes. Respond with only a JSON object like this: {{"queries": ["query1", "query2", "query3"]}}\n\nOriginal Question: "{question}"'
    try:
        response = await model.generate_content_async(prompt, generation_config=JSON_GENERATION_CONFIG)
        result = json.loads(response.text)
        expanded = result.get("queries", [])
        return list(set([question] + expanded))
    except Exception as e:
        logging.warning(f"Query expansion failed: {e}. Using original question only.")
        return [question]

def reciprocal_rank_fusion(ranked_lists: List[List[int]], k: int = 60) -> List[int]:
    fused_scores = {}
    for rank_list in ranked_lists:
        for rank, doc_index in enumerate(rank_list):
            if doc_index not in fused_scores:
                fused_scores[doc_index] = 0
            fused_scores[doc_index] += 1 / (k + rank + 1)
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_index for doc_index, score in reranked_results]

async def process_single_question(question: str, doc_data: Dict[str, Any], semaphore: asyncio.Semaphore) -> Answer:
    async with semaphore:
        logging.info(f"Processing question: '{question[:30]}...'")
        faiss_index, bm25_index, chunks_with_meta = doc_data["faiss_index"], doc_data["bm25_index"], doc_data["chunks_with_meta"]
        chunks_text = [c['text'] for c in chunks_with_meta]

        all_queries = await expand_query(question)
        
        embed_model = "models/text-embedding-004"
        embedding_result = await genai.embed_content_async(
            model=embed_model, content=all_queries, task_type="RETRIEVAL_QUERY"
        )
        all_query_embeddings = embedding_result['embedding']

        faiss_results, bm25_results = [], []
        for query_text, query_embedding_list in zip(all_queries, all_query_embeddings):
            query_embedding_np = np.array(query_embedding_list, dtype='float32').reshape(1, -1)
            k_faiss = min(20, len(chunks_text))
            _, faiss_indices = faiss_index.search(query_embedding_np, k_faiss)
            faiss_results.append(list(faiss_indices[0]))
            
            tokenized_query = query_text.lower().split(" ")
            bm25_scores = await asyncio.to_thread(bm25_index.get_scores, tokenized_query)
            top_n_bm25_indices = np.argsort(bm25_scores)[::-1][:10].tolist()
            bm25_results.append(top_n_bm25_indices)
        
        fused_indices = reciprocal_rank_fusion(faiss_results + bm25_results)
        unique_fused_indices = list(dict.fromkeys(fused_indices))
        
        retrieved_chunks_for_rerank = [{"index": int(i), "text": chunks_with_meta[i]['text'], "source": chunks_with_meta[i]['source']} for i in unique_fused_indices[:20]]

        if not retrieved_chunks_for_rerank:
            return Answer(question=question, answer="Could not find relevant information.", context=[])

        rerank_model = genai.GenerativeModel('gemini-1.5-flash')
        rerank_prompt = f'You are a meticulous relevance expert for insurance documents. Your task is to identify the top 5 most relevant document chunks to answer the user\'s question. Prioritize chunks that contain specific numbers, limits, percentages, or explicit conditions. Use the "source" (e.g., page number) as a contextual clue.\nRespond with only a JSON object like this: {{"relevant_indices": [index1, index2, ...]}}\n\nUser Question: "{question}"\n\nDocument Chunks:\n{json.dumps(retrieved_chunks_for_rerank)}'
        
        final_context = []
        try:
            rerank_response = await rerank_model.generate_content_async(rerank_prompt, generation_config=JSON_GENERATION_CONFIG)
            rerank_result = json.loads(rerank_response.text)
            top_indices = rerank_result.get("relevant_indices", [])
            chunk_map = {chunk["index"]: chunk["text"] for chunk in retrieved_chunks_for_rerank}
            final_context = [chunk_map[i] for i in top_indices if i in chunk_map]
        except Exception as e:
            logging.warning(f"Re-ranking failed: {e}. Falling back.")
        
        if not final_context:
             final_context = [chunk["text"] for chunk in retrieved_chunks_for_rerank[:5]]

        answer_model = genai.GenerativeModel('gemini-1.5-pro')
        answer_prompt = f'You are a highly precise AI assistant. Your task is to answer the user\'s question based *only* on the provided context. You MUST NOT use any external knowledge.\n\nContext:\n{"\n\n---\n\n".join(final_context)}\n\nQuestion: {question}\n\nInstructions:\n1. Analyze the context carefully. Your entire answer must be derived from it.\n2. Provide a direct, concise answer.\n3. If the context contains specific numbers, dates, or conditions, you must include them in your answer.\n4. If the answer cannot be found in the context, you MUST state: "The provided context does not contain the answer to this question."'
        
        answer_response = await answer_model.generate_content_async(answer_prompt)
        return Answer(question=question, answer=answer_response.text.strip(), context=final_context)

@app.post("/hackrx/run")
async def run_query_retrieval(request: HackRxRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")
    try:
        # Use the string representation of the Pydantic HttpUrl
        doc_url_str = str(request.documents)
        doc_data = await get_or_create_document_index(doc_url_str)
        
        semaphore = asyncio.Semaphore(5)
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
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")