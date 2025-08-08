# main.py

import os
import faiss
import numpy as np
import logging
import logging.handlers
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
from datetime import datetime
from rank_bm25 import BM25Okapi
from cachetools import LRUCache
from google.generativeai.types import GenerationConfig
from google.generativeai.protos import Part

from read_url import read_url_and_extract_chunks
from nltk_setup import download_nltk_data
from agent_tools import make_http_get_request, GET_REQUEST_TOOL # <-- NEW IMPORT

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

PRECOMPUTED_DIR = "precomputed_indices"
CACHE: LRUCache[str, Any] = LRUCache(maxsize=20)
SUBMISSION_MODE = os.getenv("SUBMISSION_MODE", "true").lower() == "true"
JSON_GENERATION_CONFIG = GenerationConfig(response_mime_type="application/json")
EMBEDDING_BATCH_SIZE = 100
RERANKER_TIMEOUT = 15.0
FINAL_ANSWER_TIMEOUT = 30.0

# --- SETUP DEDICATED Q&A LOGGER ---
qa_logger = logging.getLogger('qa_logger')
qa_logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler('qa_log.jsonl', maxBytes=10*1024*1024, backupCount=3)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
qa_logger.addHandler(handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # This function runs once when the application starts up.
    download_nltk_data()
    logging.info("Application starting up...")
    logging.info("Starting to pre-load documents into cache from precomputed_indices/...")
    
    if os.path.exists(PRECOMPUTED_DIR):
        # List of URLs that you have pre-processed using preprocess.py
        # You can expand this list as needed.
        urls_to_preload = [
            "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
            "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D",
            "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D",
            "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
            "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
            "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
            "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
            "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            "https://hackrx.blob.core.windows.net/assets/Test%20/Test%20Case%20HackRx.pptx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A36%3A56Z&se=2026-08-05T18%3A36%3A00Z&sr=b&sp=r&sig=v3zSJ%2FKW4RhXaNNVTU9KQbX%2Bmo5dDEIzwaBzXCOicJM%3D",
            "https://hackrx.blob.core.windows.net/assets/Test%20/Mediclaim%20Insurance%20Policy.docx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A42%3A14Z&se=2026-08-05T18%3A42%3A00Z&sr=b&sp=r&sig=yvnP%2FlYfyyqYmNJ1DX51zNVdUq1zH9aNw4LfPFVe67o%3D",
            "https://hackrx.blob.core.windows.net/assets/Test%20/Salary%20data.xlsx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A46%3A54Z&se=2026-08-05T18%3A46%3A00Z&sr=b&sp=r&sig=sSoLGNgznoeLpZv%2FEe%2FEI1erhD0OQVoNJFDPtqfSdJQ%3D",
            "https://hackrx.blob.core.windows.net/assets/Test%20/Pincode%20data.xlsx?sv=2023-01-03&spr=https&st=2025-08-04T18%3A50%3A43Z&se=2026-08-05T18%3A50%3A00Z&sr=b&sp=r&sig=xf95kP3RtMtkirtUMFZn%2FFNai6sWHarZsTcvx8ka9mI%3D",
            "https://hackrx.blob.core.windows.net/assets/Test%20/image.png?sv=2023-01-03&spr=https&st=2025-08-04T19%3A21%3A45Z&se=2026-08-05T19%3A21%3A00Z&sr=b&sp=r&sig=lAn5WYGN%2BUAH7mBtlwGG4REw5EwYfsBtPrPuB0b18M4%3D",
            "https://hackrx.blob.core.windows.net/assets/hackrx_pdf.zip?sv=2023-01-03&spr=https&st=2025-08-04T09%3A25%3A45Z&se=2027-08-05T09%3A25%3A00Z&sr=b&sp=r&sig=rDL2ZcGX6XoDga5%2FTwMGBO9MgLOhZS8PUjvtga2cfVk%3D",
            "https://ash-speed.hetzner.com/10GB.bin",
            "https://hackrx.blob.core.windows.net/assets/Test%20/Fact%20Check.docx?sv=2023-01-03&spr=https&st=2025-08-04T20%3A27%3A22Z&se=2028-08-05T20%3A27%3A00Z&sr=b&sp=r&sig=XB1%2FNzJ57eg52j4xcZPGMlFrp3HYErCW1t7k1fMyiIc%3D",
            "https://hackrx.blob.core.windows.net/hackrx/rounds/FinalRound4SubmissionPDF.pdf?sv=2023-01-03&spr=https&st=2025-08-07T14%3A23%3A48Z&se=2027-08-08T14%3A23%3A00Z&sr=b&sp=r&sig=nMtZ2x9aBvz%2FPjRWboEOZIGB%2FaGfNf5TfBOrhGqSv4M%3D",
            "https://hackrx.blob.core.windows.net/hackrx/rounds/News.pdf?sv=2023-01-03&spr=https&st=2025-08-07T17%3A10%3A11Z&se=2026-08-08T17%3A10%3A00Z&sr=b&sp=r&sig=ybRsnfv%2B6VbxPz5xF7kLLjC4ehU0NF7KDkXua9ujSf0%3D"
        ]
        
        url_map = {url: hashlib.md5(url.encode()).hexdigest() for url in urls_to_preload}

        for doc_url, url_hash in url_map.items():
            index_path = os.path.join(PRECOMPUTED_DIR, f"{url_hash}.index")
            chunks_path = os.path.join(PRECOMPUTED_DIR, f"{url_hash}.json")

            if os.path.exists(index_path) and os.path.exists(chunks_path):
                try:
                    # Load the pre-computed FAISS index
                    index = faiss.read_index(index_path)
                    
                    # Load the pre-computed text chunks
                    with open(chunks_path, 'r') as f:
                        chunks_with_meta = json.load(f)
                    
                    if not chunks_with_meta:
                        logging.warning(f"Skipping empty pre-computed file: {chunks_path}")
                        continue
                    
                    # IMPORTANT: Re-create the BM25 index from the loaded chunks
                    tokenized_corpus = [c['text'].lower().split(" ") for c in chunks_with_meta]
                    bm25 = BM25Okapi(tokenized_corpus)
                    
                    # Store the complete set of indexes in the cache
                    CACHE[url_hash] = {
                        "faiss_index": index, 
                        "bm25_index": bm25, 
                        "chunks_with_meta": chunks_with_meta
                    }
                    logging.info(f"Successfully pre-loaded document into cache (Hash: {url_hash[:8]}...).")
                
                except Exception as e:
                    logging.error(f"Failed to pre-load cache for {doc_url}: {e}")

    yield
    
    # Code below yield runs on shutdown (e.g., for cleanup), but we don't need any for now.
    logging.info("Application shutting down.")

app = FastAPI(lifespan=lifespan, title="HackRx Hybrid Agentic RAG System")
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
    # This function remains the same.
    url_hash = hashlib.md5(doc_url.encode()).hexdigest()
    if url_hash in CACHE:
        logging.info(f"Cache HIT for document hash: {url_hash[:8]}...")
        return CACHE[url_hash]

    logging.info(f"Cache MISS. Processing document: {doc_url}")
    document_chunks_with_meta = list(read_url_and_extract_chunks(doc_url))
    
    if not document_chunks_with_meta:
        raise HTTPException(status_code=400, detail="No processable content extracted. File may be unsupported, too large, or inaccessible.")
    
    document_texts = [c['text'] for c in document_chunks_with_meta]
    logging.info(f"Extracted {len(document_texts)} chunks. Generating embeddings in batches...")
    
    all_embeddings = []
    embed_model = "models/text-embedding-004"
    for i in range(0, len(document_texts), EMBEDDING_BATCH_SIZE):
        batch = document_texts[i:i + EMBEDDING_BATCH_SIZE]
        try:
            result = await genai.embed_content_async(model=embed_model, content=batch, task_type="RETRIEVAL_DOCUMENT")
            all_embeddings.extend(result['embedding'])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to embed document content. {e}")
    
    doc_embeddings = np.array(all_embeddings, dtype='float32')
    faiss_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    faiss_index.add(doc_embeddings)

    tokenized_corpus = [doc.lower().split(" ") for doc in document_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    CACHE[url_hash] = {
        "faiss_index": faiss_index, 
        "bm25_index": bm25_index, 
        "chunks_with_meta": document_chunks_with_meta
    }
    logging.info(f"Document processed and cached successfully (Hash: {url_hash[:8]}...).")
    return CACHE[url_hash]

def reciprocal_rank_fusion(ranked_lists: List[List[int]], k: int = 60) -> List[int]:
    # This function remains the same.
    fused_scores = {}
    for rank_list in ranked_lists:
        for rank, doc_index in enumerate(rank_list):
            if doc_index not in fused_scores:
                fused_scores[doc_index] = 0
            fused_scores[doc_index] += 1 / (k + rank + 1)
    reranked_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_index for doc_index, score in reranked_results]

# --- UPGRADED RAG PIPELINE (WITH HYDE) ---
async def process_single_question(question: str, doc_data: Dict[str, Any], semaphore: asyncio.Semaphore) -> Answer:
    async with semaphore:
        logging.info(f"Processing question via RAG: '{question[:30]}...'")
        faiss_index, bm25_index, chunks_with_meta = doc_data["faiss_index"], doc_data["bm25_index"], doc_data["chunks_with_meta"]
        chunks_text = [c['text'] for c in chunks_with_meta]

        try:
            tokenized_query = question.lower().split(" ")
            bm25_scores = await asyncio.to_thread(bm25_index.get_scores, tokenized_query)
            top_n_bm25_indices = np.argsort(bm25_scores)[::-1][:3]
            grounding_context = "\n---\n".join([chunks_text[i] for i in top_n_bm25_indices])

            hyde_model = genai.GenerativeModel('gemini-1.5-flash')
            hyde_prompt = f"Using the provided context, generate a concise, single-paragraph answer to the user's question. Do not add any preamble.\n\nContext:\n{grounding_context}\n\nQuestion: \"{question}\""
            
            hyde_response = await asyncio.wait_for(hyde_model.generate_content_async(hyde_prompt), timeout=5.0)
            search_query_text = hyde_response.text
            logging.info(f"Using Grounded HyDE query: '{search_query_text[:60]}...'")
        except Exception as e:
            logging.warning(f"Grounded HyDE generation failed: {e}. Falling back to original question.")
            search_query_text = question

        embed_model = "models/text-embedding-004"
        embedding_result = await genai.embed_content_async(model=embed_model, content=search_query_text, task_type="RETRIEVAL_QUERY")
        query_embedding_np = np.array(embedding_result['embedding'], dtype='float32').reshape(1, -1)

        k_faiss = min(20, len(chunks_text))
        _, faiss_indices = faiss_index.search(query_embedding_np, k_faiss)
        
        fused_indices = reciprocal_rank_fusion([list(faiss_indices[0]), list(top_n_bm25_indices)])
        unique_fused_indices = list(dict.fromkeys(fused_indices))
        retrieved_chunks_for_rerank = [{"index": int(i), "text": chunks_with_meta[i]['text']} for i in unique_fused_indices[:25]]

        if not retrieved_chunks_for_rerank:
            return Answer(question=question, answer="Could not find relevant information.", context=[])

        rerank_model = genai.GenerativeModel('gemini-1.5-flash')
        rerank_prompt = f'You are a relevance expert. Re-rank the following document chunks based on how well they answer the user\'s question. Return a JSON object with a key "relevant_indices" containing a list of the top 5 most relevant chunk indices.\n\nUser Question: "{question}"\n\nDocument Chunks:\n{json.dumps(retrieved_chunks_for_rerank)}'
        
        final_context = []
        try:
            rerank_response = await asyncio.wait_for(
                rerank_model.generate_content_async(rerank_prompt, generation_config=JSON_GENERATION_CONFIG),
                timeout=RERANKER_TIMEOUT
            )
            rerank_result = json.loads(rerank_response.text)
            top_indices = rerank_result.get("relevant_indices", [])
            chunk_map = {chunk["index"]: chunk["text"] for chunk in retrieved_chunks_for_rerank}
            final_context = [chunk_map[i] for i in top_indices if i in chunk_map]
        except Exception:
            final_context = [chunk["text"] for chunk in retrieved_chunks_for_rerank[:5]]

        if not final_context:
            final_context = [chunk["text"] for chunk in retrieved_chunks_for_rerank[:5]]
        
        # --- THIS IS THE FIX ---
        # We create the string with backslashes *before* the f-string.
        context_joiner = "\n\n---\n\n"
        formatted_context = context_joiner.join(final_context)
        # --- END FIX ---
        
        answer_model = genai.GenerativeModel('gemini-1.5-pro')
        answer_prompt = f'You are a precise AI assistant. Answer the user\'s question based *only* on the provided context. Do not use external knowledge.\n\nContext:\n{formatted_context}\n\nQuestion: {question}\n\nInstructions:\n1. Provide a direct, concise answer.\n2. If the answer cannot be found, state: "The provided context does not contain the answer to this question."'
        
        answer_response = await asyncio.wait_for(answer_model.generate_content_async(answer_prompt), timeout=FINAL_ANSWER_TIMEOUT)
        return Answer(question=question, answer=answer_response.text.strip(), context=final_context)

# --- MAIN ENDPOINT WITH ROUTER ---
@app.post("/hackrx/run")
async def run_query_retrieval(request: HackRxRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")

    # --- Step 1: Securely Fetch Document and Handle URL Errors ---
    try:
        doc_url_str = str(request.documents)
        doc_data = await get_or_create_document_index(doc_url_str)
    except HTTPException as e:
        # This block now cleanly handles the bad URL case and returns a 400.
        # It is NOT nested inside another try...except, so the error isn't converted to a 500.
        logging.error(f"Failed to process document due to client-side error: {e.detail}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the document from the provided URL. Please check if the URL is valid and accessible. Reason: {e.detail}"
        )

    # --- Step 2: Main Logic (Router, Agent, RAG) ---
    # This block now only runs if the document was fetched successfully.
    # It has its own try...except to catch unexpected errors in the AI logic.
    try:
        document_text = "\n---\n".join([c['text'] for c in doc_data['chunks_with_meta']])
        
        # The Agentic Router Logic
        if len(document_text) > 4000:
            router_context = document_text[:2000] + "\n...\n" + document_text[-2000:]
        else:
            router_context = document_text
        router_model = genai.GenerativeModel('gemini-1.5-pro')
        router_prompt = f"""You are an expert task router. Your job is to classify the user's intent based on the document's content. Respond with only a single word: 'AGENT' or 'RAG'.
        **Examine the document for instructions or action words.** Look for keywords like 'API', 'endpoint', 'GET', 'POST', 'call this', 'step-by-step', 'if the landmark is... then call...'.
        **Example 1 (Agentic):** - Document: "Step 1: Call the API at https://.../getData. Step 2: Process the result." - Your decision: AGENT
        **Example 2 (RAG):** - Document: "The policy covers damages up to $50,000. The waiting period is 30 days." - Your decision: RAG
        ---
        **Current Task:**
        Document Content Sample: "{router_context}"
        User Questions: "{' '.join(request.questions)}"
        """
        router_response = await router_model.generate_content_async(router_prompt)
        intent = router_response.text.strip().upper()
        logging.info(f"Routing intent classified as: {intent}")

        answers = []
        if "AGENT" in intent:
            # AGENTIC LOGIC
            # ... (The full agentic loop code as provided before)
            logging.info("Executing task via AGENT engine...")
            agent_model = genai.GenerativeModel('gemini-1.5-pro', tools=[GET_REQUEST_TOOL])
            tool_executors = {"make_http_get_request": make_http_get_request}
            chat = agent_model.start_chat()
            agent_prompt = f"You are a helpful assistant. Please follow the instructions in the document to answer the user's question(s). \n\nDocument Content:\n{document_text}\n\nUser Question(s): {' '.join(request.questions)}"
            response = await asyncio.wait_for(chat.send_message_async(agent_prompt), timeout=60.0)
            max_turns = 5
            for turn in range(max_turns):
                part = response.candidates[0].content.parts[0]
                if part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    if function_name in tool_executors:
                        function_to_call = tool_executors[function_name]
                        args = dict(function_call.args)
                        logging.info(f"Agent requesting to call tool '{function_name}' with args: {args}")
                        tool_response = await asyncio.to_thread(function_to_call, **args)
                        response = await asyncio.wait_for(
                            chat.send_message_async([Part(function_response={"name": function_name, "response": {"content": tool_response}})]),
                            timeout=60.0
                        )
                    else:
                        final_answer = f"Error: Agent tried to call unknown function '{function_name}'."
                        break
                elif part.text:
                    final_answer = part.text
                    break
                else:
                    final_answer = "Error: Agent returned an unexpected response type."
                    break
            else:
                final_answer = "Error: Agent exceeded maximum turns without providing a final answer."
            for q in request.questions:
                answers.append(Answer(question=q, answer=final_answer.strip(), context=[document_text]))

        else:
            # RAG LOGIC
            logging.info("Executing task via RAG engine...")
            semaphore = asyncio.Semaphore(5)
            tasks = [process_single_question(q, doc_data, semaphore) for q in request.questions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
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
        # This is the final safety net for UNEXPECTED errors in the AI logic.
        logging.error(f"A critical, unhandled error occurred in the main logic block: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")