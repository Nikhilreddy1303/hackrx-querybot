# import os
# import cohere
# import faiss
# import numpy as np
# import logging
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, HttpUrl
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# import nltk
# import ssl
# from fastapi import FastAPI, Depends, HTTPException # <-- Make sure Depends is imported  #!
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials # <-- Import these   #!
# nltk.download('punkt_tab')
# nltk.download('punkt')
# nltk.download('stopwords')

# from read_url import read_url_and_extract_chunks
# from nltk_setup import download_nltk_data # <-- Verify this import exists

# # --- Initialization & Caching Setup ---
# load_dotenv()

# download_nltk_data() # <-- Verify this function call exists

# auth_scheme = HTTPBearer()   #!

# # IMPROVEMENT: Configure logging for robust monitoring
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# app = FastAPI()

# # IMPROVEMENT: Add a simple size limit to the in-memory cache
# CACHE: Dict[str, Any] = {}
# MAX_CACHE_SIZE = 10  # Store up to 10 processed documents

# # IMPROVEMENT: Add a submission mode flag controlled by an environment variable
# SUBMISSION_MODE = os.getenv("SUBMISSION_MODE", "false").lower() == "true"
# if SUBMISSION_MODE:
#     logging.warning("Application is running in SUBMISSION MODE. API will return a simple list of answers.")

# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# if not COHERE_API_KEY:
#     raise RuntimeError("COHERE_API_KEY not found in .env file")
# co = cohere.Client(COHERE_API_KEY)


# # --- API Models ---
# class HackRxRequest(BaseModel):
#     documents: HttpUrl
#     questions: List[str]

# class Answer(BaseModel):
#     question: str
#     answer: str
#     context: List[str]

# class HackRxResponse(BaseModel):
#     answers: List[Answer]


# # --- Caching and Processing Logic ---
# def get_or_create_document_index(doc_url: str):
#     if doc_url in CACHE:
#         logging.info(f"Cache HIT for document: {doc_url}")
#         return CACHE[doc_url]

#     logging.info(f"Cache MISS. Processing document: {doc_url}")

#     chunks_generator = read_url_and_extract_chunks(doc_url)
#     document_chunks = [chunk['text'] for chunk in chunks_generator if chunk and chunk.get('text')]
#     if not document_chunks:
#         raise ValueError("Could not extract text from the document.")

#     response = co.embed(
#         texts=document_chunks,
#         model="embed-english-v3.0",
#         input_type="search_document"
#     )
#     doc_embeddings = np.array(response.embeddings, dtype='float32')

#     embedding_dim = doc_embeddings.shape[1]
#     index = faiss.IndexFlatL2(embedding_dim)
#     index.add(doc_embeddings)

#     # Enforce cache size limit
#     if len(CACHE) >= MAX_CACHE_SIZE:
#         oldest_key = next(iter(CACHE))
#         del CACHE[oldest_key]
#         logging.info(f"Cache full. Removed oldest item: {oldest_key}")

#     CACHE[doc_url] = {"index": index, "chunks": document_chunks}
#     logging.info(f"Cached new document index: {doc_url}")

#     return CACHE[doc_url]


# # --- API Endpoint ---
# @app.post("/hackrx/run")
# async def run_query_retrieval(request: HackRxRequest,token: HTTPAuthorizationCredentials = Depends(auth_scheme)):    #!
#     structured_answers = []
#     try:
#         doc_data = get_or_create_document_index(str(request.documents))
#         index = doc_data["index"]
#         document_chunks = doc_data["chunks"]

#         for question in request.questions:
#             response = co.embed(
#                 texts=[question],
#                 model="embed-english-v3.0",
#                 input_type="search_query"
#             )
#             question_embedding = np.array(response.embeddings, dtype='float32')

#             k_initial = 10
#             distances, indices = index.search(question_embedding, k_initial)
#             initial_chunks = [document_chunks[i] for i in indices[0]]

#             rerank_response = co.rerank(
#                 model="rerank-english-v3.0",
#                 query=question,
#                 documents=initial_chunks,
#                 top_n=3
#             )

#             final_context_chunks = []
#             if rerank_response and rerank_response.results:
#                 for result in rerank_response.results:
#                     if result.document and result.document.get('text'):
#                         final_context_chunks.append(result.document['text'])

#             if not final_context_chunks:
#                 final_context_chunks = initial_chunks[:3]

#             context_str = "\n\n---\n\n".join(final_context_chunks)
#             prompt = f"""Please answer the user's question based on the following context.\n\nContext:\n{context_str}\n\nQuestion:\n{question}"""
            
#             answer_response = co.chat(
#                 model="command-r-plus",
#                 message=prompt,
#                 temperature=0.1
#             )

#             structured_answers.append(
#                 Answer(
#                     question=question,
#                     answer=answer_response.text,
#                     context=final_context_chunks
#                 )
#             )

#         # Handle different response formats based on submission mode
#         if SUBMISSION_MODE:
#             # Return the simple list of answers for the automated platform
#             final_answers = [item.answer for item in structured_answers]
#             return {"answers": final_answers}
#         else:
#             # Return the detailed object for judging and testing
#             return HackRxResponse(answers=structured_answers)

#     except Exception as e:
#         # Use logging to capture the full traceback in your logs
#         logging.error(f"An error occurred in run_query_retrieval: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")



# import os
# import cohere
# import faiss
# import numpy as np
# import logging
# import time
# import asyncio
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel, HttpUrl
# from typing import List, Dict, Any, Optional
# from dotenv import load_dotenv
# import nltk
# from contextlib import asynccontextmanager

# from read_url import read_url_and_extract_chunks
# from nltk_setup import download_nltk_data

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     download_nltk_data()
#     nltk.download('punkt_tab', quiet=True)
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     yield

# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log'),
#         logging.StreamHandler()
#     ]
# )

# app = FastAPI(lifespan=lifespan, title="HackRx Query-Retrieval System")
# auth_scheme = HTTPBearer()

# CACHE: Dict[str, Any] = {}
# MAX_CACHE_SIZE = 10
# RATE_LIMIT_DELAY = 1.2

# SUBMISSION_MODE = os.getenv("SUBMISSION_MODE", "false").lower() == "true"
# if SUBMISSION_MODE:
#     logging.warning("Application is running in SUBMISSION MODE.")

# COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# if not COHERE_API_KEY:
#     raise RuntimeError("COHERE_API_KEY not found in .env file")

# co = cohere.Client(COHERE_API_KEY)

# class HackRxRequest(BaseModel):
#     documents: HttpUrl
#     questions: List[str]

# class Answer(BaseModel):
#     question: str
#     answer: str
#     context: List[str]

# class HackRxResponse(BaseModel):
#     answers: List[Answer]

# class RateLimiter:
#     def __init__(self, calls_per_minute: int = 50):
#         self.calls_per_minute = calls_per_minute
#         self.calls = []
    
#     async def wait_if_needed(self):
#         now = time.time()
#         self.calls = [call_time for call_time in self.calls if now - call_time < 60]
        
#         if len(self.calls) >= self.calls_per_minute:
#             sleep_time = 60 - (now - self.calls[0]) + 1
#             if sleep_time > 0:
#                 logging.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
#                 await asyncio.sleep(sleep_time)
        
#         self.calls.append(now)

# rate_limiter = RateLimiter(calls_per_minute=45)

# async def safe_cohere_request(func, *args, max_retries: int = 3, **kwargs):
#     for attempt in range(max_retries):
#         try:
#             await rate_limiter.wait_if_needed()
#             result = func(*args, **kwargs)
#             await asyncio.sleep(RATE_LIMIT_DELAY)
#             return result
        
#         except cohere.errors.TooManyRequestsError as e:
#             wait_time = min(60, (2 ** attempt) * 5)
#             logging.warning(f"Rate limit hit on attempt {attempt + 1}, waiting {wait_time}s")
#             if attempt < max_retries - 1:
#                 await asyncio.sleep(wait_time)
#             else:
#                 raise HTTPException(
#                     status_code=429,
#                     detail="Rate limit exceeded. Please try again later."
#                 )
        
#         except Exception as e:
#             if attempt < max_retries - 1:
#                 wait_time = (2 ** attempt) * 2
#                 logging.warning(f"Request failed: {e}, retrying in {wait_time}s")
#                 await asyncio.sleep(wait_time)
#             else:
#                 raise e

# def process_chunks_in_batches(chunks: List[str], batch_size: int = 30) -> List[List[float]]:
#     all_embeddings = []
    
#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i + batch_size]
#         try:
#             response = co.embed(
#                 texts=batch,
#                 model="embed-english-v3.0",
#                 input_type="search_document",
#                 truncate="END"
#             )
#             all_embeddings.extend(response.embeddings)
#             time.sleep(RATE_LIMIT_DELAY)
#         except Exception as e:
#             logging.error(f"Error processing batch {i//batch_size + 1}: {e}")
#             raise
    
#     return all_embeddings

# async def get_or_create_document_index(doc_url: str) -> Dict[str, Any]:
#     if doc_url in CACHE:
#         logging.info(f"Cache HIT for document: {doc_url}")
#         return CACHE[doc_url]

#     logging.info(f"Cache MISS. Processing document: {doc_url}")

#     try:
#         chunks_generator = read_url_and_extract_chunks(doc_url)
#         document_chunks = []
        
#         for chunk in chunks_generator:
#             if chunk and chunk.get('text') and len(chunk['text'].strip()) > 50:
#                 document_chunks.append(chunk['text'].strip())
        
#         if not document_chunks:
#             raise ValueError("Could not extract meaningful text from the document.")

#         logging.info(f"Extracted {len(document_chunks)} chunks from document")

#         embeddings = process_chunks_in_batches(document_chunks, batch_size=25)
#         doc_embeddings = np.array(embeddings, dtype='float32')

#         embedding_dim = doc_embeddings.shape[1]
#         index = faiss.IndexFlatL2(embedding_dim)
#         index.add(doc_embeddings)

#         if len(CACHE) >= MAX_CACHE_SIZE:
#             oldest_key = next(iter(CACHE))
#             del CACHE[oldest_key]
#             logging.info(f"Cache full. Removed oldest item: {oldest_key}")

#         CACHE[doc_url] = {
#             "index": index, 
#             "chunks": document_chunks,
#             "embeddings_count": len(embeddings)
#         }
#         logging.info(f"Cached new document index: {doc_url}")

#         return CACHE[doc_url]
    
#     except Exception as e:
#         logging.error(f"Error processing document: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# def extract_rerank_text(result) -> Optional[str]:
#     if not result or not hasattr(result, 'document'):
#         return None
    
#     document = result.document
    
#     if hasattr(document, 'text'):
#         return document.text
#     elif isinstance(document, dict):
#         return document.get('text')
#     elif isinstance(document, str):
#         return document
#     else:
#         try:
#             return str(document)
#         except:
#             return None

# async def process_single_question(question: str, doc_data: Dict[str, Any]) -> Answer:
#     try:
#         index = doc_data["index"]
#         document_chunks = doc_data["chunks"]

#         response = await safe_cohere_request(
#             co.embed,
#             texts=[question],
#             model="embed-english-v3.0",
#             input_type="search_query",
#             truncate="END"
#         )
        
#         question_embedding = np.array(response.embeddings, dtype='float32')

#         k_initial = min(15, len(document_chunks))
#         distances, indices = index.search(question_embedding, k_initial)
        
#         valid_indices = [i for i in indices[0] if 0 <= i < len(document_chunks)]
#         initial_chunks = [document_chunks[i] for i in valid_indices]

#         if not initial_chunks:
#             return Answer(
#                 question=question,
#                 answer="No relevant information found in the document.",
#                 context=[]
#             )

#         try:
#             rerank_response = await safe_cohere_request(
#                 co.rerank,
#                 model="rerank-english-v3.0",
#                 query=question,
#                 documents=initial_chunks,
#                 top_n=min(5, len(initial_chunks)),
#                 return_documents=True
#             )

#             final_context_chunks = []
#             if rerank_response and hasattr(rerank_response, 'results'):
#                 for result in rerank_response.results:
#                     text = extract_rerank_text(result)
#                     if text and len(text.strip()) > 20:
#                         final_context_chunks.append(text.strip())

#         except Exception as rerank_error:
#             logging.warning(f"Rerank failed for question '{question}': {rerank_error}")
#             final_context_chunks = initial_chunks[:3]

#         if not final_context_chunks:
#             final_context_chunks = initial_chunks[:3]

#         context_str = "\n\n---\n\n".join(final_context_chunks[:3])
        
#         prompt = f"""Based on the provided context, answer the question accurately and concisely.

# Context:
# {context_str}

# Question: {question}

# Instructions:
# - Answer directly based on the context
# - If the context doesn't contain the answer, say so clearly
# - Be specific and cite relevant details from the context"""

#         answer_response = await safe_cohere_request(
#             co.chat,
#             model="command-r-plus",
#             message=prompt,
#             temperature=0.1,
#             max_tokens=500
#         )

#         return Answer(
#             question=question,
#             answer=answer_response.text.strip(),
#             context=final_context_chunks[:3]
#         )

#     except Exception as e:
#         logging.error(f"Error processing question '{question}': {e}")
#         return Answer(
#             question=question,
#             answer=f"Error processing question: {str(e)}",
#             context=[]
#         )

# @app.post("/hackrx/run")
# async def run_query_retrieval(
#     request: HackRxRequest, 
#     token: HTTPAuthorizationCredentials = Depends(auth_scheme)
# ):
#     start_time = time.time()
    
#     try:
#         if not request.questions:
#             raise HTTPException(status_code=400, detail="No questions provided")

#         doc_data = await get_or_create_document_index(str(request.documents))
        
#         tasks = [process_single_question(question, doc_data) for question in request.questions]
#         structured_answers = await asyncio.gather(*tasks, return_exceptions=True)

#         valid_answers = []
#         for i, result in enumerate(structured_answers):
#             if isinstance(result, Exception):
#                 logging.error(f"Question {i} failed: {result}")
#                 valid_answers.append(Answer(
#                     question=request.questions[i],
#                     answer="Error processing this question.",
#                     context=[]
#                 ))
#             else:
#                 valid_answers.append(result)

#         processing_time = time.time() - start_time
#         logging.info(f"Processed {len(request.questions)} questions in {processing_time:.2f}s")

#         if SUBMISSION_MODE:
#             return {"answers": [answer.answer for answer in valid_answers]}
#         else:
#             return HackRxResponse(answers=valid_answers)

#     except HTTPException:
#         raise
#     except Exception as e:
#         logging.error(f"Error in run_query_retrieval: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "cache_size": len(CACHE),
#         "submission_mode": SUBMISSION_MODE
#     }

# @app.get("/cache/stats")
# async def cache_stats():
#     stats = {
#         "cached_documents": len(CACHE),
#         "max_cache_size": MAX_CACHE_SIZE,
#         "documents": {}
#     }
    
#     for url, data in CACHE.items():
#         stats["documents"][url] = {
#             "chunks_count": len(data.get("chunks", [])),
#             "embeddings_count": data.get("embeddings_count", 0)
#         }
    
#     return stats

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# import os
# import cohere
# import faiss
# import numpy as np
# import logging
# import time
# import asyncio
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from pydantic import BaseModel, HttpUrl
# from typing import List, Dict, Any, Optional
# from dotenv import load_dotenv
# import nltk
# from contextlib import asynccontextmanager

# from read_url import read_url_and_extract_chunks
# from nltk_setup import download_nltk_data

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     download_nltk_data()
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     yield

# load_dotenv()

# logging.basicConfig(
#     level=logging.INFO, 
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log'),
#         logging.StreamHandler()
#     ]
# )

# app = FastAPI(lifespan=lifespan, title="HackRx Query-Retrieval System")
# auth_scheme = HTTPBearer()

# CACHE: Dict[str, Any] = {}
# MAX_CACHE_SIZE = 10
# RATE_LIMIT_DELAY = 1.2

# SUBMISSION_MODE = os.getenv("SUBMISSION_MODE", "false").lower() == "true"
# if SUBMISSION_MODE:
#     logging.warning("Application is running in SUBMISSION MODE.")

# API_KEYS = [
#     os.getenv("COHERE_API_KEY_1"),
#     os.getenv("COHERE_API_KEY_2"),
#     os.getenv("COHERE_API_KEY_3"),
#     os.getenv("COHERE_API_KEY_4"),
#     os.getenv("COHERE_API_KEY_5")
# ]

# if not all(API_KEYS):
#     raise RuntimeError("All 5 Cohere API keys must be set in the .env file.")

# class HackRxRequest(BaseModel):
#     documents: HttpUrl
#     questions: List[str]

# class Answer(BaseModel):
#     question: str
#     answer: str
#     context: List[str]

# class HackRxResponse(BaseModel):
#     answers: List[Answer]

# class RateLimiter:
#     def __init__(self, calls_per_minute: int = 45):
#         self.calls_per_minute = calls_per_minute
#         self.calls: Dict[str, List[float]] = {key: [] for key in API_KEYS}

#     async def wait_for_available_key(self) -> str:
#         while True:
#             now = time.time()
#             for key, call_times in self.calls.items():
#                 self.calls[key] = [t for t in call_times if now - t < 60]
#                 if len(self.calls[key]) < self.calls_per_minute:
#                     self.calls[key].append(now)
#                     return key
#             await asyncio.sleep(1)

# rate_limiter = RateLimiter()

# async def safe_cohere_request(method_name: str, *args, max_retries: int = 3, **kwargs):
#     for attempt in range(max_retries):
#         try:
#             api_key = await rate_limiter.wait_for_available_key()
#             client = cohere.Client(api_key)
#             method = getattr(client, method_name)
#             result = method(*args, **kwargs)
#             await asyncio.sleep(RATE_LIMIT_DELAY)
#             return result
#         except cohere.errors.CohereError as e:
#             logging.warning(f"Cohere API error on attempt {attempt+1}: {e}")
#             if attempt == max_retries - 1:
#                 raise HTTPException(status_code=500, detail=str(e))
#             await asyncio.sleep(2 ** attempt)
#         except Exception as e:
#             logging.error(f"Unexpected error: {e}")
#             if attempt == max_retries - 1:
#                 raise HTTPException(status_code=500, detail=str(e))
#             await asyncio.sleep(2 ** attempt)

# async def process_chunks_in_batches(chunks: List[str], batch_size: int = 30) -> List[List[float]]:
#     all_embeddings = []
#     for i in range(0, len(chunks), batch_size):
#         batch = chunks[i:i + batch_size]
#         response = await safe_cohere_request(
#             "embed",
#             texts=batch,
#             model="embed-english-v3.0",
#             input_type="search_document",
#             truncate="END"
#         )
#         all_embeddings.extend(response.embeddings)
#         time.sleep(RATE_LIMIT_DELAY)
#     return all_embeddings

# async def get_or_create_document_index(doc_url: str) -> Dict[str, Any]:
#     if doc_url in CACHE:
#         logging.info(f"Cache HIT for document: {doc_url}")
#         return CACHE[doc_url]

#     logging.info(f"Cache MISS. Processing document: {doc_url}")

#     chunks_generator = read_url_and_extract_chunks(doc_url)
#     document_chunks = [chunk['text'].strip() for chunk in chunks_generator if chunk.get('text') and len(chunk['text'].strip()) > 50]

#     if not document_chunks:
#         raise HTTPException(status_code=400, detail="No valid content extracted from document.")

#     embeddings = await process_chunks_in_batches(document_chunks)
#     doc_embeddings = np.array(embeddings, dtype='float32')
#     embedding_dim = doc_embeddings.shape[1]

#     index = faiss.IndexFlatL2(embedding_dim)
#     index.add(doc_embeddings)

#     if len(CACHE) >= MAX_CACHE_SIZE:
#         oldest_key = next(iter(CACHE))
#         del CACHE[oldest_key]

#     CACHE[doc_url] = {
#         "index": index,
#         "chunks": document_chunks,
#         "embeddings_count": len(embeddings)
#     }
#     return CACHE[doc_url]

# def extract_rerank_text(result) -> Optional[str]:
#     doc = getattr(result, 'document', None)
#     if isinstance(doc, str):
#         return doc
#     elif isinstance(doc, dict):
#         return doc.get('text')
#     return None

# async def process_single_question(question: str, doc_data: Dict[str, Any]) -> Answer:
#     index = doc_data["index"]
#     chunks = doc_data["chunks"]

#     response = await safe_cohere_request(
#         "embed",
#         texts=[question],
#         model="embed-english-v3.0",
#         input_type="search_query",
#         truncate="END"
#     )
#     question_embedding = np.array(response.embeddings, dtype='float32')
#     k = min(15, len(chunks))
#     distances, indices = index.search(question_embedding, k)
#     retrieved = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

#     try:
#         rerank = await safe_cohere_request(
#             "rerank",
#             model="rerank-english-v3.0",
#             query=question,
#             documents=retrieved,
#             top_n=min(5, len(retrieved)),
#             return_documents=True
#         )
#         context = [extract_rerank_text(r) for r in rerank.results if extract_rerank_text(r)]
#     except Exception:
#         context = retrieved[:3]

#     prompt = f"""Based on the provided context, answer the question accurately.

# Context:
# {chr(10).join(context)}

# Question: {question}

# Instructions:
# - Answer directly based on the context
# - If the context doesn't contain the answer, say so clearly
# - Be specific and cite relevant details from the context"""

#     answer = await safe_cohere_request(
#         "chat",
#         model="command-r-plus",
#         message=prompt,
#         temperature=0.1,
#         max_tokens=500
#     )

#     return Answer(question=question, answer=answer.text.strip(), context=context)

# @app.post("/hackrx/run")
# async def run_query_retrieval(request: HackRxRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
#     if not request.questions:
#         raise HTTPException(status_code=400, detail="No questions provided")

#     doc_data = await get_or_create_document_index(str(request.documents))
#     tasks = [process_single_question(q, doc_data) for q in request.questions]
#     results = await asyncio.gather(*tasks, return_exceptions=True)

#     answers = []
#     for i, res in enumerate(results):
#         if isinstance(res, Exception):
#             answers.append(Answer(question=request.questions[i], answer="Error occurred.", context=[]))
#         else:
#             answers.append(res)

#     return HackRxResponse(answers=answers)

# @app.get("/health")
# async def health():
#     return {"status": "ok", "cache_size": len(CACHE)}



import os
import cohere
import faiss
import numpy as np
import logging
import time
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import nltk
from contextlib import asynccontextmanager

from read_url import read_url_and_extract_chunks
from nltk_setup import download_nltk_data

@asynccontextmanager
async def lifespan(app: FastAPI):
    # download_nltk_data()
    # nltk.download('punkt', quiet=True)
    # nltk.download('stopwords', quiet=True)
    yield

load_dotenv()

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

app = FastAPI(lifespan=lifespan, title="HackRx Query-Retrieval System")
auth_scheme = HTTPBearer()

CACHE: Dict[str, Any] = {}
MAX_CACHE_SIZE = 10
RATE_LIMIT_DELAY = 1.2

SUBMISSION_MODE = os.getenv("SUBMISSION_MODE", "true").lower() == "true"
if SUBMISSION_MODE:
    logging.warning("Application is running in SUBMISSION MODE.")

API_KEYS = [
    os.getenv("COHERE_API_KEY_1"),
    os.getenv("COHERE_API_KEY_2"),
    os.getenv("COHERE_API_KEY_3"),
    os.getenv("COHERE_API_KEY_4"),
    os.getenv("COHERE_API_KEY_5")
]

API_KEYS = [key for key in API_KEYS if key]
if not API_KEYS:
    raise RuntimeError("At least one Cohere API key must be set in the .env file.")

KEY_INDEX_MAP = {key: idx + 1 for idx, key in enumerate(API_KEYS)}

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class Answer(BaseModel):
    question: str
    answer: str
    context: List[str]

class HackRxResponse(BaseModel):
    answers: List[Answer]

class RateLimiter:
    def __init__(self, calls_per_minute: int = 45):
        self.calls_per_minute = calls_per_minute
        self.calls: Dict[str, List[float]] = {key: [] for key in API_KEYS}

    async def wait_for_available_key(self) -> str:
        while True:
            now = time.time()
            for key, call_times in self.calls.items():
                self.calls[key] = [t for t in call_times if now - t < 60]
                if len(self.calls[key]) < self.calls_per_minute:
                    self.calls[key].append(now)
                    key_number = KEY_INDEX_MAP.get(key, "Unknown")
                    logging.info(f"Using Cohere API Key #{key_number}")
                    return key
            await asyncio.sleep(1)

rate_limiter = RateLimiter()

async def safe_cohere_request(method_name: str, *args, max_retries: int = 3, **kwargs):
    for attempt in range(max_retries):
        try:
            api_key = await rate_limiter.wait_for_available_key()
            client = cohere.Client(api_key)
            method = getattr(client, method_name)
            result = await method(*args, **kwargs) if asyncio.iscoroutinefunction(method) else method(*args, **kwargs)
            await asyncio.sleep(RATE_LIMIT_DELAY)
            return result
        except cohere.errors.CohereError as e:
            logging.warning(f"Cohere API error on attempt {attempt+1}: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            await asyncio.sleep(2 ** attempt)

async def process_chunks_in_batches(chunks: List[str], batch_size: int = 30) -> List[List[float]]:
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = await safe_cohere_request(
            "embed",
            texts=batch,
            model="embed-english-v3.0",
            input_type="search_document",
            truncate="END"
        )
        all_embeddings.extend(response.embeddings)
        await asyncio.sleep(RATE_LIMIT_DELAY)
    return all_embeddings

async def get_or_create_document_index(doc_url: str) -> Dict[str, Any]:
    if doc_url in CACHE:
        logging.info(f"Cache HIT for document: {doc_url}")
        return CACHE[doc_url]

    logging.info(f"Cache MISS. Processing document: {doc_url}")

    chunks_generator = read_url_and_extract_chunks(doc_url)
    document_chunks = [chunk['text'].strip() for chunk in chunks_generator if chunk.get('text') and len(chunk['text'].strip()) > 50]

    if not document_chunks:
        raise HTTPException(status_code=400, detail="No valid content extracted from document.")

    embeddings = await process_chunks_in_batches(document_chunks)
    doc_embeddings = np.array(embeddings, dtype='float32')
    embedding_dim = doc_embeddings.shape[1]

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(doc_embeddings)

    if len(CACHE) >= MAX_CACHE_SIZE:
        oldest_key = next(iter(CACHE))
        del CACHE[oldest_key]

    CACHE[doc_url] = {
        "index": index,
        "chunks": document_chunks,
        "embeddings_count": len(embeddings)
    }
    return CACHE[doc_url]

def extract_rerank_text(result) -> Optional[str]:
    doc = getattr(result, 'document', None)
    if isinstance(doc, str):
        return doc
    elif isinstance(doc, dict):
        return doc.get('text')
    return None

async def process_single_question(question: str, doc_data: Dict[str, Any]) -> Answer:
    index = doc_data["index"]
    chunks = doc_data["chunks"]

    response = await safe_cohere_request(
        "embed",
        texts=[question],
        model="embed-english-v3.0",
        input_type="search_query",
        truncate="END"
    )
    question_embedding = np.array(response.embeddings, dtype='float32')
    k = min(15, len(chunks))
    distances, indices = index.search(question_embedding, k)
    retrieved = [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

    try:
        rerank = await safe_cohere_request(
            "rerank",
            model="rerank-english-v3.0",
            query=question,
            documents=retrieved,
            top_n=min(5, len(retrieved)),
            return_documents=True
        )
        context = [extract_rerank_text(r) for r in rerank.results if extract_rerank_text(r)]
    except Exception:
        context = retrieved[:3]

    prompt = f"""Based on the provided context, answer the question accurately.

Context:
{chr(10).join(context)}

Question: {question}

Instructions:
- Answer directly based on the context
- If the context doesn't contain the answer, say so clearly
- Be specific and cite relevant details from the context"""

    answer = await safe_cohere_request(
        "chat",
        model="command-r-plus",
        message=prompt,
        temperature=0.1,
        max_tokens=500
    )

    return Answer(question=question, answer=answer.text.strip(), context=context)

@app.post("/hackrx/run")
async def run_query_retrieval(request: HackRxRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if not request.questions:
        raise HTTPException(status_code=400, detail="No questions provided")

    doc_data = await get_or_create_document_index(str(request.documents))
    tasks = [process_single_question(q, doc_data) for q in request.questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    answers = []
    for i, res in enumerate(results):
        if isinstance(res, Exception):
            answers.append(Answer(question=request.questions[i], answer="Error occurred.", context=[]))
        else:
            answers.append(res)

    return HackRxResponse(answers=answers)

@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(CACHE)}