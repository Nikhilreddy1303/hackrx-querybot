import os
import faiss
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import json
import asyncio
import hashlib
import logging

from read_url import read_url_and_extract_chunks
from nltk_setup import download_nltk_data

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file.")
genai.configure(api_key=GEMINI_API_KEY)

SAVE_DIR = "precomputed_indices"
os.makedirs(SAVE_DIR, exist_ok=True)

async def preprocess_document(doc_url: str):
    """
    Processes a single document and saves its index and chunks (with metadata) to disk.
    """
    logging.info(f"Processing: {doc_url}")
    url_hash = hashlib.md5(doc_url.encode()).hexdigest()
    logging.info(f"--> URL Hash: {url_hash}")
    
    document_chunks_with_meta = list(read_url_and_extract_chunks(doc_url))

    if not document_chunks_with_meta:
        logging.warning(f"--> No content found in {doc_url}. Skipping.")
        return

    document_texts = [c['text'] for c in document_chunks_with_meta]
    logging.info(f"--> Found {len(document_texts)} chunks. Embedding...")

    embed_model = "models/text-embedding-004"
    try:
        result = await genai.embed_content_async(
            model=embed_model,
            content=document_texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
    except Exception as e:
        logging.error(f"--> Embedding failed for {doc_url}: {e}")
        return
    
    doc_embeddings = np.array(result['embedding'], dtype='float32')
    
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    index_path = os.path.join(SAVE_DIR, f"{url_hash}.index")
    faiss.write_index(index, index_path)
    logging.info(f"--> Saved FAISS index to {index_path}")

    # Save chunks with their metadata
    chunks_path = os.path.join(SAVE_DIR, f"{url_hash}.json")
    with open(chunks_path, 'w') as f:
        json.dump(document_chunks_with_meta, f)
    logging.info(f"--> Saved chunks with metadata to {chunks_path}")

async def main():
    download_nltk_data()
    
    unlocked_document_urls = [
        "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
        "https://hackrx.blob.core.windows.net/assets/Happy%20Family%20Floater%20-%202024%20OICHLIP25046V062425%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A24%3A30Z&se=2026-08-01T17%3A24%3A00Z&sr=b&sp=r&sig=VNMTTQUjdXGYb2F4Di4P0zNvmM2rTBoEHr%2BnkUXIqpQ%3D",
        "https://hackrx.blob.core.windows.net/assets/UNI%20GROUP%20HEALTH%20INSURANCE%20POLICY%20-%20UIIHLGP26043V022526%201.pdf?sv=2023-01-03&spr=https&st=2025-07-31T17%3A06%3A03Z&se=2026-08-01T17%3A06%3A00Z&sr=b&sp=r&sig=wLlooaThgRx91i2z4WaeggT0qnuUUEzIUKj42GsvMfg%3D",
        "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D",
        "https://hackrx.blob.core.windows.net/assets/indian_constitution.pdf?sv=2023-01-03&st=2025-07-28T06%3A42%3A00Z&se=2026-11-29T06%3A42%3A00Z&sr=b&sp=r&sig=5Gs%2FOXqP3zY00lgciu4BZjDV5QjTDIx7fgnfdz6Pu24%3D",
        "https://hackrx.blob.core.windows.net/assets/Family%20Medicare%20Policy%20(UIN-%20UIIHLIP22070V042122)%201.pdf?sv=2023-01-03&st=2025-07-22T10%3A17%3A39Z&se=2025-08-23T10%3A17%3A00Z&sr=b&sp=r&sig=dA7BEMIZg3WcePcckBOb4QjfxK%2B4rIfxBs2%2F%2BNwoPjQ%3D",
        "https://hackrx.blob.core.windows.net/assets/Super_Splendor_(Feb_2023).pdf?sv=2023-01-03&st=2025-07-21T08%3A10%3A00Z&se=2025-09-22T08%3A10%3A00Z&sr=b&sp=r&sig=vhHrl63YtrEOCsAy%2BpVKr20b3ZUo5HMz1lF9%2BJh6LQ0%3D",
        "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    ]
    
    tasks = [preprocess_document(url) for url in unlocked_document_urls]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())