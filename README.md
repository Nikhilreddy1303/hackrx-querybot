# HackRx 6.0 - Intelligent Query-Retrieval System

This project is an intelligent query-retrieval system built for the HackRx 6.0 hackathon. It processes large documents and answers natural language questions by using a Retrieval-Augmented Generation (RAG) pipeline.

---

## Features

- **Multi-Format Document Support**: Handles PDF, DOCX, and EML files from a URL.
- **Advanced RAG Pipeline**: Implements a full RAG pipeline including document chunking, embedding, and indexing.
- **High-Accuracy Retrieval**: Uses FAISS for initial semantic search and Cohere's `rerank` model to provide the most relevant context to the LLM.
- **Explainable AI**: Returns detailed JSON responses with the source context used to generate each answer.
- **Efficient**: Includes an in-memory cache to reduce API costs and latency on repeated requests for the same document.

---

## Tech Stack

- **Backend**: FastAPI, Uvicorn
- **LLM & Embeddings**: Cohere (Command R+, Embed v3, Rerank v3)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Document Processing**: PyMuPDF, python-docx, NLTK

---

## Setup and Running Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Nikhilreddy1303/hackrx-querybot.git](https://github.com/Nikhilreddy1303/hackrx-querybot.git)
    cd hackrx-querybot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create an environment file:**
    Create a file named `.env` and add your Cohere API key:
    ```
    COHERE_API_KEY="your_api_key_here"
    ```

5.  **Run the server:**
    ```bash
    uvicorn main:app --reload
    ```
The server will be running at `http://127.0.0.1:8000`.

---

## API Usage

You can test the API using `cURL`:

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/hackrx/run](http://127.0.0.1:8000/hackrx/run)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": "[https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf](https://www.adobe.com/support/products/enterprise/knowledgecenter/media/c4611_sample_explain.pdf)",
    "questions": [
        "What is this document about?"
    ]
}'
```