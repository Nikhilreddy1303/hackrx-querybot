import fitz  # PyMuPDF
import nltk

def sentence_chunker(text: str, source_id: str, sentences_per_chunk: int = 5, overlap: int = 1):
    """
    Splits text into chunks of N sentences with an overlap of M sentences.
    Now includes metadata for the source (e.g., page number).
    """
    try:
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sentence_tokenizer.tokenize(text)
    except Exception:
        sentences = text.split('. ')

    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk - overlap):
        chunk_text = " ".join(sentences[i:i + sentences_per_chunk])
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text.strip(),
                "source": source_id 
            })
    return chunks

def read_pdf_and_chunk(pdf_path: str):
    """
    Reads a PDF, extracts text page by page, and chunks it with page number metadata.
    """
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, 1):
            text = page.get_text("text", sort=True)
            if text.strip():
                yield from sentence_chunker(text, source_id=f"Page {i}")