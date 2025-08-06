import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
import nltk

def sentence_chunker(text: str, source_id: str, sentences_per_chunk: int = 5, overlap: int = 1):
    """
    Splits text into chunks of N sentences with an overlap of M sentences.
    Now includes metadata for the source (e.g., block number).
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

def read_docx_and_chunk(docx_path: str):
    """
    Extracts full text from a DOCX including tables and chunks it by sentences.
    """
    doc = docx.Document(docx_path)
    block_num = 0
    for element in doc.iter_inner_content():
        text = ""
        if isinstance(element, Paragraph):
            text = element.text.strip()
        elif isinstance(element, Table):
            table_rows = []
            for row in element.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    table_rows.append(" | ".join(row_text))
            if table_rows:
                text = "\n".join(table_rows)
        
        if text:
            block_num += 1
            yield from sentence_chunker(text, source_id=f"Block {block_num}")