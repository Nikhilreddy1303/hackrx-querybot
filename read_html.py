import logging
from bs4 import BeautifulSoup
import re
import nltk

# This is the same sentence chunker used in your other parsers.
def sentence_chunker(text: str, source_id: str, sentences_per_chunk: int = 10, overlap: int = 2):
    """
    Splits text into larger chunks of N sentences with an overlap, suitable for web content.
    """
    try:
        # Ensure the NLTK data is available
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sentence_tokenizer.tokenize(text)
    except Exception:
        # Fallback for languages where Punkt might not be trained
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

def read_html_and_chunk(html_path: str):
    """
    Reads an HTML file, cleans it, extracts the main text content,
    and splits it into semantic chunks.
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml')

        # 1. Decompose (remove) all irrelevant tags like scripts, styles, navs, and footers.
        for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
            element.decompose()
        
        # 2. Get the text from the main body.
        # This will be much cleaner now that the noisy tags are gone.
        body_text = soup.body.get_text(separator=' ', strip=True)

        # 3. Clean up excessive whitespace and newlines.
        clean_text = re.sub(r'\s+', ' ', body_text).strip()

        if clean_text:
            logging.info(f"Successfully extracted and cleaned text from {html_path}")
            yield from sentence_chunker(clean_text, source_id="HTML Content")
        else:
            logging.warning(f"Could not find any processable text in {html_path}")

    except Exception as e:
        logging.error(f"Failed to process HTML file {html_path}: {e}")
        return