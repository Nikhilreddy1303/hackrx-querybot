import email
from email import policy
from email.parser import BytesParser
import nltk
import html2text

def sentence_chunker(text: str, source_id: str, sentences_per_chunk: int = 5, overlap: int = 1):
    """
    Splits text into chunks of N sentences with an overlap of M sentences.
    Now includes metadata for the source (e.g., email section).
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

def read_eml_and_chunk(eml_path: str):
    """
    Extracts full text from an .eml file and chunks it by sentences.
    """
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    header_lines = []
    for h in ['Subject', 'From', 'To', 'Cc', 'Date']:
        if msg[h]:
            header_lines.append(f"{h}: {msg[h]}")
    header_text = "\n".join(header_lines)
    if header_text:
        yield from sentence_chunker(header_text, source_id="Header")

    plain_parts = []
    html_parts = []
    for part in msg.walk():
        if part.is_attachment():
            continue
        try:
            content = part.get_payload(decode=True).decode(part.get_content_charset('utf-8'))
            if part.get_content_type() == 'text/plain':
                plain_parts.append(content.strip())
            elif part.get_content_type() == 'text/html':
                html_parts.append(content.strip())
        except:
            continue

    body_text = ""
    if plain_parts:
        body_text = "\n".join(plain_parts)
    elif html_parts:
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        body_text = "\n".join([converter.handle(html) for html in html_parts])
        
    if body_text.strip():
        yield from sentence_chunker(body_text.strip(), source_id="Body")