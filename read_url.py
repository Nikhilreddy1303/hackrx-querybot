import os
import requests
import tempfile
from urllib.parse import urlparse
from mimetypes import guess_extension

from read_pdf import read_pdf_and_chunk
from read_docx import read_docx_and_chunk
from read_eml import read_eml_and_chunk

def download_file_from_url(url):
    """
    Downloads file from URL to a secure temporary file.
    """
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    ext = guess_extension(content_type.split(";")[0].strip()) or ""

    if "drive.google.com" in url:
        ext = ".pdf"

    parsed = urlparse(url)
    if not ext:
        ext = os.path.splitext(parsed.path)[-1].lower()

    fd, temp_path = tempfile.mkstemp(suffix=ext or ".tmp")
    with os.fdopen(fd, 'wb') as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)
    return temp_path, ext.lower()

def read_url_and_extract_chunks(url):
    """
    Detects the file type and delegates to the appropriate sentence chunker.
    """
    temp_path = None
    try:
        temp_path, ext = download_file_from_url(url)

        if ext == ".pdf":
            yield from read_pdf_and_chunk(temp_path)
        elif ext == ".docx":
            yield from read_docx_and_chunk(temp_path)
        elif ext == ".eml":
            yield from read_eml_and_chunk(temp_path)
        else:
            # Fallback for other text-based files
            with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            from read_pdf import sentence_chunker # Re-use a chunker
            yield from sentence_chunker(text, source_id="File")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)