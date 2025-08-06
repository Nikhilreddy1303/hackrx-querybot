import os
import requests
import tempfile
from urllib.parse import urlparse
from mimetypes import guess_extension
import logging

from read_pdf import read_pdf_and_chunk
from read_docx import read_docx_and_chunk
from read_eml import read_eml_and_chunk
from read_excel import read_excel_and_chunk
from read_ppt import read_ppt_and_chunk
from read_image import read_image_and_chunk

# Read the limit from an environment variable, defaulting to 25 MB.
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "100"))
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024 

SUPPORTED_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".eml": "message/rfc822",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

def download_file_from_url(url):
    """
    Securely downloads a file after checking its size and content type.
    """
    try:
        with requests.head(url, timeout=10, allow_redirects=True) as h:
            h.raise_for_status()
            
            content_length = h.headers.get('content-length')
            if content_length and int(content_length) > MAX_FILE_SIZE:
                raise ValueError(f"File size {int(content_length) / 1024**2:.2f} MB exceeds {MAX_FILE_SIZE_MB} MB limit.")
            
            content_type = h.headers.get("Content-Type", "").split(";")[0].strip()
            
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[-1].lower()
        if not ext:
            ext = guess_extension(content_type)
        
        if not ext or ext not in SUPPORTED_CONTENT_TYPES:
            raise ValueError(f"Unsupported file type: {ext or 'unknown'} ({content_type})")
            
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            fd, temp_path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, 'wb') as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
            return temp_path, ext
            
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to download or access URL: {e}")
    except ValueError as e:
        raise e

def read_url_and_extract_chunks(url):
    """
    Detects file type and delegates to the appropriate specialized parser.
    """
    temp_path = None
    try:
        temp_path, ext = download_file_from_url(url)
        filename = os.path.basename(temp_path)

        if ext in [".png", ".jpg", ".jpeg"]:
            yield from read_image_and_chunk(temp_path, source_file=filename)
        elif ext == ".pdf":
            yield from read_pdf_and_chunk(temp_path)
        elif ext == ".docx":
            yield from read_docx_and_chunk(temp_path)
        elif ext == ".eml":
            yield from read_eml_and_chunk(temp_path)
        elif ext == ".xlsx":
            yield from read_excel_and_chunk(temp_path)
        elif ext == ".pptx":
            yield from read_ppt_and_chunk(temp_path)

    except (ValueError, ConnectionError) as e:
        logging.warning(f"Skipping document processing for {url}: {e}")
        return
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)