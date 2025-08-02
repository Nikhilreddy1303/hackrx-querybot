import os
import requests
import tempfile
from urllib.parse import urlparse
from mimetypes import guess_extension
from bs4 import BeautifulSoup

from read_pdf import read_pdf_pages, stitch_pages_with_sentence_overlap
from read_docx import extract_docx_text_with_tables, stitch_docx_chunks_with_sentence_overlap
from read_eml import extract_eml_text, split_eml_text_into_chunks, stitch_eml_chunks_with_sentence_overlap


def download_file_from_url(url):
    """
    Downloads file from URL to a secure temporary file.
    Returns: (temp_path, guessed_extension)
    """
    # IMPROVEMENT: Add a timeout to prevent hanging indefinitely
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    ext = guess_extension(content_type.split(";")[0].strip()) or ""

    parsed = urlparse(url)
    if not ext:
        ext = os.path.splitext(parsed.path)[-1].lower()

    fd, temp_path = tempfile.mkstemp(suffix=ext or ".tmp")
    with os.fdopen(fd, 'wb') as tmp:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                tmp.write(chunk)

    return temp_path, ext.lower()

def extract_html_text(url):
    """
    Extracts visible text from an HTML page.
    """
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "header", "footer", "nav", "form"]):
        tag.decompose()

    text_parts = [p.get_text(separator=" ", strip=True) for p in soup.find_all(["p", "li", "div", "span", "h1", "h2", "h3"])]
    full_text = "\n".join(filter(None, text_parts))
    return full_text


def split_html_text_into_chunks(html_text, logical_page_size=1000):
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(html_text)
    buffer = []
    char_count = 0
    page_number = 1

    for sentence in sentences:
        buffer.append(sentence)
        char_count += len(sentence)
        if char_count >= logical_page_size:
            yield page_number, " ".join(buffer).strip()
            page_number += 1
            buffer = []
            char_count = 0

    if buffer:
        yield page_number, " ".join(buffer).strip()


def stitch_html_chunks_with_sentence_overlap(chunks, overlap_chars=250):
    from nltk.tokenize import sent_tokenize

    prev_text = None
    prev_page_num = None

    chunk_iter = iter(chunks)
    try:
        prev_page_num, prev_text = next(chunk_iter)
        yield {
            "from_page": prev_page_num,
            "to_page": prev_page_num,
            "text": prev_text.strip()
        }
    except StopIteration:
        return

    for curr_page_num, curr_text in chunk_iter:
        prev_overlap = prev_text[-overlap_chars:]
        prev_sentences = sent_tokenize(prev_overlap)
        prev_snippet = prev_sentences[-1] if prev_sentences else ''

        if prev_snippet.strip() and prev_snippet.strip() in curr_text:
            stitched = curr_text.strip()
        else:
            stitched = f"{prev_snippet} {curr_text}".strip()

        yield {
            "from_page": prev_page_num,
            "to_page": curr_page_num,
            "text": stitched
        }

        prev_text = curr_text
        prev_page_num = curr_page_num


def read_url_and_extract_chunks(url):
    """
    Detects the file type and delegates processing logic.
    Supports PDF, DOCX, EML, and HTML.
    Yields stitched text chunks with metadata.
    """
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[-1].lower()

    if ext in [".html", ".htm"]:
        html_text = extract_html_text(url)
        chunks = split_html_text_into_chunks(html_text)
        yield from stitch_html_chunks_with_sentence_overlap(chunks)
        return

    # Else: download the file
    temp_path = None
    try:
        temp_path, ext = download_file_from_url(url)

        if ext == ".pdf":
            pages = read_pdf_pages(temp_path)
            yield from stitch_pages_with_sentence_overlap(pages)
        elif ext == ".docx":
            chunks = extract_docx_text_with_tables(temp_path)
            yield from stitch_docx_chunks_with_sentence_overlap(chunks)
        elif ext == ".eml":
            eml_text = extract_eml_text(temp_path)
            chunks = split_eml_text_into_chunks(eml_text)
            yield from stitch_eml_chunks_with_sentence_overlap(chunks)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# === DEMO ===
if __name__ == "__main__":
    test_url = input("Enter a file or HTML URL: ").strip()

    try:
        for chunk in read_url_and_extract_chunks(test_url):
            print(f"\n--- Pages {chunk['from_page']} to {chunk['to_page']} ---\n")
            print(chunk['text'])
            print("\n" + "=" * 80 + "\n")
    except Exception as e:
        print("Error:", e)








# import requests
# import tempfile
# import mimetypes
# import os

# from read_pdf import extract_text_from_pdf
# from read_docx import extract_text_from_docx
# from read_eml import extract_text_from_eml

# def extract_text_from_url(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()  # Raise an error for bad responses (404, etc.)

#         # Try to get file extension from the URL itself
#         path_part = url.split('?')[0]
#         file_ext = os.path.splitext(path_part)[-1].lower()

#         # If URL doesn't contain extension, fallback using the response's Content-Type
#         if not file_ext:
#             content_type = response.headers.get("Content-Type", "")
#             file_ext = mimetypes.guess_extension(content_type) or '.eml'

#         file_ext = file_ext.replace('.', '')  # Clean up extension for use

#         # Create a temporary file to save the downloaded content
#         with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmp_file:
#             tmp_file.write(response.content)
#             tmp_path = tmp_file.name  # Path to the saved temp file

#         # Call the appropriate extraction function based on file type
#         if file_ext == 'pdf':
#             return extract_text_from_pdf(tmp_path)
#         elif file_ext == 'docx':
#             return extract_text_from_docx(tmp_path)
#         elif file_ext == 'eml':
#             return extract_text_from_eml(tmp_path)
#         else:
#             return f"Unsupported file type: {file_ext}"

#     except Exception as e:
#         return f"Error fetching or processing file: {e}"


# # 🧪 Example usage
# if __name__ == "__main__":
#     test_url = "https://gist.githubusercontent.com/billsinc/967795/raw"  # Testing
#     text = extract_text_from_url(test_url)
#     print("\n=== Extracted URL File Text ===\n")
#     print(text)