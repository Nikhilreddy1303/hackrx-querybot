# eml_reader.py

import os
import email
from email import policy
from email.parser import BytesParser
from nltk.tokenize import sent_tokenize

try:
    import html2text
except ImportError:
    html2text = None


def extract_eml_text(eml_path):
    """
    Extracts structured text from .eml file, including headers and body.
    Falls back to HTML if plain text is not available.
    """
    with open(eml_path, 'rb') as f:
        msg = BytesParser(policy=policy.default).parse(f)

    # Extract key headers
    header_lines = []
    for h in ['Subject', 'From', 'To', 'Cc', 'Date']:
        if msg[h]:
            header_lines.append(f"{h}: {msg[h]}")

    # Extract plain and/or HTML parts
    plain_parts = []
    html_parts = []
    for part in msg.walk():
        content_type = part.get_content_type()
        content_disposition = str(part.get("Content-Disposition", ""))
        if 'attachment' in content_disposition:
            continue

        try:
            content = part.get_content()
        except Exception:
            continue

        if content_type == 'text/plain':
            plain_parts.append(content.strip())
        elif content_type == 'text/html':
            html_parts.append(content.strip())

    # Fallback to HTML to plain conversion if needed
    body_text = ""
    if plain_parts:
        body_text = "\n".join(plain_parts)
    elif html_parts and html2text:
        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converted = [converter.handle(html) for html in html_parts]
        body_text = "\n".join(converted)
    elif html_parts:
        body_text = "\n".join(html_parts)  # If html2text not installed, return raw

    return "\n".join(header_lines) + "\n\n" + body_text.strip()


def split_eml_text_into_chunks(eml_text, logical_page_size=1000):
    """
    Generator that yields (page_number, chunk_text) based on sentence boundaries.
    """
    sentences = sent_tokenize(eml_text)
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


def stitch_eml_chunks_with_sentence_overlap(chunks, overlap_chars=250):
    """
    Generator that stitches chunks with sentence-level continuity.
    Avoids redundancy using substring heuristics.
    """
    prev_text = None
    prev_page_num = None

    chunk_gen = iter(chunks)
    try:
        prev_page_num, prev_text = next(chunk_gen)
        yield {
            "from_page": prev_page_num,
            "to_page": prev_page_num,
            "text": prev_text.strip()
        }
    except StopIteration:
        return

    for curr_page_num, curr_text in chunk_gen:
        prev_overlap = prev_text[-overlap_chars:]
        prev_sentences = sent_tokenize(prev_overlap)
        prev_snippet = prev_sentences[-1] if prev_sentences else ''

        # Stitch logic with duplication avoidance
        stitched = f"{prev_snippet} {curr_text}".strip()
        if prev_snippet.strip() and prev_snippet.strip() in curr_text:
            stitched = curr_text.strip()

        yield {
            "from_page": prev_page_num,
            "to_page": curr_page_num,
            "text": stitched
        }

        prev_text = curr_text
        prev_page_num = curr_page_num


# === DEMO ===
if __name__ == "__main__":
    eml_path = "sample.eml"  # Replace with your path

    eml_text = extract_eml_text(eml_path)
    eml_chunks = split_eml_text_into_chunks(eml_text)

    print("=== Stitched EML Chunks ===\n")
    for chunk in stitch_eml_chunks_with_sentence_overlap(eml_chunks):
        print(f"--- Pages {chunk['from_page']} to {chunk['to_page']} ---\n")
        print(chunk['text'])
        print("\n" + "=" * 80 + "\n")








# import email
# from email import policy
# from email.parser import BytesParser
# from bs4 import BeautifulSoup

# def extract_text_from_eml(file_path):
#     try:
#         with open(file_path, 'rb') as f:
#             msg = BytesParser(policy=policy.default).parse(f)

#         subject = msg['subject']
#         sender = msg['from']
#         recipient = msg['to']
#         date = msg['date']
        
#         body = None

#         if msg.is_multipart():
#             for part in msg.walk():
#                 content_type = part.get_content_type()
#                 if content_type == "text/plain":
#                     body = part.get_content()
#                     break
#                 elif content_type == "text/html" and body is None:
#                     html = part.get_content()
#                     soup = BeautifulSoup(html, 'html.parser')
#                     body = soup.get_text()
#         else:
#             content_type = msg.get_content_type()
#             if content_type == "text/plain":
#                 body = msg.get_content()
#             elif content_type == "text/html":
#                 html = msg.get_content()
#                 soup = BeautifulSoup(html, 'html.parser')
#                 body = soup.get_text()

#         return f"Subject: {subject}\nFrom: {sender}\nTo: {recipient}\nDate: {date}\n\nBody:\n{body.strip() if body else '[No body found]'}"

#     except Exception as e:
#         return f"Error reading EML: {e}"

# #Example useage
# if __name__ == "__main__":
#     eml_path = "sample.eml"  # Testing
#     extracted_text = extract_text_from_eml(eml_path)

#     print("\n=== Extracted EML Text ===\n")
#     print(extracted_text)