import docx
from docx.table import Table
from docx.text.paragraph import Paragraph
from nltk.tokenize import sent_tokenize
from typing import Generator, Tuple


def extract_docx_text_with_tables(docx_path: str, logical_page_size: int = 1000) -> Generator[Tuple[int, str], None, None]:
    """
    Generator to yield logical page chunks from a DOCX file.
    Each chunk includes both paragraph and table text in natural document order.
    """
    doc = docx.Document(docx_path)
    buffer = []
    char_count = 0
    logical_page_number = 1

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

        if not text:
            continue

        buffer.append(text)
        char_count += len(text)

        if char_count >= logical_page_size:
            yield logical_page_number, "\n".join(buffer).strip()
            logical_page_number += 1
            buffer = []
            char_count = 0

    if buffer:
        yield logical_page_number, "\n".join(buffer).strip()


def stitch_docx_chunks_with_sentence_overlap(
    chunks: Generator[Tuple[int, str], None, None], overlap_chars: int = 250
) -> Generator[dict, None, None]:
    """
    Generator to stitch DOCX logical chunks using sentence-aware overlap.
    Avoids duplicating sentence fragments across chunks.
    """
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
        curr_overlap = curr_text[:overlap_chars]

        prev_sentences = sent_tokenize(prev_overlap)
        curr_sentences = sent_tokenize(curr_overlap)

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


# Example usage:
if __name__ == "__main__":
    import sys
    docx_path = sys.argv[1] if len(sys.argv) > 1 else "sample.docx"

    print("\n=== Stitched DOCX Chunks ===\n")
    chunks = extract_docx_text_with_tables(docx_path)
    stitched_chunks = stitch_docx_chunks_with_sentence_overlap(chunks)

    for chunk in stitched_chunks:
        print(f"--- Pages {chunk['from_page']} to {chunk['to_page']} ---\n")
        print(chunk["text"])
        print("\n" + "=" * 80 + "\n")








# from docx import Document  #To process the docx files

# def extract_text_from_docx(file_path):
#     try:
#         doc = Document(file_path)   #Opens the file
#         full_text = []

#         for para in doc.paragraphs: #Iterates over each paragraph
#             full_text.append(para.text.strip())

#         return "\n".join([p for p in full_text if p])  # remove empty lines and joins the text

#     except Exception as e:
#         print(f"Error reading DOCX: {e}")
#         return ""

# # Example usage
# if __name__ == "__main__":
#     docx_path = "sample.docx"  # Testing
#     extracted_text = extract_text_from_docx(docx_path)

#     print("\n=== Extracted DOCX Text ===\n")
#     print(extracted_text)