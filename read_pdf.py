# import os
# import fitz  # PyMuPDF
# import nltk
# from nltk.tokenize import PunktSentenceTokenizer

# # Load tokenizer once to avoid repeated I/O
# tokenizer_path = os.path.expanduser('~/nltk_data/tokenizers/punkt/english.pickle')
# if not os.path.exists(tokenizer_path):
#     nltk.download('punkt')
# with open(tokenizer_path, 'rb') as f:
#     sentence_tokenizer = PunktSentenceTokenizer(f.read())


# def read_pdf_pages(pdf_path):
#     """
#     Generator to yield (page_number, page_text) for each page in the PDF.
#     """
#     with fitz.open(pdf_path) as doc:
#         doc = fitz.open(pdf_path)
#         for i, page in enumerate(doc,1):
#             yield i, page.get_text().strip() or ""  # Ensure non-None text


# def stitch_pages_with_sentence_overlap(pages, overlap_chars=250):
#     """
#     Stitch pages together using clean sentence overlaps without duplication.
#     Each chunk contains context from the end of the previous page.
#     """
#     prev_text = None
#     prev_page_num = None

#     for curr_page_num, curr_text in pages:
#         if prev_text is None:
#             # First page, yield as is
#             yield {
#                 "from_page": curr_page_num,
#                 "to_page": curr_page_num,
#                 "text": curr_text.strip()
#             }
#         else:
#             # Extract overlap from prev_text and curr_text
#             prev_overlap = prev_text[-overlap_chars:]
#             curr_overlap = curr_text[:overlap_chars]

#             prev_sentences = sentence_tokenizer.tokenize(prev_overlap)
#             curr_sentences = sentence_tokenizer.tokenize(curr_overlap)

#             prev_snippet = prev_sentences[-1] if prev_sentences else ''
#             curr_snippet = curr_sentences[0] if curr_sentences else ''

#             # Avoid redundant duplication if prev_snippet already in curr_text
#             stitched = f"{prev_snippet} {curr_text}".strip()
#             if prev_snippet.strip() in curr_text:
#                 stitched = curr_text.strip()

#             yield {
#                 "from_page": prev_page_num,
#                 "to_page": curr_page_num,
#                 "text": stitched
#             }

#         prev_text = curr_text
#         prev_page_num = curr_page_num


# # Example usage
# if __name__ == "__main__":
#     pdf_path = "sample.pdf"  # Replace with your actual file
#     chunks = stitch_pages_with_sentence_overlap(read_pdf_pages(pdf_path))

#     print("\n=== Stitched PDF Chunks ===\n")
#     for chunk in chunks:
#         print(f"\n--- Pages {chunk['from_page']} to {chunk['to_page']} ---\n")
#         print(chunk['text'])






# import fitz  #To process pdf file

# def extract_text_from_pdf(file_path):
#     try:
#         doc = fitz.open(file_path)  #opens the file
#         print(f"Opened PDF: {file_path} | Total pages: {len(doc)}")

#         full_text = ""
#         for page_num, page in enumerate(doc, start=1):
#             text = page.get_text()
#             full_text += f"\n--- Page {page_num} ---\n{text.strip()}\n"

#         doc.close()
#         return full_text.strip() #Removes extra empty lines if any and returns the text

#     except Exception as e:
#         print(f"Error reading PDF: {e}")
#         return ""

# # Example usage
# if __name__ == "__main__":
#     pdf_path = "sample.pdf"  #testind
#     extracted_text = extract_text_from_pdf(pdf_path)
    
#     print("\n=== Extracted PDF Text ===\n")
#     print(extracted_text)



# import os
# import fitz  # PyMuPDF
# import nltk
# from nltk.tokenize import PunktSentenceTokenizer
# import nltk.data # Import nltk.data

# # Load tokenizer once to avoid repeated I/O
# # Use nltk.data.load() for robust loading of Punkt tokenizer
# try:
#     # This will load the tokenizer from nltk_data paths
#     sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# except LookupError:
#     # If the tokenizer is not found, download it
#     nltk.download('punkt')
#     sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# def read_pdf_pages(pdf_path):
#     """
#     Generator to yield (page_number, page_text) for each page in the PDF.
#     """
#     with fitz.open(pdf_path) as doc:
#         # doc = fitz.open(pdf_path) # This line is redundant if already opened with 'with' statement
#         for i, page in enumerate(doc,1):
#             yield i, page.get_text().strip() or ""  # Ensure non-None text


# def stitch_pages_with_sentence_overlap(pages, overlap_chars=250):
#     """
#     Stitch pages together using clean sentence overlaps without duplication.
#     Each chunk contains context from the end of the previous page.
#     """
#     prev_text = None
#     prev_page_num = None

#     for curr_page_num, curr_text in pages:
#         if prev_text is None:
#             # First page, yield as is
#             yield {
#                 "from_page": curr_page_num,
#                 "to_page": curr_page_num,
#                 "text": curr_text.strip()
#             }
#         else:
#             # Extract overlap from prev_text and curr_text
#             prev_overlap = prev_text[-overlap_chars:]
#             curr_overlap = curr_text[:overlap_chars]

#             # Use the globally loaded sentence_tokenizer
#             prev_sentences = sentence_tokenizer.tokenize(prev_overlap)
#             curr_sentences = sentence_tokenizer.tokenize(curr_overlap)

#             prev_snippet = prev_sentences[-1] if prev_sentences else ''
#             curr_snippet = curr_sentences[0] if curr_sentences else ''

#             # Avoid redundant duplication if prev_snippet already in curr_text
#             stitched = f"{prev_snippet} {curr_text}".strip()
#             if prev_snippet.strip() in curr_text:
#                 stitched = curr_text.strip()

#             yield {
#                 "from_page": prev_page_num,
#                 "to_page": curr_page_num,
#                 "text": stitched
#             }

#         prev_text = curr_text
#         prev_page_num = curr_page_num


# # Example usage
# if __name__ == "__main__":
#     pdf_path = "sample.pdf"  # Replace with your actual file
#     chunks = stitch_pages_with_sentence_overlap(read_pdf_pages(pdf_path))

#     print("\n=== Stitched PDF Chunks ===\n")
#     for chunk in chunks:
#         print(f"\n--- Pages {chunk['from_page']} to {chunk['to_page']} ---\n")
#         print(chunk['text'])


import fitz  # PyMuPDF
import nltk

def read_pdf_pages(pdf_path):
    """
    Generator to yield (page_number, page_text) for each page in the PDF.
    """
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc,1):
            yield i, page.get_text().strip() or ""


def stitch_pages_with_sentence_overlap(pages, overlap_chars=250):
    """
    Stitch pages together using clean sentence overlaps without duplication.
    """
    # FIX: Load the tokenizer inside the function to ensure NLTK data is ready.
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    
    prev_text = None
    prev_page_num = None

    for curr_page_num, curr_text in pages:
        if prev_text is None:
            yield {
                "from_page": curr_page_num,
                "to_page": curr_page_num,
                "text": curr_text.strip()
            }
        else:
            prev_overlap = prev_text[-overlap_chars:]
            
            prev_sentences = sentence_tokenizer.tokenize(prev_overlap)
            
            prev_snippet = prev_sentences[-1] if prev_sentences else ''

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