# import PyPDF2

# def load_pdf(file):
#     text = ""
#     reader = PyPDF2.PdfReader(file)
#     for page in reader.pages:
#         text += page.extract_text()
#     return text

# def split_text(text, chunk_size=500, chunk_overlap=50):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start += chunk_size - chunk_overlap
#     return chunks

import PyPDF2
from typing import List


def load_pdf(file) -> str:
    """
    Extract text from all pages of a PDF file.

    Args:
        file: A file-like object containing the PDF.

    Returns:
        str: The combined text from all pages of the PDF.
    """
    extracted_text: str = ""
    reader = PyPDF2.PdfReader(file)
    
    # Extract text from each page in the PDF
    for page in reader.pages:
        extracted_text += page.extract_text()
    
    return extracted_text


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """
    Split a large block of text into overlapping chunks.

    Args:
        text (str): The full text to be split.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 500.
        chunk_overlap (int, optional): The number of overlapping characters between consecutive chunks. Defaults to 50.

    Returns:
        List[str]: A list of text chunks.
    """
    text_chunks: List[str] = []
    start: int = 0

    # Slide over the text and create overlapping chunks
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        text_chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return text_chunks
