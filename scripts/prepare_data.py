from pypdf import PdfReader
import re
import nltk
from nltk.tokenize import sent_tokenize
import nltk
import mlflow
import tempfile
import pickle
from dotenv import load_dotenv

nltk.download('punkt_tab')
#open the PDF file and extract text from each page, storing it in a list of dictionaries with page numbers and text content
def pdf_to_pages(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page_number": i + 1, "text": text})
    return pages

#clean the text by replacing non-breaking spaces with regular spaces and collapsing multiple whitespace 
#characters into a single space, while also stripping leading and trailing whitespace
def clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")  # non‑breaking spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text: str, max_words=200, overlap_words=50):
    sentences = sent_tokenize(text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
            # start new chunk with overlap
            overlap = current[-overlap_words:] if overlap_words and len(current) > overlap_words else current
            current = overlap[:] + words
            current_len = len(current)
        else:
            current.append(sent)
            current_len += len(words)

    if current:
        chunks.append(" ".join(current))


    return chunks


def save_chunks_as_pickle( pdf_path: str, output_path: 'vector_db.pkl'):
    pages= pdf_to_pages(pdf_path)

    docs = []
    for page in pages:
        cleaned_text = clean_text(page["text"])
        if not cleaned_text:
            print(f"Page {page['page_number']} is empty after cleaning.")
            continue

        for chunk in chunk_text(cleaned_text):
            docs.append({"content": chunk, 'metadata': {"page_number": page['page_number'],
                                                        "source_name": pdf_path}})
    with open(output_path, "wb") as f:
        pickle.dump(docs, f)
    print(f"Saved {len(docs)} chunks to {output_path}")
    return output_path    

if __name__ == "__main__":

    # Example usage
    path = "docs-ai4eosc-eu-en-latest.pdf"
    output_path = "vector_db.pkl"
    save_chunks_as_pickle(pdf_path=path, output_path=output_path)