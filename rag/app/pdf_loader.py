"""
PDF loading utilities
Author: Ravi

Loads PDFs from the data directory and returns a list of documents with metadata.
"""
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


def load_pdf(path: Path) -> List[Dict]:
    """Load a PDF and return a list of pages as documents.

    Each document is a dict:
    {
        "id": "filename::page_no",
        "text": "...",
        "metadata": {"source": filename, "page_number": n}
    }
    """
    docs = []
    reader = PdfReader(str(path))
    filename = path.name
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        docs.append({
            "id": f"{filename}::p{i}",
            "text": text.strip(),
            "metadata": {"source": filename, "page_number": i},
        })
    return docs


def load_all_pdfs(folder: Path) -> List[Dict]:
    """Load all pdf files from folder and return flattened list of page-documents."""
    docs = []
    for path in sorted(folder.glob("*.pdf")):
        docs.extend(load_pdf(path))
    return docs

