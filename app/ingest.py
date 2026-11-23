"""Data pipeline: PDF -> pages -> chunks -> embeddings in a vector DB (ChromaDB)."""
import hashlib
import sys
import tempfile
from pathlib import Path

try:  # newer sqlite on Linux hosts (needed by chromadb on some cloud images)
    import pysqlite3  # noqa: F401

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DB_PATH = str(Path(tempfile.gettempdir()) / "esglens_chroma")


def read_pdf_pages(pdf_path_or_file) -> list[tuple[int, str]]:
    reader = PdfReader(pdf_path_or_file)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ")
        text = " ".join(text.split())  # clean whitespace
        if text:
            pages.append((i, text))
    return pages


def chunk_pages(pages, chunk_size=900, overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for page_no, text in pages:
        for part in splitter.split_text(text):
            chunks.append({"text": part, "page": page_no})
    return chunks


def doc_id_for(file_bytes: bytes) -> str:
    return "doc_" + hashlib.md5(file_bytes).hexdigest()[:12]


def build_collection(pdf_file, file_bytes: bytes, embedding_function=None):
    """Ingest a PDF (idempotent per file). Returns (collection, n_chunks)."""
    client = chromadb.PersistentClient(path=DB_PATH)
    name = doc_id_for(file_bytes)
    kwargs = {"embedding_function": embedding_function} if embedding_function else {}
    col = client.get_or_create_collection(name=name, **kwargs)
    if col.count() == 0:
        chunks = chunk_pages(read_pdf_pages(pdf_file))
        if not chunks:
            raise ValueError("No extractable text found (scanned PDF? try an OCR'd copy).")
        for i in range(0, len(chunks), 100):
            batch = chunks[i:i + 100]
            col.add(
                ids=[f"{name}_{i + j}" for j in range(len(batch))],
                documents=[c["text"] for c in batch],
                metadatas=[{"page": c["page"]} for c in batch],
            )
    return col, col.count()


def all_chunks(col) -> list[dict]:
    res = col.get(include=["documents", "metadatas"])
    return [{"text": d, "page": m["page"]} for d, m in zip(res["documents"], res["metadatas"])]
