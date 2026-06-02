"""
Embedding and ChromaDB storage.

This module owns the "chunks -> vectors -> vector database" step.
"""

import hashlib
from pathlib import Path
from typing import Iterable

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_PREFIX, EMBEDDING_MODEL_NAME


def create_document_id(pdf_path: Path) -> str:
    """Create a stable ID from the PDF path and contents."""
    hasher = hashlib.sha256()

    # Include path plus bytes so copied PDFs can be indexed separately if needed.
    hasher.update(str(pdf_path.resolve()).encode("utf-8"))

    with pdf_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(block)

    return hasher.hexdigest()[:16]


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load the local SentenceTransformers embedding model."""
    return SentenceTransformer(model_name)


def embed_texts(model: SentenceTransformer, texts: Iterable[str]) -> list[list[float]]:
    """Turn text strings into vector embeddings."""
    text_list = list(texts)

    embeddings = model.encode(
        text_list,
        normalize_embeddings=True,
        show_progress_bar=len(text_list) > 1,
    )

    # Chroma stores JSON-like values, so convert numpy arrays to plain lists.
    return embeddings.tolist()


def get_collection(pdf_path: Path, chroma_dir: Path = CHROMA_DIR):
    """Open the Chroma collection for one PDF."""
    document_id = create_document_id(pdf_path)

    # PersistentClient writes vectors to disk, so indexing survives restarts.
    client = chromadb.PersistentClient(path=str(chroma_dir))

    return client.get_or_create_collection(name=f"{COLLECTION_PREFIX}_{document_id}")


def index_chunks(pdf_path: Path, chunks: list[str], model: SentenceTransformer):
    """Embed chunks and store them in Chroma if the PDF is not already indexed."""
    collection = get_collection(pdf_path)

    # Reuse existing vectors so repeated runs start fast and do not duplicate data.
    if collection.count() > 0:
        return collection

    document_id = create_document_id(pdf_path)
    embeddings = embed_texts(model, chunks)

    ids = [f"{document_id}_chunk_{index}" for index in range(len(chunks))]
    metadatas = [
        {
            "source": str(pdf_path),
            "chunk_index": index,
        }
        for index in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection
