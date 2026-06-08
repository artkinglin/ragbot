"""
Retrieval logic.

This module owns the "query -> top matching chunks" step.
"""

from sentence_transformers import SentenceTransformer

from config import MAX_RETRIEVAL_DISTANCE, TOP_K
from embeddings import embed_texts


def load_indexed_chunks(collection) -> list[dict]:
    """Load indexed Chroma chunks with their metadata."""
    results = collection.get(include=["documents", "metadatas"])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    ids = results.get("ids", [])

    chunks: list[dict] = []
    for index, document in enumerate(documents):
        chunks.append(
            {
                "id": ids[index] if index < len(ids) else str(index),
                "document": document,
                "metadata": metadatas[index] if index < len(metadatas) else {},
            }
        )

    return chunks


def format_retrieved_chunk(
    rank: int,
    document: str,
    metadata: dict,
    score_label: str,
) -> str:
    """Format one retrieved chunk with source metadata."""
    page_label = f" | page={metadata.get('page')}" if metadata.get("page") else ""
    return (
        f"Source {rank} | chunk={metadata.get('chunk_index')}{page_label} | {score_label}\n"
        f"{document}"
    )


def retrieve_top_chunks(
    query: str,
    collection,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
    max_distance: float | None = MAX_RETRIEVAL_DISTANCE,
) -> list[str]:
    """Return the most relevant document chunks for a user query."""
    query_embedding = embed_texts(embedding_model, [query])[0]

    # Passing query_embeddings keeps Chroma from hiding embedding behavior.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"],
    )

    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    formatted_chunks: list[str] = []
    for rank, (document, distance, metadata) in enumerate(
        zip(documents, distances, metadatas),
        start=1,
    ):
        if max_distance is not None and distance > max_distance:
            continue

        formatted_chunks.append(
            format_retrieved_chunk(rank, document, metadata, f"distance={distance:.4f}")
        )

    return formatted_chunks
