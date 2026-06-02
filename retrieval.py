"""
Retrieval logic.

This module owns the "query -> top matching chunks" step.
"""

from sentence_transformers import SentenceTransformer

from config import TOP_K
from embeddings import embed_texts


def retrieve_top_chunks(
    query: str,
    collection,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
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
        # The labels help both the LLM and the developer see where evidence came from.
        page_label = f" | page={metadata.get('page')}" if metadata.get("page") else ""
        formatted_chunks.append(
            f"Source {rank} | chunk={metadata.get('chunk_index')}{page_label} | distance={distance:.4f}\n"
            f"{document}"
        )

    return formatted_chunks
