"""
Retrieval logic.

This module owns the "query -> top matching chunks" step.
"""

from sentence_transformers import SentenceTransformer

from bm25 import BM25Index
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


def chunk_key(match: dict) -> str:
    """Return a stable key for merging vector and BM25 matches."""
    metadata = match.get("metadata") or {}
    source = metadata.get("source", "")
    chunk_index = metadata.get("chunk_index")
    if chunk_index is not None:
        return f"{source}:{chunk_index}"

    return str(match.get("id", match.get("document", "")))


def reciprocal_rank(rank: int, rank_constant: int = 60) -> float:
    """Convert a result rank into a reciprocal-rank fusion score."""
    return 1 / (rank_constant + rank)


def fuse_ranked_matches(
    vector_matches: list[dict],
    bm25_matches: list[dict],
    top_k: int = TOP_K,
    vector_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict]:
    """Merge vector and BM25 matches using weighted reciprocal-rank fusion."""
    fused: dict[str, dict] = {}

    for match in vector_matches:
        key = chunk_key(match)
        fused.setdefault(key, dict(match))
        fused[key]["vector_rank"] = match["vector_rank"]
        fused[key]["hybrid_score"] = fused[key].get("hybrid_score", 0.0) + (
            vector_weight * reciprocal_rank(match["vector_rank"])
        )

    for match in bm25_matches:
        key = chunk_key(match)
        fused.setdefault(key, dict(match))
        fused[key]["bm25_rank"] = match["bm25_rank"]
        fused[key]["bm25_score"] = match["bm25_score"]
        fused[key]["hybrid_score"] = fused[key].get("hybrid_score", 0.0) + (
            bm25_weight * reciprocal_rank(match["bm25_rank"])
        )

    matches = list(fused.values())
    matches.sort(key=lambda match: match["hybrid_score"], reverse=True)
    return matches[:top_k]


def search_vector_chunks(
    query: str,
    collection,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
    max_distance: float | None = MAX_RETRIEVAL_DISTANCE,
) -> list[dict]:
    """Return vector search matches with documents, metadata, and distances."""
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

    matches: list[dict] = []
    for index, (document, distance, metadata) in enumerate(
        zip(documents, distances, metadatas),
    ):
        if max_distance is not None and distance > max_distance:
            continue

        matches.append(
            {
                "id": str(metadata.get("chunk_index", index)),
                "document": document,
                "metadata": metadata,
                "distance": distance,
                "vector_rank": index + 1,
            }
        )

    return matches


def search_bm25_chunks(query: str, collection, top_k: int = TOP_K) -> list[dict]:
    """Return BM25 keyword matches from the indexed Chroma documents."""
    indexed_chunks = load_indexed_chunks(collection)
    documents = [chunk["document"] for chunk in indexed_chunks]
    bm25_index = BM25Index(documents)
    scores = bm25_index.score(query)

    matches: list[dict] = []
    for chunk, score in zip(indexed_chunks, scores):
        if score <= 0:
            continue

        matches.append(
            {
                "id": chunk["id"],
                "document": chunk["document"],
                "metadata": chunk["metadata"],
                "bm25_score": score,
            }
        )

    matches.sort(key=lambda match: match["bm25_score"], reverse=True)
    for index, match in enumerate(matches, start=1):
        match["bm25_rank"] = index

    return matches[:top_k]


def retrieve_top_chunks(
    query: str,
    collection,
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
    max_distance: float | None = MAX_RETRIEVAL_DISTANCE,
) -> list[str]:
    """Return the most relevant document chunks for a user query."""
    matches = search_vector_chunks(query, collection, embedding_model, top_k, max_distance)
    return [
        format_retrieved_chunk(
            rank,
            match["document"],
            match["metadata"],
            f"distance={match['distance']:.4f}",
        )
        for rank, match in enumerate(matches, start=1)
    ]
