"""
Cross-encoder reranking.

Hybrid retrieval creates a candidate set. A cross-encoder then scores each
query/chunk pair directly, which is slower than embedding search but usually
more precise for the final context ordering.
"""

from sentence_transformers import CrossEncoder


def load_cross_encoder_model(model_name: str) -> CrossEncoder:
    """Load the cross-encoder reranker model."""
    return CrossEncoder(model_name)


def rerank_matches(query: str, matches: list[dict], reranker, top_k: int) -> list[dict]:
    """Rerank candidate matches by cross-encoder relevance score."""
    if not matches:
        return []

    pairs = [(query, match["document"]) for match in matches]
    scores = reranker.predict(pairs)

    reranked_matches: list[dict] = []
    for match, score in zip(matches, scores):
        reranked_match = dict(match)
        reranked_match["rerank_score"] = float(score)
        reranked_matches.append(reranked_match)

    reranked_matches.sort(key=lambda match: match["rerank_score"], reverse=True)
    for index, match in enumerate(reranked_matches, start=1):
        match["rerank_rank"] = index

    return reranked_matches[:top_k]
