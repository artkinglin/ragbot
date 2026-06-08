"""
BM25 keyword retrieval.

This module keeps lexical search local and dependency-free so the RAG pipeline
can combine exact keyword matches with vector similarity.
"""

import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    """Return normalized keyword tokens for BM25 scoring."""
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def calculate_idf(
    document_frequency: int,
    document_count: int,
) -> float:
    """Calculate a smoothed BM25 inverse document frequency value."""
    return math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))


class BM25Index:
    """In-memory BM25 index over a list of document chunks."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokenized_documents = [tokenize(document) for document in documents]
        self.document_lengths = [len(tokens) for tokens in self.tokenized_documents]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
            if self.document_lengths
            else 0.0
        )
        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized_documents]
        self.idf = self._build_idf()

    def _build_idf(self) -> dict[str, float]:
        document_frequencies: Counter[str] = Counter()
        for tokens in self.tokenized_documents:
            document_frequencies.update(set(tokens))

        document_count = len(self.documents)
        return {
            term: calculate_idf(document_frequency, document_count)
            for term, document_frequency in document_frequencies.items()
        }

    def score(self, query: str) -> list[float]:
        """Return BM25 scores for every indexed document."""
        query_terms = tokenize(query)
        if not query_terms or not self.documents:
            return [0.0 for _ in self.documents]

        scores: list[float] = []
        for index, term_frequency in enumerate(self.term_frequencies):
            document_length = self.document_lengths[index]
            score = 0.0

            for term in query_terms:
                frequency = term_frequency.get(term, 0)
                if frequency == 0:
                    continue

                length_ratio = (
                    document_length / self.average_document_length
                    if self.average_document_length
                    else 0.0
                )
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += self.idf.get(term, 0.0) * ((frequency * (self.k1 + 1)) / denominator)

            scores.append(score)

        return scores
