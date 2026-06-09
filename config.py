"""
Central settings for the RAG chatbot.

Keeping settings in one file makes experiments easier: you can change chunk
size, model names, or storage paths without hunting through the application.
"""

import os
from pathlib import Path


# SentenceTransformers runs locally, so this model needs no API key.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model availability can change, so this can be overridden with GROQ_MODEL.
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# The Chroma directory is local persistent storage for document vectors.
CHROMA_DIR = Path(".chroma_rag")

# Chroma collection names must be stable so the same PDF can be reused later.
COLLECTION_PREFIX = "rag"

# Character-based chunking is simple to learn; production systems often use token chunking.
CHUNK_SIZE = 1200

# Overlap keeps context around chunk boundaries, reducing chopped-off answers.
CHUNK_OVERLAP = 200

# The requirement asks retrieval to return the top 3 relevant chunks.
TOP_K = 3

# Hybrid retrieval fetches a wider candidate pool before fusing results.
VECTOR_CANDIDATES = 8
BM25_CANDIDATES = 8

# Equal weights make lexical and semantic rank signals equally important.
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 1.0

# Cross-encoder reranking scores final candidate query/chunk pairs directly.
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_CANDIDATES = 8

# Optional retrieval distance cutoff. Lower distances are better for Chroma queries.
MAX_RETRIEVAL_DISTANCE = None

# Low temperature makes answers more grounded and less creative.
GROQ_TEMPERATURE = 0.2
