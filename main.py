"""
CLI entry point for the RAG chatbot.

Run:
    python main.py path\\to\\document.pdf
"""

import argparse
import os
import sys
import textwrap
from pathlib import Path


def load_local_env() -> None:
    """Load simple KEY=value pairs from the project .env file."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL_NAME,
    MAX_RETRIEVAL_DISTANCE,
    TOP_K,
)
from embeddings import index_chunks, load_embedding_model
from generation import create_groq_client, generate_answer_with_client
from ingestion import chunk_text, load_pdf_text
from retrieval import retrieve_top_chunks


def require_groq_api_key() -> str:
    """Read the Groq API key from the environment."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY before running the chatbot.")
    return api_key


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Chat with a PDF using a raw Python RAG pipeline.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF you want to chat with.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of chunks to retrieve.")
    parser.add_argument("--max-distance", type=float, default=MAX_RETRIEVAL_DISTANCE, help="Optional maximum Chroma distance to keep.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Number of characters per text chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Characters repeated between neighboring chunks.")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME, help="SentenceTransformers model used for local embeddings.")
    parser.add_argument("--groq-model", default=GROQ_MODEL_NAME, help="Groq model used to generate answers.")
    parser.add_argument("--chroma-dir", type=Path, default=CHROMA_DIR, help="Directory where ChromaDB stores embeddings.")
    parser.add_argument("--debug", action="store_true", help="Print retrieved chunks before each answer.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild the Chroma collection before chatting.")
    return parser.parse_args()


def validate_pdf_path(pdf_path: Path) -> None:
    """Fail early with clear messages for common input mistakes."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path}")


def validate_cli_options(args: argparse.Namespace) -> None:
    """Fail early when numeric settings cannot produce useful retrieval."""
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    if args.max_distance is not None and args.max_distance < 0:
        raise ValueError("--max-distance cannot be negative.")

    if args.chunk_size < 100:
        raise ValueError("--chunk-size must be at least 100 characters.")

    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap cannot be negative.")

    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size.")


def prepare_document(
    pdf_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model_name: str,
    chroma_dir: Path,
    reindex: bool,
):
    """Load, chunk, embed, and store a PDF in Chroma."""
    print("Loading embedding model...")
    embedding_model = load_embedding_model(embedding_model_name)

    print("Reading PDF...")
    text = load_pdf_text(pdf_path)

    print("Chunking PDF text...")
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    print(f"Indexing {len(chunks)} chunks in ChromaDB...")
    collection = index_chunks(
        pdf_path,
        chunks,
        embedding_model,
        chroma_dir,
        reindex,
        embedding_model_name,
        chunk_size,
        chunk_overlap,
    )

    return collection, embedding_model


def chat_loop(
    collection,
    embedding_model,
    groq_client,
    top_k: int,
    max_distance: float | None,
    groq_model: str,
    debug: bool,
) -> None:
    """Run the interactive chat session."""
    print("\nChat with your document. Type 'exit' or 'quit' to stop.\n")

    while True:
        query = normalize_query(input("You: "))
        if query.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return

        if not query:
            continue

        retrieved_chunks = retrieve_top_chunks(
            query,
            collection,
            embedding_model,
            top_k,
            max_distance,
        )
        if debug:
            print_retrieved_chunks(retrieved_chunks)

        answer = generate_answer_with_client(query, retrieved_chunks, groq_client, groq_model)

        print("\nAssistant:")
        print(textwrap.fill(answer, width=100))
        print()


def print_retrieved_chunks(retrieved_chunks: list[str]) -> None:
    """Print retrieved context so users can debug the retrieval step."""
    print("\nRetrieved chunks:")
    for chunk in retrieved_chunks:
        print(chunk)
        print("---")


def normalize_query(raw_query: str) -> str:
    """Normalize terminal input before retrieval."""
    return raw_query.strip()


def main() -> int:
    """Coordinate setup, indexing, and chatting."""
    args = parse_args()

    try:
        validate_pdf_path(args.pdf)
        validate_cli_options(args)
        groq_api_key = require_groq_api_key()
        groq_client = create_groq_client(groq_api_key)
        collection, embedding_model = prepare_document(
            args.pdf,
            args.chunk_size,
            args.chunk_overlap,
            args.embedding_model,
            args.chroma_dir,
            args.reindex,
        )
        chat_loop(
            collection,
            embedding_model,
            groq_client,
            args.top_k,
            args.max_distance,
            args.groq_model,
            args.debug,
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
