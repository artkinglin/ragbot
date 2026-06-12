# Raw Python RAG Chatbot

This project is a small Retrieval Augmented Generation chatbot backbone. It reads a PDF, chunks the text, embeds the chunks locally with SentenceTransformers, stores vectors in ChromaDB, retrieves matching chunks with hybrid BM25 keyword search plus vector search, reranks the fused candidates with a cross-encoder, asks Groq to answer from that retrieved context, and validates the answer's source citations.

No LangChain. No LlamaIndex. The goal is to make every moving part visible.

## Pipeline

```text
PDF -> text -> chunks -> embeddings -> ChromaDB -> BM25 + vector retrieval -> fused candidates -> cross-encoder rerank -> Groq -> citation validation -> answer
```

## Common Workflow

1. Add a PDF path when starting the CLI.
2. Let the app build or reuse the local Chroma index.
3. Ask questions in the terminal.
4. Turn on `--debug` if the answer looks wrong and inspect retrieval first.

## Files

- `config.py` - all settings in one place.
- `ingestion.py` - PDF loading and chunking.
- `embeddings.py` - turns text into vectors and stores them in ChromaDB.
- `bm25.py` - scores exact keyword matches with local BM25.
- `reranking.py` - reranks fused candidates with a cross-encoder.
- `retrieval.py` - finds relevant chunks with BM25, vector search, rank fusion, and reranking.
- `citations.py` - extracts source labels and validates answer citations.
- `generation.py` - builds the prompt, calls Groq, and repairs invalid citations once.
- `main.py` - CLI entry point that ties everything together.
- `README.md` - setup guide, learning notes, and interview Q&A.

Each Python file owns one RAG stage so the code can be read in pipeline order.

## Setup

```powershell
pip install -r requirements.txt
$env:GROQ_API_KEY="your-groq-api-key"
python main.py path\to\your.pdf
```

Optional model override:

```powershell
$env:GROQ_MODEL="llama-3.3-70b-versatile"
```

Use `.env.example` as a reference for the environment variables this project expects.

## Environment Variables

| Name | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | Authenticates Groq API requests. |
| `GROQ_MODEL` | No | Overrides the default Groq model. |

## Useful CLI Options

```powershell
python main.py --help
python main.py paper.pdf --top-k 5 --chunk-size 900 --chunk-overlap 150 --debug
python main.py paper.pdf --vector-candidates 12 --bm25-candidates 12 --vector-weight 1.0 --bm25-weight 1.0
python main.py paper.pdf --rerank-candidates 12 --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2
python main.py paper.pdf --no-rerank
python main.py paper.pdf --max-distance 0.8
python main.py paper.pdf --reindex
python main.py paper.pdf --embedding-model sentence-transformers/all-MiniLM-L6-v2
```

The app automatically rebuilds the Chroma index when chunking settings or the embedding model change.
Use `--reindex` when you want to force a rebuild anyway.
Use `--max-distance` to drop weak retrieval matches before generation.
Use `--debug` when you want to inspect retrieved chunks before Groq generates the final answer.
Use `--vector-candidates`, `--bm25-candidates`, `--vector-weight`, and `--bm25-weight` to tune hybrid retrieval.
Use `--rerank-candidates` and `--rerank-model` to tune cross-encoder reranking, or `--no-rerank` for faster hybrid-only retrieval.

## Citation Enforcement

Generated factual claims use canonical inline labels such as `[Source 1]`. The app checks that every generated answer contains at least one citation, uses square-bracket syntax, and refers only to source labels present in the retrieved context.

If the first response fails validation, the app asks Groq to repair it once using an explicit list of allowed labels. If the repaired response is still invalid, the app returns a fixed citation failure message instead of showing unsupported or fabricated citations. The no-context response is returned directly because there is no retrieved source to cite.

## Tests

```powershell
python -m unittest discover -s tests
```

## Concepts

- RAG: retrieves document evidence before asking an LLM to answer.
- PDF extraction: uses `pypdf` to pull text from PDF pages.
- Chunking: splits long text into smaller searchable pieces.
- Chunk overlap: repeats boundary text so ideas are not cut in half.
- Embeddings: converts text into numeric vectors representing meaning.
- Vector database: stores embeddings and supports similarity search.
- Chroma collection: a named group of stored vectors and source text.
- Query embedding: converts the user's question into the same vector space as chunks.
- Top-k retrieval: returns the most similar chunks, here the top 3.
- BM25: keyword search that rewards exact term overlap with the question.
- Hybrid retrieval: combines BM25 and vector ranks so exact keywords and semantic similarity can both surface evidence.
- Reciprocal-rank fusion: merges ranked lists without comparing unrelated raw score scales.
- Cross-encoder reranking: scores each question/chunk pair directly after retrieval to improve final ordering.
- Distance cutoff: optionally filters out weak matches before prompting the LLM.
- Prompt grounding: gives the LLM retrieved context and rules for answering.
- Citation validation: requires canonical source labels and rejects labels outside the retrieved context.
- Environment variable: keeps `GROQ_API_KEY` out of source code.
- CLI loop: keeps accepting questions until the user exits.

## Limitations

- Scanned PDFs need OCR before `pypdf` can extract useful text.
- Character chunking is easy to learn from, but token chunking is more precise.
- Retrieved chunks can still be irrelevant if both lexical and semantic signals miss the user's intent.
- Citation validation verifies source-label usage, not whether every cited sentence is semantically entailed by its source.

## Things To Modify

1. Change `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py` and compare answer quality.
2. Use `--debug` to inspect what the hybrid retriever found before generation.
3. Swap `EMBEDDING_MODEL_NAME` for another SentenceTransformers model and compare speed and relevance.
4. Adjust `VECTOR_WEIGHT` and `BM25_WEIGHT` in `config.py` to compare semantic-heavy and keyword-heavy retrieval.

## Interview Q&A

**1. Why is RAG useful?**  
RAG gives the LLM relevant external context at question time, which helps answer from private or recent documents without retraining the model.

**2. Why chunk the PDF instead of embedding the whole document once?**  
Chunking makes retrieval more precise because the system can return only the parts related to the question instead of a huge unrelated document blob.

**3. What is an embedding?**  
An embedding is a vector representation of text where semantically similar text ends up near each other in vector space.

**4. Why use ChromaDB?**  
ChromaDB stores embeddings locally and provides similarity search, so the app can quickly find chunks semantically close to the user's query.

**5. Why add BM25?**  
BM25 catches exact terms, names, acronyms, and numbers that embedding search can miss or under-rank.

**6. What does top-k mean?**  
Top-k is the number of best matching chunks returned from vector search; this project uses `TOP_K = 3`.

**7. Why rerank after hybrid retrieval?**  
Hybrid retrieval is good at gathering candidates. A cross-encoder is better at deciding which candidate most directly answers the specific question.

**8. What can cause bad answers in a RAG system?**  
Bad chunking, irrelevant retrieval, missing PDF text, weak prompts, or an LLM that ignores the supplied context can all degrade answers.

**9. How are citations enforced?**
The generator requires `[Source N]` labels, validates them against retrieved source headers, retries one invalid response with repair instructions, and suppresses the answer if the repair still fails.

## Portfolio Talking Points

- Built the RAG pipeline without LangChain or LlamaIndex to show the underlying mechanics.
- Added hybrid BM25 plus vector retrieval with reciprocal-rank fusion.
- Added cross-encoder reranking to improve final context ordering.
- Enforced valid inline citations with deterministic validation and a constrained repair attempt.
- Used local embeddings to avoid embedding API keys and keep retrieval inexpensive.
- Added debug retrieval mode so answer quality can be traced back to source chunks.
