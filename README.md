# Raw Python RAG Chatbot

This project is a small Retrieval Augmented Generation chatbot backbone. It reads a PDF, chunks the text, embeds the chunks locally with SentenceTransformers, stores vectors in ChromaDB, retrieves the top 3 matching chunks for each question, and asks Groq to answer from that retrieved context.

No LangChain. No LlamaIndex. The goal is to make every moving part visible.

## Pipeline

```text
PDF -> text -> chunks -> embeddings -> ChromaDB -> top chunks -> Groq -> answer
```

## Files

- `config.py` - all settings in one place.
- `ingestion.py` - PDF loading and chunking.
- `embeddings.py` - turns text into vectors and stores them in ChromaDB.
- `retrieval.py` - finds relevant chunks for a query.
- `generation.py` - builds the prompt and calls Groq.
- `main.py` - CLI entry point that ties everything together.
- `README.md` - setup guide, learning notes, and interview Q&A.

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
- Prompt grounding: gives the LLM retrieved context and rules for answering.
- Environment variable: keeps `GROQ_API_KEY` out of source code.
- CLI loop: keeps accepting questions until the user exits.

## Things To Modify

1. Change `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py` and compare answer quality.
2. Print `retrieved_chunks` in `main.py` before generation to inspect what the retriever found.
3. Swap `EMBEDDING_MODEL_NAME` for another SentenceTransformers model and compare speed and relevance.

## Interview Q&A

**1. Why is RAG useful?**  
RAG gives the LLM relevant external context at question time, which helps answer from private or recent documents without retraining the model.

**2. Why chunk the PDF instead of embedding the whole document once?**  
Chunking makes retrieval more precise because the system can return only the parts related to the question instead of a huge unrelated document blob.

**3. What is an embedding?**  
An embedding is a vector representation of text where semantically similar text ends up near each other in vector space.

**4. Why use ChromaDB?**  
ChromaDB stores embeddings locally and provides similarity search, so the app can quickly find chunks close to the user's query.

**5. What does top-k mean?**  
Top-k is the number of best matching chunks returned from vector search; this project uses `TOP_K = 3`.

**6. What can cause bad answers in a RAG system?**  
Bad chunking, irrelevant retrieval, missing PDF text, weak prompts, or an LLM that ignores the supplied context can all degrade answers.
