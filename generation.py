"""
Prompt building and Groq generation.

This module owns the "retrieved chunks + question -> answer" step.
"""

from groq import Groq

from citations import extract_available_sources, validate_citations
from config import GROQ_MODEL_NAME, GROQ_TEMPERATURE


SYSTEM_PROMPT = (
    "You are a careful RAG assistant that answers only from supplied context "
    "and cites supporting source labels exactly."
)
NO_CONTEXT_ANSWER = "I do not know from the document because retrieval did not find relevant context."
CITATION_FAILURE_ANSWER = (
    "I could not produce an answer with valid citations from the retrieved context."
)


def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    """Build the prompt sent to the LLM."""
    context = "\n\n---\n\n".join(retrieved_chunks)

    return f"""
Use the retrieved document context to answer the question.

Rules:
- Answer only from the provided context when possible.
- If the context does not contain the answer, say you do not know from the document.
- Be concise, but include enough detail to be useful.
- Cite every factual claim with one or more supporting labels in square brackets.
- Use only labels from the retrieved context, formatted exactly as [Source 1] or [Source 2].
- Never invent a source label or mention a source without square brackets.

Retrieved context:
{context}

Question:
{query}
""".strip()


def create_groq_client(groq_api_key: str) -> Groq:
    """Create the Groq client once for a chat session."""
    return Groq(api_key=groq_api_key)


def build_citation_repair_prompt(
    answer: str,
    retrieved_chunks: list[str],
    errors: tuple[str, ...],
) -> str:
    """Build a constrained request to repair an invalid answer."""
    available_sources = extract_available_sources(retrieved_chunks)
    allowed_labels = ", ".join(
        f"[Source {source_number}]" for source_number in sorted(available_sources)
    )
    error_list = "\n".join(f"- {error}" for error in errors)

    return f"""
Rewrite the answer so it follows the citation rules.

Validation errors:
{error_list}

Allowed citation labels:
{allowed_labels}

Requirements:
- Preserve only claims supported by the retrieved context.
- Cite factual claims inline using only the allowed labels.
- Use the exact format [Source N].
- Return only the corrected answer.

Invalid answer:
{answer}
""".strip()


def request_completion(client: Groq, model_name: str, messages: list[dict]) -> str:
    """Request one completion and normalize an empty response."""
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=GROQ_TEMPERATURE,
    )
    return response.choices[0].message.content or ""


def generate_answer_with_client(
    query: str,
    retrieved_chunks: list[str],
    client: Groq,
    model_name: str = GROQ_MODEL_NAME,
) -> str:
    """Call Groq with an existing client and return the assistant answer."""
    if not retrieved_chunks:
        return NO_CONTEXT_ANSWER

    prompt = build_prompt(query, retrieved_chunks)
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    answer = request_completion(client, model_name, messages)
    validation = validate_citations(answer, retrieved_chunks)
    if validation.is_valid:
        return answer

    repair_prompt = build_citation_repair_prompt(
        answer,
        retrieved_chunks,
        validation.errors,
    )
    repaired_answer = request_completion(
        client,
        model_name,
        [
            *messages,
            {"role": "assistant", "content": answer},
            {"role": "user", "content": repair_prompt},
        ],
    )
    repaired_validation = validate_citations(repaired_answer, retrieved_chunks)
    if repaired_validation.is_valid:
        return repaired_answer

    return CITATION_FAILURE_ANSWER


def generate_answer(
    query: str,
    retrieved_chunks: list[str],
    groq_api_key: str,
    model_name: str = GROQ_MODEL_NAME,
) -> str:
    """Call Groq with the RAG prompt and return the assistant answer."""
    client = create_groq_client(groq_api_key)
    return generate_answer_with_client(query, retrieved_chunks, client, model_name)
