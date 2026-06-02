"""
Prompt building and Groq generation.

This module owns the "retrieved chunks + question -> answer" step.
"""

from groq import Groq

from config import GROQ_MODEL_NAME, GROQ_TEMPERATURE


SYSTEM_PROMPT = "You are a careful RAG assistant that answers only from supplied context."


def build_prompt(query: str, retrieved_chunks: list[str]) -> str:
    """Build the prompt sent to the LLM."""
    context = "\n\n---\n\n".join(retrieved_chunks)

    return f"""
Use the retrieved document context to answer the question.

Rules:
- Answer only from the provided context when possible.
- If the context does not contain the answer, say you do not know from the document.
- Be concise, but include enough detail to be useful.
- Mention source labels, such as Source 1 or Source 2, when they support the answer.

Retrieved context:
{context}

Question:
{query}
""".strip()


def generate_answer(
    query: str,
    retrieved_chunks: list[str],
    groq_api_key: str,
    model_name: str = GROQ_MODEL_NAME,
) -> str:
    """Call Groq with the RAG prompt and return the assistant answer."""
    client = Groq(api_key=groq_api_key)
    prompt = build_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=GROQ_TEMPERATURE,
    )

    return response.choices[0].message.content or ""
