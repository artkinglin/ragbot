"""
Citation parsing and validation for generated RAG answers.
"""

import re
from dataclasses import dataclass


SOURCE_HEADER_PATTERN = re.compile(r"^Source\s+(\d+)\b", re.MULTILINE)
CITATION_PATTERN = re.compile(r"\[Source\s+(\d+)\]")
SOURCE_MENTION_PATTERN = re.compile(r"(?<!\[)\bSource\s+\d+\b(?!\])")


@dataclass(frozen=True)
class CitationValidation:
    """Describe whether an answer follows the citation policy."""

    is_valid: bool
    cited_sources: frozenset[int]
    available_sources: frozenset[int]
    errors: tuple[str, ...]


def extract_available_sources(retrieved_chunks: list[str]) -> frozenset[int]:
    """Return source numbers declared by retrieved chunk headers."""
    return frozenset(
        int(match.group(1))
        for chunk in retrieved_chunks
        for match in SOURCE_HEADER_PATTERN.finditer(chunk)
    )


def extract_citations(answer: str) -> frozenset[int]:
    """Return canonical inline citation numbers from an answer."""
    return frozenset(int(match.group(1)) for match in CITATION_PATTERN.finditer(answer))


def validate_citations(answer: str, retrieved_chunks: list[str]) -> CitationValidation:
    """Require canonical citations that refer only to retrieved sources."""
    available_sources = extract_available_sources(retrieved_chunks)
    cited_sources = extract_citations(answer)
    errors: list[str] = []

    if not cited_sources:
        errors.append("Answer must include at least one citation in the form [Source N].")

    invalid_sources = cited_sources - available_sources
    if invalid_sources:
        labels = ", ".join(f"Source {number}" for number in sorted(invalid_sources))
        errors.append(f"Answer cites unavailable sources: {labels}.")

    if SOURCE_MENTION_PATTERN.search(answer):
        errors.append("Source mentions must use the canonical form [Source N].")

    return CitationValidation(
        is_valid=not errors,
        cited_sources=cited_sources,
        available_sources=available_sources,
        errors=tuple(errors),
    )
