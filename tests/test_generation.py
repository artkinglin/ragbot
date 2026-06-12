import unittest

from generation import (
    CITATION_FAILURE_ANSWER,
    NO_CONTEXT_ANSWER,
    build_citation_repair_prompt,
    build_prompt,
    generate_answer,
    generate_answer_with_client,
)


def make_response(content: str):
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message})()
    return type("Response", (), {"choices": [choice]})()


class FakeCompletions:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.last_kwargs = None
        self.calls = []
        self.responses = responses or ["Grounded answer [Source 1]."]

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        self.calls.append(kwargs)
        return make_response(self.responses.pop(0))


class FakeChat:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.completions = FakeCompletions(responses)


class FakeGroqClient:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.chat = FakeChat(responses)


class BuildPromptTests(unittest.TestCase):
    def test_prompt_contains_query_and_context(self) -> None:
        prompt = build_prompt("What happened?", ["Source 1\nImportant context."])

        self.assertIn("What happened?", prompt)
        self.assertIn("Important context.", prompt)

    def test_prompt_instructs_model_to_admit_missing_context(self) -> None:
        prompt = build_prompt("Unknown?", ["Source 1\nPartial context."])

        self.assertIn("say you do not know from the document", prompt)

    def test_prompt_requires_canonical_inline_citations(self) -> None:
        prompt = build_prompt("What happened?", ["Source 1\nImportant context."])

        self.assertIn("Cite every factual claim", prompt)
        self.assertIn("[Source 1]", prompt)
        self.assertIn("Never invent a source label", prompt)

    def test_repair_prompt_lists_errors_and_allowed_sources(self) -> None:
        prompt = build_citation_repair_prompt(
            "Unsupported answer.",
            ["Source 1\nContext.", "Source 2\nMore context."],
            ("Answer must include a citation.",),
        )

        self.assertIn("Answer must include a citation.", prompt)
        self.assertIn("[Source 1], [Source 2]", prompt)
        self.assertIn("Unsupported answer.", prompt)


class GenerateAnswerTests(unittest.TestCase):
    def test_returns_no_context_answer_without_calling_groq(self) -> None:
        answer = generate_answer("Unknown?", [], "unused-key")

        self.assertEqual(answer, NO_CONTEXT_ANSWER)

    def test_generate_answer_with_client_uses_existing_client(self) -> None:
        client = FakeGroqClient()

        answer = generate_answer_with_client("Question?", ["Source 1\nContext."], client, "model-a")

        self.assertEqual(answer, "Grounded answer [Source 1].")
        self.assertEqual(client.chat.completions.last_kwargs["model"], "model-a")
        self.assertEqual(client.chat.completions.last_kwargs["messages"][0]["role"], "system")
        self.assertEqual(len(client.chat.completions.calls), 1)

    def test_repairs_an_answer_without_citations(self) -> None:
        client = FakeGroqClient(
            [
                "Grounded answer.",
                "Grounded answer [Source 1].",
            ]
        )

        answer = generate_answer_with_client(
            "Question?",
            ["Source 1\nContext."],
            client,
            "model-a",
        )

        self.assertEqual(answer, "Grounded answer [Source 1].")
        self.assertEqual(len(client.chat.completions.calls), 2)
        repair_messages = client.chat.completions.calls[1]["messages"]
        self.assertEqual(repair_messages[-2]["role"], "assistant")
        self.assertIn("Allowed citation labels", repair_messages[-1]["content"])

    def test_repairs_an_answer_with_an_invalid_source(self) -> None:
        client = FakeGroqClient(
            [
                "Invented support [Source 9].",
                "Supported answer [Source 1].",
            ]
        )

        answer = generate_answer_with_client(
            "Question?",
            ["Source 1\nContext."],
            client,
        )

        self.assertEqual(answer, "Supported answer [Source 1].")
        self.assertIn("Source 9", client.chat.completions.calls[1]["messages"][-1]["content"])

    def test_returns_safe_failure_when_repair_is_still_invalid(self) -> None:
        client = FakeGroqClient(
            [
                "Uncited answer.",
                "Still cites a fabricated source [Source 7].",
            ]
        )

        answer = generate_answer_with_client(
            "Question?",
            ["Source 1\nContext."],
            client,
        )

        self.assertEqual(answer, CITATION_FAILURE_ANSWER)
        self.assertEqual(len(client.chat.completions.calls), 2)


if __name__ == "__main__":
    unittest.main()
