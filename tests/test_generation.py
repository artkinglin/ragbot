import unittest

from generation import (
    NO_CONTEXT_ANSWER,
    build_prompt,
    generate_answer,
    generate_answer_with_client,
)


class Message:
    content = "Grounded answer."


class Choice:
    message = Message()


class Response:
    choices = [Choice()]


class FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return Response()


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeGroqClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


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


class GenerateAnswerTests(unittest.TestCase):
    def test_returns_no_context_answer_without_calling_groq(self) -> None:
        answer = generate_answer("Unknown?", [], "unused-key")

        self.assertEqual(answer, NO_CONTEXT_ANSWER)

    def test_generate_answer_with_client_uses_existing_client(self) -> None:
        client = FakeGroqClient()

        answer = generate_answer_with_client("Question?", ["Source 1\nContext."], client, "model-a")

        self.assertEqual(answer, "Grounded answer.")
        self.assertEqual(client.chat.completions.last_kwargs["model"], "model-a")
        self.assertEqual(client.chat.completions.last_kwargs["messages"][0]["role"], "system")


if __name__ == "__main__":
    unittest.main()
