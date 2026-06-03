import unittest

from generation import NO_CONTEXT_ANSWER, build_prompt, generate_answer


class BuildPromptTests(unittest.TestCase):
    def test_prompt_contains_query_and_context(self) -> None:
        prompt = build_prompt("What happened?", ["Source 1\nImportant context."])

        self.assertIn("What happened?", prompt)
        self.assertIn("Important context.", prompt)

    def test_prompt_instructs_model_to_admit_missing_context(self) -> None:
        prompt = build_prompt("Unknown?", ["Source 1\nPartial context."])

        self.assertIn("say you do not know from the document", prompt)


class GenerateAnswerTests(unittest.TestCase):
    def test_returns_no_context_answer_without_calling_groq(self) -> None:
        answer = generate_answer("Unknown?", [], "unused-key")

        self.assertEqual(answer, NO_CONTEXT_ANSWER)


if __name__ == "__main__":
    unittest.main()
