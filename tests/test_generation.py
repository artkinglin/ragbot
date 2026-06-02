import unittest

from generation import build_prompt


class BuildPromptTests(unittest.TestCase):
    def test_prompt_contains_query_and_context(self) -> None:
        prompt = build_prompt("What happened?", ["Source 1\nImportant context."])

        self.assertIn("What happened?", prompt)
        self.assertIn("Important context.", prompt)

    def test_prompt_instructs_model_to_admit_missing_context(self) -> None:
        prompt = build_prompt("Unknown?", ["Source 1\nPartial context."])

        self.assertIn("say you do not know from the document", prompt)


if __name__ == "__main__":
    unittest.main()
