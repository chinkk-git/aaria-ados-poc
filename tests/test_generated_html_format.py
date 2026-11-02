import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from normalize_generated_html import normalize_html


class GeneratedHtmlFormatTests(unittest.TestCase):
    def test_unwraps_exact_html_fence_and_matches_preserved_validated_artifact(self):
        raw = (ROOT / "evidence/local-model-attempt-2-raw-output.html").read_text(encoding="utf-8")
        cleaned = normalize_html(raw)
        committed = (ROOT / "evidence/local-model-attempt-2-validated.html").read_text(encoding="utf-8")
        self.assertEqual(cleaned, committed)
        self.assertTrue(cleaned.startswith("<!DOCTYPE html>"))
        self.assertTrue(cleaned.rstrip().endswith("</html>"))

    def test_accepts_raw_complete_html(self):
        self.assertEqual(normalize_html("<!DOCTYPE html><html></html>"), "<!DOCTYPE html><html></html>\n")

    def test_rejects_unclosed_or_non_html_fence(self):
        for source in (
            "```html\n<!DOCTYPE html><html></html>",
            "```text\n<!DOCTYPE html><html></html>\n```",
            "```html\n<!DOCTYPE html><html></html>\n```\nprose",
            "intro\n<!DOCTYPE html><html></html>",
            "```html\n```html\n<!DOCTYPE html><html></html>\n```\n```",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                normalize_html(source)

    def test_rejects_unbalanced_or_external_reference_html(self):
        for source in (
            "<!DOCTYPE html><html><body></html>",
            '<!DOCTYPE html><html><head><script src="https://x"></script></head></html>',
            "<!DOCTYPE html><html></html><p>extra wrapper</p>",
        ):
            with self.subTest(source=source), self.assertRaises(ValueError):
                normalize_html(source)


if __name__ == "__main__":
    unittest.main()
