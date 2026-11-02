import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LocalPageHarnessTests(unittest.TestCase):
    def test_schema_and_example_match_zero_cost_scope(self):
        schema = json.loads((ROOT / "schemas/task-manifest.schema.json").read_text())
        task = json.loads((ROOT / "tasks/example-task.json").read_text())
        self.assertEqual(schema["properties"]["cost_ceiling_usd"]["const"], 0.0)
        self.assertEqual(task["cost_ceiling_usd"], 0.0)
        self.assertEqual(task["scope_paths"], ["daily-practice.html"])

    def test_browser_harness_checks_click_reload_accessibility_and_network(self):
        source = (ROOT / "scripts/check_local_generated_page.cjs").read_text()
        for required in (
            'getByRole("button", { name: "Mark Complete", exact: true })',
            'getByRole("status")',
            'page.reload(',
            'page.keyboard.press("Enter")',
            'unexpectedRequests',
            'localStorage',
            'sessionStorage',
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
