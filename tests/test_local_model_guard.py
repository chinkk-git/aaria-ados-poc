from pathlib import Path
import unittest


class LocalModelGuardPolicyTests(unittest.TestCase):
    def test_guard_paginates_history_and_requires_every_prior_download_and_inference_skipped(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "python-multiprovider-executor.yml").read_text()
        self.assertIn("gh api --paginate --slurp", workflow)
        self.assertIn("repos/$GH_REPOSITORY/actions/runs/$prior_id/jobs?per_page=100", workflow)
        self.assertIn('($steps | length) == 2 and ($steps | all(. == "skipped"))', workflow)

    def test_guard_has_no_arbitrary_prior_run_count_cutoff(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "python-multiprovider-executor.yml").read_text()
        self.assertNotIn("prior_count >", workflow)
        self.assertIn('test "$CURRENT_ATTEMPT" = "1"', workflow)


if __name__ == "__main__":
    unittest.main()
