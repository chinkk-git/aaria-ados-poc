from pathlib import Path
import re
import unittest


class LocalModelGuardPolicyTests(unittest.TestCase):
    def test_guard_paginates_history_and_requires_every_prior_download_and_inference_skipped(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "python-multiprovider-executor.yml").read_text()
        self.assertIn("gh api --paginate --slurp", workflow)
        self.assertIn("select((.id | tostring) != $current)", workflow)
        self.assertIn("repos/$GH_REPOSITORY/actions/runs/$prior_id/jobs?per_page=100", workflow)
        self.assertIn('($steps | length) == 2 and ($steps | all(. == "skipped"))', workflow)
        self.assertIn('if ($jobs | length) == 0 then true', workflow)
        self.assertIn('($jobs | length) != 1 then false', workflow)
        self.assertIn('$jobs[0].conclusion == "skipped"', workflow)
        self.assertIn('select(.name == "local-model-one-shot")', workflow)
        self.assertIn('"41469249556"', workflow)
        self.assertIn('"8b81750df10d96aa4ef1cf2c0a48e82fcd5b3176"', workflow)
        self.assertIn("LOCAL_MODEL_DOWNLOAD: BLOCKED; code=model_redirect_host_not_allowlisted", workflow)

    def test_guard_has_no_arbitrary_prior_run_count_cutoff(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "python-multiprovider-executor.yml").read_text()
        self.assertIn("if (( prior_count > 0 )); then", workflow)
        self.assertNotRegex(workflow, r"prior_count\s*(?:>=|>)\s*(?!0(?:\D|$))\d+")
        self.assertIn('test "$CURRENT_ATTEMPT" = "1"', workflow)

    def test_model_revision_is_a_full_immutable_commit(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts" / "local_model_preflight.py").read_text()
        self.assertRegex(source, r'(?m)^MODEL_REVISION = "[0-9a-f]{40}"$')


if __name__ == "__main__":
    unittest.main()
