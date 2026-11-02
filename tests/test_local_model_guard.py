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

    def test_second_attempt_uses_a_distinct_owner_authorization(self):
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "python-multiprovider-executor.yml").read_text()
        record = (root / "tasks" / "local-model-attempt-2.json").read_text()
        self.assertIn("Claim authorized second inference", workflow)
        self.assertIn("steps.one-shot-guard.outputs.second_attempt", workflow)
        self.assertIn("SECOND_INFERENCE_PARENT_VERIFIED", workflow)
        self.assertIn("LOCAL_MODEL_ATTEMPT_BLOCKED", workflow)
        self.assertIn("OWNER_AUTHORIZED", record)
        self.assertIn("41470648697", record)
        self.assertIn("157ba9063292d9e9451b1ebcc6649c95f376a6e5", record)
        self.assertIn("f2891a03a3a3f7408107e1014d1f2643c0f186dc52e4dde2a4576ea9e2293185", record)
        self.assertIn("preserve_original_one_shot_guard", record)
        self.assertIn('"authorization_count": 1', record)
        self.assertIn('"max_inference_requests": 1', record)
        self.assertIn('"no_paid_infrastructure": true', record)


if __name__ == "__main__":
    unittest.main()
