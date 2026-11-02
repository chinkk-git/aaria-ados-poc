"""Policy regression tests for the single-use Issue #9 artifact handoff workflow."""
from pathlib import Path
import re
import unittest

WORKFLOW = Path(".github/workflows/ados-artifact-handoff.yml").read_text(encoding="utf-8")


class ArtifactHandoffWorkflowPolicyTests(unittest.TestCase):
    def test_manual_only_and_main_only_dispatch(self):
        trigger = WORKFLOW.split("\npermissions:", 1)[0]
        self.assertRegex(trigger, r"(?m)^on:\n  workflow_dispatch:$")
        self.assertNotRegex(trigger, r"(?m)^  (issues|push|pull_request|schedule):")
        self.assertIn("github.ref == 'refs/heads/main'", WORKFLOW)
        self.assertIn("github.actor == 'chinkk-git'", WORKFLOW)
        self.assertIn("github.repository == 'chinkk-git/aaria-ados-poc'", WORKFLOW)

    def test_task_source_digest_and_scope_are_pinned(self):
        for value in (
            "ADOS-POC-002-ARTIFACT-HANDOFF-001",
            "0b3dde4c94f045544632fe39d3fdc4c06bd55686",
            "f6b3e7d235ad13a6862e7c579598cc0ad7a0408c0a84d5a441f0e74f59eaf250",
            "feature/ados-poc-002",
            "657e51007d9bb7cb95d9ca3931ec45da1c34c149",
        ):
            self.assertIn(value, WORKFLOW)
        self.assertIn("git add -- daily-practice.html", WORKFLOW)
        self.assertIn('git diff --cached --name-only', WORKFLOW)
        self.assertIn('git diff-tree --no-commit-id --name-only -r "$commit_sha"', WORKFLOW)

    def test_shared_exclusive_lease_and_non_force_compare_and_swap(self):
        self.assertIn("group: ados-poc-002-exclusive-write-lease", WORKFLOW)
        self.assertIn("cancel-in-progress: false", WORKFLOW)
        self.assertIn('git push --porcelain origin "$commit_sha:refs/heads/$TARGET_BRANCH"', WORKFLOW)
        self.assertNotRegex(WORKFLOW, r"(?i)git\s+push\s+[^\n]*--force")

    def test_durable_single_use_claim_precedes_any_write(self):
        marker_check = WORKFLOW.index('grep -Fq "$CLAIM_MARKER"')
        claim_write = WORKFLOW.index('"repos/$GH_REPOSITORY/issues/9/comments"')
        branch_write = WORKFLOW.index('git push --porcelain origin')
        self.assertLess(marker_check, claim_write)
        self.assertLess(claim_write, branch_write)
        self.assertIn("gh api --paginate --jq", WORKFLOW)

    def test_minimal_permissions_and_no_external_ai_route(self):
        self.assertIn("contents: read\n  issues: read\n  actions: read", WORKFLOW)
        self.assertIn("contents: write\n      issues: write", WORKFLOW)
        for forbidden in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "openrouter/", "generativelanguage.googleapis.com"):
            self.assertNotIn(forbidden, WORKFLOW)


if __name__ == "__main__":
    unittest.main()
