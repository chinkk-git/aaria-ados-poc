"""Focused tests for the Issue #9 artifact handoff authorization and recovery policy."""
from pathlib import Path
import os
import re
import subprocess
import textwrap
import unittest

WORKFLOW = Path(".github/workflows/ados-artifact-handoff.yml").read_text(encoding="utf-8")
POLICY_MATCH = re.search(
    r"(?ms)^ {10}# ADOS_HANDOFF_POLICY_BEGIN\n(.*?)^ {10}# ADOS_HANDOFF_POLICY_END$",
    WORKFLOW,
)
if not POLICY_MATCH:
    raise RuntimeError("handoff policy block not found")
POLICY_SOURCE = "\n".join(
    line[10:] if line.startswith("          ") else line
    for line in POLICY_MATCH.group(1).splitlines()
)
POLICY = {"__name__": "handoff_policy_tests"}
exec(compile(POLICY_SOURCE, "<embedded-handoff-policy>", "exec"), POLICY)


CLAIM_ID = "ADOS-POC-002-ARTIFACT-HANDOFF-001"
CLAIM_MARKER = "<!-- ADOS_CLAIM:ADOS-POC-002-ARTIFACT-HANDOFF-001 -->"
CONSUMED_MARKER = "<!-- ADOS_CONSUMED:ADOS-POC-002-ARTIFACT-HANDOFF-001 -->"
SOURCE_SHA = "0b3dde4c94f045544632fe39d3fdc4c06bd55686"
ARTIFACT_SHA = "f6b3e7d235ad13a6862e7c579598cc0ad7a0408c0a84d5a441f0e74f59eaf250"
EXPECTED = "657e51007d9bb7cb95d9ca3931ec45da1c34c149"
COMMIT = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


def run_record(conclusion="cancelled", status="completed"):
    return {
        "id": 1001,
        "path": ".github/workflows/ados-artifact-handoff.yml",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "actor": {"login": "chinkk-git"},
        "triggering_actor": {"login": "chinkk-git"},
        "status": status,
        "conclusion": conclusion,
    }


def claim_comment(run_id=1001, author="github-actions[bot]"):
    return {
        "id": 9,
        "user": {"login": author},
        "body": "\n".join((
            CLAIM_MARKER,
            "ADOS-POC-002 artifact handoff CLAIMED.",
            f"Claim ID: {CLAIM_ID}",
            f"Run: {run_id}",
            f"Source: {SOURCE_SHA} (evidence/local-model-attempt-2-validated.html)",
            "Reviewed PR head: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            f"Artifact SHA-256: {ARTIFACT_SHA}",
            f"Target: feature/ados-poc-002 at {EXPECTED}",
            "Write scope: daily-practice.html only",
            "Lease: ados-poc-002-exclusive-write-lease",
        )),
    }


def base_state():
    return {
        "claim_id": CLAIM_ID,
        "claim_marker": CLAIM_MARKER,
        "consumed_marker": CONSUMED_MARKER,
        "source_commit_sha": SOURCE_SHA,
        "source_path": "evidence/local-model-attempt-2-validated.html",
        "source_sha256": ARTIFACT_SHA,
        "target_branch": "feature/ados-poc-002",
        "expected_target_sha": EXPECTED,
        "target_path": "daily-practice.html",
        "target_head": EXPECTED,
        "target_file_exists": False,
        "target_parent": "",
        "target_parent_count": 0,
        "target_digest": "",
        "target_files": [],
        "comments": [],
        "prior_run": {},
        "pr_merged": True,
        "live_pr_head": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "live_pr_author": "chinkk-git",
        "authorized_pr_head": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "live_pr_head_repo": "chinkk-git/aaria-ados-poc",
        "live_pr_head_ref": "experiment/python-multiprovider-executor",
        "compare": {
            "status": "ahead",
            "base_commit": {"sha": SOURCE_SHA},
            "merge_base_commit": {"sha": SOURCE_SHA},
            "head_commit": {"sha": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
        },
    }


class HandoffPolicyTests(unittest.TestCase):
    def test_authorization_uses_stable_reviewed_source_against_live_head(self):
        state = base_state()
        POLICY["verify_reviewed_source"](
            SOURCE_SHA, state["authorized_pr_head"], True, state["live_pr_head"], "chinkk-git", state["live_pr_head_repo"], state["live_pr_head_ref"], state["compare"]
        )
        mismatch = dict(state["compare"])
        mismatch["merge_base_commit"] = {"sha": "cccccccccccccccccccccccccccccccccccccccc"}
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["verify_reviewed_source"](SOURCE_SHA, state["authorized_pr_head"], True, state["live_pr_head"], "chinkk-git", state["live_pr_head_repo"], state["live_pr_head_ref"], mismatch)
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["verify_reviewed_source"](SOURCE_SHA, state["authorized_pr_head"], True, state["live_pr_head"], "other-user", state["live_pr_head_repo"], state["live_pr_head_ref"], state["compare"])
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["verify_reviewed_source"](SOURCE_SHA, "c" * 40, True, state["live_pr_head"], "chinkk-git", state["live_pr_head_repo"], state["live_pr_head_ref"], state["compare"])
        self.assertIn("reviewed_pr_head:", WORKFLOW)
        self.assertIn("required: true", WORKFLOW)
        self.assertIn('authorized_pr_head != live_pr_head', WORKFLOW)
        self.assertIn("Reviewed PR head: $AUTHORIZED_PR_HEAD", WORKFLOW)
        self.assertNotIn("APPROVED_PR_HEAD", WORKFLOW)

    def test_claim_exists_target_unchanged_resumes_same_claim(self):
        state = base_state()
        state["comments"] = [claim_comment()]
        state["prior_run"] = run_record(conclusion="failure")
        result = POLICY["decide"](state)
        self.assertEqual(result["mode"], "resume_same_claim")
        self.assertEqual(result["claim_comment_id"], 9)

    def test_claim_without_reviewed_head_binding_fails_closed(self):
        state = base_state()
        legacy = claim_comment()
        legacy["body"] = "\\n".join(line for line in legacy["body"].splitlines() if not line.startswith("Reviewed PR head: "))
        state["comments"] = [legacy]
        state["prior_run"] = run_record(conclusion="cancelled")
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["decide"](state)

    def test_exact_committed_file_finalizes_without_push(self):
        state = base_state()
        state["comments"] = [claim_comment()]
        state["prior_run"] = run_record(conclusion="cancelled")
        state.update({
            "target_head": COMMIT,
            "target_file_exists": True,
            "target_parent": EXPECTED,
            "target_parent_count": 1,
            "target_digest": ARTIFACT_SHA,
            "target_mode": "100644",
            "target_type": "blob",
            "target_files": ["daily-practice.html"],
        })
        result = POLICY["decide"](state)
        self.assertEqual(result["mode"], "finalize_without_push")
        self.assertEqual(result["commit_sha"], COMMIT)

    def test_already_consumed_exact_commit_is_idempotent(self):
        state = base_state()
        state["comments"] = [claim_comment(), {
            "id": 10,
            "user": {"login": "github-actions[bot]"},
            "body": "\n".join((
                CONSUMED_MARKER,
                f"Claim: {CLAIM_ID}",
                "Workflow run: 1001",
                f"Artifact SHA-256: {ARTIFACT_SHA}",
                f"Task branch commit: {COMMIT}",
                "Only daily-practice.html was committed.",
            )),
        }]
        state["prior_run"] = run_record(conclusion="success")
        state.update({
            "target_head": COMMIT,
            "target_file_exists": True,
            "target_parent": EXPECTED,
            "target_parent_count": 1,
            "target_digest": ARTIFACT_SHA,
            "target_mode": "100644",
            "target_type": "blob",
            "target_files": ["daily-practice.html"],
        })
        self.assertEqual(POLICY["decide"](state)["mode"], "already_finalized")

    def test_duplicate_or_wrong_owner_claim_fails_closed(self):
        state = base_state()
        state["comments"] = [claim_comment(), claim_comment(run_id=1002)]
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["decide"](state)
        state["comments"] = [claim_comment(author="chinkk-git")]
        state["prior_run"] = run_record(conclusion="failure")
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["decide"](state)

    def test_conflicting_target_and_active_or_successful_unchanged_run_fail_closed(self):
        state = base_state()
        state["comments"] = [claim_comment()]
        state["prior_run"] = run_record(conclusion="failure")
        state["target_head"] = "dddddddddddddddddddddddddddddddddddddddd"
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["decide"](state)
        state = base_state()
        state["comments"] = [claim_comment()]
        state["prior_run"] = run_record(conclusion="success")
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["decide"](state)

    def test_legacy_consumption_record_is_recognized_without_repush(self):
        state = base_state()
        state["comments"] = [claim_comment(), {
            "id": 11,
            "user": {"login": "github-actions[bot]"},
            "body": "\n".join((
                "ADOS-POC-002 artifact handoff CONSUMED.",
                f"Claim: {CLAIM_ID} (claim comment 9)",
                "Workflow run: 1001",
                f"Artifact SHA-256: {ARTIFACT_SHA}",
                f"Task branch commit: {COMMIT}",
                "Only daily-practice.html was committed.",
            )),
        }]
        state["prior_run"] = run_record(conclusion="success")
        state.update({
            "target_head": COMMIT,
            "target_file_exists": True,
            "target_parent": EXPECTED,
            "target_parent_count": 1,
            "target_digest": ARTIFACT_SHA,
            "target_mode": "100644",
            "target_type": "blob",
            "target_files": ["daily-practice.html"],
        })
        self.assertEqual(POLICY["decide"](state)["mode"], "already_finalized")

    def test_exact_file_with_wrong_parent_or_digest_fails_closed(self):
        state = base_state()
        state["comments"] = [claim_comment()]
        state["prior_run"] = run_record(conclusion="cancelled")
        state.update({
            "target_head": COMMIT,
            "target_file_exists": True,
            "target_parent": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
            "target_parent_count": 1,
            "target_digest": ARTIFACT_SHA,
            "target_mode": "100644",
            "target_type": "blob",
            "target_files": ["daily-practice.html"],
        })
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["decide"](state)
        state["target_parent"] = EXPECTED
        state["target_digest"] = "f" * 64
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["decide"](state)

    def test_no_claim_only_allows_fresh_exact_target(self):
        state = base_state()
        self.assertEqual(POLICY["decide"](state)["mode"], "new_claim")
        state["target_head"] = COMMIT
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["decide"](state)

    def test_workflow_remains_manual_repo_actor_and_main_gated(self):
        self.assertIn("workflow_dispatch:\n    inputs:", WORKFLOW)
        self.assertIn("github.repository == 'chinkk-git/aaria-ados-poc'", WORKFLOW)
        self.assertIn("github.actor == 'chinkk-git'", WORKFLOW)
        self.assertIn("github.ref == 'refs/heads/main'", WORKFLOW)
        self.assertIn("required: true", WORKFLOW)
        self.assertIn("jq -r '.merged'", WORKFLOW)

    def test_task_scope_and_free_no_ai_boundary_remain_pinned(self):
        for value in (CLAIM_ID, SOURCE_SHA, ARTIFACT_SHA, EXPECTED, "feature/ados-poc-002", "daily-practice.html"):
            self.assertIn(value, WORKFLOW)
        for forbidden in ("OPENROUTER_API_KEY", "GEMINI_API_KEY", "openrouter/", "generativelanguage.googleapis.com"):
            self.assertNotIn(forbidden, WORKFLOW)
        self.assertIn('git add -- daily-practice.html', WORKFLOW)
        self.assertIn('git diff-tree --no-commit-id --name-only -r "$commit_sha"', WORKFLOW)

    def test_exclusive_lease_and_compare_and_swap_push_are_preserved(self):
        self.assertIn("group: ados-poc-002-exclusive-write-lease", WORKFLOW)
        self.assertIn("cancel-in-progress: false", WORKFLOW)
        self.assertIn('git push --porcelain origin "$commit_sha:refs/heads/$TARGET_BRANCH"', WORKFLOW)
        self.assertNotRegex(WORKFLOW, r"(?i)git\\s+push[^\\n]*--force")

    def test_finalize_path_rechecks_branch_tip_and_does_not_push(self):
        finalize = WORKFLOW.split("- name: Record completed handoff", 1)[1].split("- name: Confirm already finalized", 1)[0]
        self.assertIn('git ls-remote origin "refs/heads/$TARGET_BRANCH"', finalize)
        self.assertIn('test "$current_target" = "$COMMIT_SHA"', finalize)
        self.assertNotIn("git push", finalize)

    def _script_for_step(self, step_name):
        marker = f"      - name: {step_name}\n"
        start = WORKFLOW.index(marker)
        run_start = WORKFLOW.index("        run: |\n", start) + len("        run: |\n")
        end = WORKFLOW.find("\n      - name:", run_start)
        if end < 0:
            end = len(WORKFLOW)
        return textwrap.dedent(WORKFLOW[run_start:end])

    def _writer_record(self, step_name, variable_name, env):
        script = self._script_for_step(step_name)
        start = script.index(f'{variable_name}="$(printf')
        next_marker = "claim_comment_id=" if variable_name == "claim_body" else "gh api --method POST"
        next_position = script.index(next_marker, start)
        end = script.rfind("\n", start, next_position)
        assignment = script[start:end].rstrip()
        suffix = "\nprintf '%s' \"$" + variable_name + "\""
        result = subprocess.run(
            ["bash", "-c", assignment + suffix],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_flat_paginated_comment_payload_is_consumed_by_production_parser(self):
        state = base_state()
        first = {"id": 1, "user": {"login": "someone"}, "body": "ordinary comment"}
        second = claim_comment()
        state["comments"] = [first, second]
        self.assertIn("comments:[$comments[][]]", WORKFLOW)
        self.assertEqual(POLICY["_comments"](state), [first, second])
        state["prior_run"] = run_record(conclusion="failure")
        self.assertEqual(POLICY["decide"](state)["mode"], "resume_same_claim")
        with self.assertRaises(POLICY["PolicyError"]):
            POLICY["_comments"]({"comments": [[first], [second]]})

    def test_workflow_generated_claim_and_consumption_records_round_trip(self):
        env = {
            "CLAIM_ID": CLAIM_ID,
            "CLAIM_MARKER": CLAIM_MARKER,
            "CONSUMED_MARKER": CONSUMED_MARKER,
            "GITHUB_RUN_ID": "1001",
            "SOURCE_COMMIT_SHA": SOURCE_SHA,
            "AUTHORIZED_PR_HEAD": "b" * 40,
            "SOURCE_PATH": "evidence/local-model-attempt-2-validated.html",
            "SOURCE_SHA256": ARTIFACT_SHA,
            "TARGET_BRANCH": "feature/ados-poc-002",
            "EXPECTED_TARGET_SHA": EXPECTED,
            "TARGET_PATH": "daily-practice.html",
            "CLAIM_COMMENT_ID": "9",
            "CLAIM_RUN_ID": "1001",
            "COMMIT_SHA": COMMIT,
        }
        claim_body = self._writer_record(
            "Reconcile authorization, claim and target state", "claim_body", env
        )
        self.assertIn("\nRun: 1001\n", claim_body)
        self.assertNotIn("\\n", claim_body)
        state = base_state()
        state["comments"] = [{"id": 9, "user": {"login": "github-actions[bot]"}, "body": claim_body}]
        state["prior_run"] = run_record(conclusion="failure")
        self.assertEqual(POLICY["decide"](state)["mode"], "resume_same_claim")

        consumed_body = self._writer_record("Record completed handoff", "body", env)
        self.assertIn("\nClaim run: 1001\n", consumed_body)
        self.assertNotIn("\\n", consumed_body)
        state["comments"] = [
            {"id": 9, "user": {"login": "github-actions[bot]"}, "body": claim_body},
            {"id": 10, "user": {"login": "github-actions[bot]"}, "body": consumed_body},
        ]
        state["prior_run"] = run_record(conclusion="success")
        state.update({
            "target_head": COMMIT,
            "target_file_exists": True,
            "target_parent": EXPECTED,
            "target_parent_count": 1,
            "target_digest": ARTIFACT_SHA,
            "target_mode": "100644",
            "target_type": "blob",
            "target_files": ["daily-practice.html"],
        })
        self.assertEqual(POLICY["decide"](state)["mode"], "already_finalized")


if __name__ == "__main__":
    unittest.main()
