#!/usr/bin/env python3
"""Reviewable policy tests for .github/workflows/ados-poc-002.yml.

Uses only Python's standard library and mock gh/git commands. It never makes
provider API calls, reads credentials, or executes OpenCode.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/ados-poc-002.yml"
SOURCE = WORKFLOW.read_text(encoding="utf-8")
MARKER = "<!-- ADOS-POC-002-RESUME-CLAIM -->"


def step_run(name: str) -> str:
    marker = f"      - name: {name}\n"
    start = SOURCE.index(marker)
    next_step = SOURCE.find("      - name: ", start + len(marker))
    section = SOURCE[start: next_step if next_step >= 0 else len(SOURCE)]
    run = re.search(r"(?m)^        run: \|\n", section)
    if not run:
        raise AssertionError(f"step has no literal run block: {name}")
    body = section[run.end():]
    lines = []
    for line in body.splitlines():
        if line and (len(line) - len(line.lstrip(" ")) < 10):
            break
        lines.append(line[10:] if line.startswith("          ") else line)
    return "\n".join(lines) + "\n"


class WorkflowPolicyTests(unittest.TestCase):
    def test_issue_open_event_is_pinned_to_issue_9(self):
        job_if = SOURCE.split("  execute:\n", 1)[1].split("    runs-on:", 1)[0]
        self.assertIn("github.event.issue.number == 9", job_if)
        self.assertIn("github.event.action == 'opened'", job_if)
        self.assertIn("inputs.issue_number == '9'", job_if)
        self.assertIn("inputs.resume_failed_run == '41376777156'", job_if)

    def test_resume_claim_is_durable_and_fully_paginated(self):
        guard = step_run("Enforce one-time resume and task scope")
        self.assertIn("gh api --paginate --slurp", guard)
        self.assertIn("issues/9/comments?per_page=100", guard)
        self.assertIn(MARKER, guard)
        self.assertIn('gh api "repos/$GITHUB_REPOSITORY/issues/9/comments" --method POST', guard)
        self.assertIn('user.login == "github-actions[bot]"', guard)
        self.assertNotIn("gh run list", guard)
        self.assertNotIn("--limit 100", guard)

    def test_only_explicit_free_route_is_configured(self):
        self.assertIn("--model openrouter/free", SOURCE)
        self.assertNotIn("GEMINI_API_KEY", SOURCE)
        self.assertNotIn("GOOGLE_GENERATIVE_AI_API_KEY", SOURCE)
        self.assertNotIn("google/gemini", SOURCE)
        model_lines = re.findall(r"(?m)^\s+opencode run .*--model [^\\]+", SOURCE)
        self.assertTrue(model_lines)
        self.assertTrue(all("--model openrouter/free" in line for line in model_lines))

    def test_no_text_search_or_exit_code_sentinel_controls_failover(self):
        self.assertNotIn("grep -Eiq", SOURCE)
        self.assertNotIn("return 75", SOURCE)
        self.assertNotIn("PROVIDER_UNAVAILABLE", SOURCE)
        self.assertNotIn("failover", SOURCE.lower())

    def test_all_workflow_shell_blocks_parse(self):
        names = re.findall(r"(?m)^      - name: (.+)$", SOURCE)
        self.assertTrue(names)
        for name in names:
            if re.search(rf"(?m)^      - name: {re.escape(name)}$", SOURCE):
                try:
                    script = step_run(name)
                except (AssertionError, ValueError):
                    continue
                result = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")

    def test_functional_checker_runs_before_commit_and_uses_workflow_sha(self):
        output_gate = step_run("Enforce single-file output")
        self.assertIn("node .ados-policy/scripts/check_daily_practice.js daily-practice.html", output_gate)
        self.assertLess(output_gate.index("check_daily_practice.js"), SOURCE.index("      - name: Preserve commit checkpoint"))
        self.assertIn("ref: \${{ github.workflow_sha }}", SOURCE)
        self.assertIn("id=completeButton", SOURCE)
        self.assertIn("id=completionStatus", SOURCE)

    def run_page_check(self, html: str):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "daily-practice.html"
            page.write_text(html, encoding="utf-8")
            return subprocess.run(
                ["node", str(ROOT / "scripts/check_daily_practice.js"), str(page)],
                text=True, capture_output=True, timeout=5,
            )

    def test_valid_page_passes_initial_click_and_reload_behavior(self):
        html = """<!doctype html>
<html><body>
<h1>Daily Practice</h1>
<button id="completeButton" type="button">Mark Complete</button>
<p id="completionStatus" role="status" aria-live="polite">Not completed</p>
<script>
document.getElementById("completeButton").addEventListener("click", () => {
  document.getElementById("completionStatus").textContent = "Completed";
});
</script>
</body></html>"""
        result = self.run_page_check(html)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("initial state, click transition, and reset on reload", result.stdout)

    def test_noop_page_with_required_strings_is_rejected(self):
        html = """<!doctype html>
<html><body>
<h1>Daily Practice</h1>
<button id="completeButton" type="button">Mark Complete</button>
<p id="completionStatus" role="status" aria-live="polite">Not completed</p>
<p>Completed</p>
<script>// no-op: required words are present, but the button does nothing</script>
</body></html>"""
        result = self.run_page_check(html)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("clicking Mark Complete did not change status", result.stderr)

    def test_persisted_completion_is_rejected_after_reload(self):
        html = """<!doctype html>
<html><body>
<h1>Daily Practice</h1>
<button id="completeButton" type="button">Mark Complete</button>
<p id="completionStatus" role="status" aria-live="polite">Not completed</p>
<script>
if (localStorage.getItem("completed") === "yes") {
  document.getElementById("completionStatus").textContent = "Completed";
}
document.getElementById("completeButton").addEventListener("click", () => {
  localStorage.setItem("completed", "yes");
  document.getElementById("completionStatus").textContent = "Completed";
});
</script>
</body></html>"""
        result = self.run_page_check(html)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("did not reset after reload", result.stderr)

    def run_guard(self, *, issue_number="9", claim_count="0", run_record="41376777156\tissues\tfailure"):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        bin_dir = root / "bin"
        bin_dir.mkdir()
        gh_log = root / "gh.log"

        gh = bin_dir / "gh"
        gh.write_text("""#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$GH_LOG"
joined="$*"
if [[ "$1" == "pr" && "$2" == "list" ]]; then
  echo 0
elif [[ "$joined" == *"actions/runs/41376777156"* ]]; then
  printf '%s\\n' "$MOCK_RUN_RECORD"
elif [[ "$joined" == *"issues/9/comments?per_page=100"* ]]; then
  echo "$MOCK_CLAIM_COUNT"
elif [[ "$joined" == *"issues/9/comments"* && "$joined" == *"--method POST"* ]]; then
  echo '{"id":1}'
else
  echo "unexpected gh command" >&2
  exit 90
fi
""", encoding="utf-8")
        gh.chmod(0o755)

        git = bin_dir / "git"
        git.write_text("""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "status" && "$2" == "--porcelain" ]]; then
  exit 0
fi
echo "unexpected git command" >&2
exit 91
""", encoding="utf-8")
        git.chmod(0o755)

        env = os.environ.copy()
        env.update({
            "PATH": f"{bin_dir}:{env['PATH']}",
            "GH_LOG": str(gh_log),
            "MOCK_CLAIM_COUNT": claim_count,
            "MOCK_RUN_RECORD": run_record,
            "TASK_ISSUE_NUMBER": issue_number,
            "RESUME_FAILED_RUN": "41376777156",
            "GITHUB_EVENT_NAME": "workflow_dispatch",
            "GITHUB_REPOSITORY": "chinkk-git/aaria-ados-poc",
            "GITHUB_REPOSITORY_OWNER": "chinkk-git",
            "GITHUB_RUN_ID": "99999999999",
        })
        result = subprocess.run(["bash", "-c", step_run("Enforce one-time resume and task scope")],
                                env=env, text=True, capture_output=True)
        calls = gh_log.read_text(encoding="utf-8") if gh_log.exists() else ""
        temp.cleanup()
        return result, calls

    def test_valid_resume_records_one_durable_claim(self):
        result, calls = self.run_guard()
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        self.assertIn("--paginate --slurp", calls)
        self.assertIn("issues/9/comments?per_page=100", calls)
        self.assertIn("--method POST", calls)
        self.assertIn(MARKER, calls)

    def test_existing_claim_rejects_duplicate_before_post(self):
        result, calls = self.run_guard(claim_count="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("one-time resume claim already exists", result.stdout)
        self.assertNotIn("--method POST", calls)

    def test_wrong_issue_rejected_before_github_writes(self):
        result, calls = self.run_guard(issue_number="10")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("task issue must be #9", result.stdout)
        self.assertNotIn("--method POST", calls)

    def test_unverified_prior_run_rejected_before_claim(self):
        result, calls = self.run_guard(run_record="41376777156\\tworkflow_dispatch\\tfailure")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("prior run is not the verified failed issue dispatch", result.stdout)
        self.assertNotIn("--method POST", calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
