import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ados_executor as ae
import run_local_model_once as qwen


class FakeAdapter:
    def __init__(self, name, model, outcome):
        self.name = name
        self.model = model
        self.outcome = outcome
        self.calls = 0

    def execute(self, prompt):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return ae.ExecutionResult(self.name, self.model, self.outcome, 1, 0.01)


def elig(name, model, *, billing=ae.BillingState.VERIFIED_ZERO_COST, approved=True,
         remaining=1, free=True, license_ok=True, runner_ok=True):
    return ae.Eligibility(name, model, billing, "fixture-evidence",
                          approved, remaining, free, license_ok, runner_ok)


def manifest(providers=("local", "gemini", "openrouter"), **kwargs):
    return ae.TaskManifest(
        "ADOS-POC-999", "experiment/python-multiprovider-executor",
        ("daily-practice.html",), tuple(providers), **kwargs
    )


class AdapterResponseTests(unittest.TestCase):
    def test_gemini_adapter_uses_one_mocked_free_model_response(self):
        payload = {
            "candidates": [{"content": {"parts": [{"text": "<html>mock</html>"}]}}],
            "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 8},
        }
        with patch.object(ae, "_http_json", return_value=(200, payload)) as request:
            result = ae.GeminiAdapter(lambda: "mock-only-key").execute("synthetic prompt")
        self.assertEqual((result.provider, result.model, result.request_count),
                         ("gemini", "gemini-3.5-flash-lite", 1))
        self.assertEqual((result.prompt_tokens, result.output_tokens), (12, 8))
        self.assertEqual(request.call_count, 1)
        self.assertNotIn("mock-only-key", repr(result))

    def test_openrouter_adapter_is_pinned_to_free_route(self):
        payload = {"choices": [{"message": {"content": "<html>mock</html>"}}],
                   "usage": {"prompt_tokens": 10, "completion_tokens": 7}}
        with patch.object(ae, "_http_json", return_value=(200, payload)) as request:
            adapter = ae.OpenRouterAdapter(lambda: "mock-only-key")
            result = adapter.execute("synthetic prompt")
        self.assertEqual(adapter.model, "openrouter/free")
        self.assertEqual(result.request_count, 1)
        self.assertEqual(request.call_args.args[0], "https://openrouter.ai/api/v1/chat/completions")

    def test_local_adapter_requires_loopback_and_parses_mock_response(self):
        with self.assertRaises(ValueError):
            ae.LocalHTTPAdapter("http://10.0.0.1:8090", "fixture-model")
        payload = {"choices": [{"message": {"content": "<html>local</html>"}}]}
        with patch.object(ae, "_http_json", return_value=(200, payload)) as request:
            result = ae.LocalHTTPAdapter("http://127.0.0.1:8090", "qwen-fixture").execute("prompt")
        self.assertEqual((result.provider, result.model, result.request_count),
                         ("local", "qwen-fixture", 1))
        self.assertTrue(request.call_args.args[0].startswith("http://127.0.0.1:"))


class ErrorClassificationTests(unittest.TestCase):
    def test_google_quota_requires_structured_quota_failure(self):
        quota = {"error": {"status": "RESOURCE_EXHAUSTED", "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": []}
        ]}}
        generic = {"error": {"status": "RESOURCE_EXHAUSTED", "message": "retry later"}}
        self.assertEqual(ae.classify_google_error(429, quota)[0], ae.FailureKind.QUOTA_EXHAUSTED)
        self.assertEqual(ae.classify_google_error(429, generic)[0], ae.FailureKind.RATE_LIMIT_UNKNOWN)
        self.assertEqual(ae.classify_google_error(403, {"error": {"status": "PERMISSION_DENIED"}})[0],
                         ae.FailureKind.AUTHENTICATION)

    def test_openrouter_quota_requires_exact_structured_code(self):
        quota = {"error": {"code": "free-models-per-day", "message": "ignored"}}
        generic = {"error": {"code": 429, "message": "rate limit"}}
        self.assertEqual(ae.classify_openrouter_error(429, quota)[0], ae.FailureKind.QUOTA_EXHAUSTED)
        self.assertEqual(ae.classify_openrouter_error(429, generic)[0], ae.FailureKind.RATE_LIMIT_UNKNOWN)
        self.assertEqual(ae.classify_openrouter_error(401, {"error": {"code": "401"}})[0],
                         ae.FailureKind.AUTHENTICATION)


class DispatchValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest_dir = self.root / "tasks" / "issues"
        self.manifest_dir.mkdir(parents=True)
        self.issue = 999
        self.sha = "a" * 40
        self.event_key = "b" * 64
        self.payload = {
            "issue_number": self.issue,
            "authorized_actors": ["chinkk-git"],
            "repository": "chinkk-git/aaria-ados-poc",
            "task_id": "ADOS-POC-999",
            "branch": "experiment/ados-e01-synthetic",
            "expected_base_sha": self.sha,
            "permitted_paths": ["daily-practice.html"],
            "task_prompt": "Synthetic fixture only. No external model call.",
            "acceptance_criteria": ["Fake worker observes the approved task."],
            "cost_ceiling_usd": 0.0,
            "provider": "local",
            "provider_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            "provider_model_revision": "f86cb2c1fa58255f8052cc32aeede1b7482d4361",
            "provider_model_sha256": "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046",
            "inference_authorization_id": "AUTH-SYNTHETIC-999",
            "max_provider_calls": 1,
            "max_workflow_recoveries": 2,
            "enabled": True,
        }
        self.path = self.manifest_dir / f"{self.issue}.json"
        self.write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self, payload=None):
        self.path.write_text(json.dumps(payload or self.payload), encoding="utf-8")

    def resolve(self, **kwargs):
        options = {
            "actor": "chinkk-git",
            "repository": "chinkk-git/aaria-ados-poc",
            "base_sha": self.sha,
            "event_key": self.event_key,
        }
        options.update(kwargs)
        return ae.load_trusted_dispatch_manifest(self.root, self.issue, **options)

    def test_approved_synthetic_issue_resolves_trusted_manifest_once_without_side_effects(self):
        class FakeWorkerAdapter:
            def __init__(self):
                self.calls = 0

            def execute(self, manifest):
                self.calls += 1
                return manifest.task_id

        worker = FakeWorkerAdapter()
        before = self.path.read_bytes()
        with patch.object(ae, "_http_json") as provider_call, \
             patch.object(ae.subprocess, "run") as repository_write:
            manifest = self.resolve()
            self.assertEqual(worker.execute(manifest), "ADOS-POC-999")
        self.assertEqual(worker.calls, 1)
        provider_call.assert_not_called()
        repository_write.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.manifest_dir.iterdir()), [self.path])

    def test_unknown_issue_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dispatch_manifest_not_allowlisted"):
            ae.load_trusted_dispatch_manifest(
                self.root, 998, actor="chinkk-git", repository="chinkk-git/aaria-ados-poc",
                base_sha=self.sha, event_key=self.event_key,
            )

    def test_actor_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dispatch_actor_mismatch"):
            self.resolve(actor="unapproved-user")

    def test_branch_and_path_violations_are_rejected(self):
        payload = dict(self.payload)
        payload["branch"] = "main"
        self.write_manifest(payload)
        with self.assertRaisesRegex(ValueError, "invalid_branch"):
            self.resolve()
        payload = dict(self.payload)
        payload["permitted_paths"] = ["../README.md"]
        self.write_manifest(payload)
        with self.assertRaisesRegex(ae.UnauthorizedPath, "unsafe_path"):
            self.resolve()
        self.write_manifest()

    def test_stale_base_sha_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dispatch_stale_base_sha"):
            self.resolve(base_sha="c" * 40)

    def test_repository_task_binding_cost_and_attempt_limits_are_enforced(self):
        with self.assertRaisesRegex(ValueError, "dispatch_repository_mismatch"):
            self.resolve(repository="other/repository")
        for field, value in (("task_id", "ADOS-POC-998"),
                             ("task_prompt", "  "),
                             ("acceptance_criteria", []),
                             ("cost_ceiling_usd", 0.01),
                             ("provider_model", "paid/model"),
                             ("max_provider_calls", 2), ("max_workflow_recoveries", 4)):
            with self.subTest(field=field):
                payload = dict(self.payload)
                payload[field] = value
                self.write_manifest(payload)
                with self.assertRaises(ValueError):
                    self.resolve()
        self.write_manifest()

    def test_allowlist_rejects_a_second_task_for_the_same_branch(self):
        other = dict(self.payload)
        other.update({"issue_number": 998, "task_id": "ADOS-POC-998"})
        (self.manifest_dir / "998.json").write_text(json.dumps(other), encoding="utf-8")
        with self.assertRaisesRegex(ae.DuplicateExecution, "dispatch_branch_or_task_already_allowlisted"):
            self.resolve()

    def test_duplicate_event_key_is_rejected(self):
        with self.assertRaisesRegex(ae.DuplicateExecution, "dispatch_duplicate_event"):
            self.resolve(seen_event_keys=[self.event_key])

    def test_unconsumed_checkpoint_can_resume_only_under_original_run(self):
        resumed = ae.load_trusted_dispatch_manifest(
            self.root, self.issue, actor="chinkk-git", repository="chinkk-git/aaria-ados-poc",
            base_sha=self.sha, event_key=self.event_key,
            seen_event_keys=[self.event_key], resumable_event_keys=[self.event_key],
        )
        self.assertEqual(resumed.task_id, "ADOS-POC-999")
        with self.assertRaises(ae.DuplicateExecution):
            self.resolve(seen_event_keys=[self.event_key])

    def test_consumed_claim_cannot_be_reused(self):
        with self.assertRaisesRegex(ae.DuplicateExecution, "dispatch_claim_consumed"):
            self.resolve(consumed_task_ids=["ADOS-POC-999"])

    def test_reconciliation_drains_unclaimed_tasks_without_concurrency_eviction(self):
        second = dict(self.payload, issue_number=998, task_id="ADOS-POC-998",
                      branch="experiment/ados-e01-second")
        third = dict(self.payload, issue_number=997, task_id="ADOS-POC-997",
                     branch="experiment/ados-e01-third")
        issues = [
            {"number": 999, "created_at": "2026-01-03T00:00:00Z", "user": {"login": "chinkk-git"}},
            {"number": 998, "created_at": "2026-01-02T00:00:00Z", "user": {"login": "chinkk-git"}},
            {"number": 997, "created_at": "2026-01-01T00:00:00Z", "user": {"login": "chinkk-git"}},
        ]
        manifests = {999: self.payload, 998: second, 997: third}
        first = ae.select_reconciliation_candidate(issues, manifests)
        self.assertEqual(first["issue_number"], 997)
        second_candidate = ae.select_reconciliation_candidate(
            issues, manifests, claimed_issue_numbers=[997], leased_branches=["experiment/ados-e01-second"])
        self.assertEqual(second_candidate["issue_number"], 999)
        event_only = ae.select_reconciliation_candidate(issues, manifests,
            event_issue_number=998, event_actor="chinkk-git")
        self.assertEqual(event_only["issue_number"], 998)
        self.assertIsNone(ae.select_reconciliation_candidate(issues, manifests,
            claimed_issue_numbers=[997, 998, 999]))

    def test_same_synthetic_event_cannot_execute_twice(self):
        class FakeWorkerAdapter:
            calls = 0

            def execute(self, manifest):
                self.calls += 1

        worker = FakeWorkerAdapter()
        ledger = {"events": set(), "tasks": set()}
        manifest = self.resolve()
        worker.execute(manifest)
        ledger["events"].add(self.event_key)
        ledger["tasks"].add(manifest.task_id)
        with self.assertRaises(ae.DuplicateExecution):
            self.resolve(seen_event_keys=ledger["events"], consumed_task_ids=ledger["tasks"])
        self.assertEqual(worker.calls, 1)


class RemediationPolicyTests(unittest.TestCase):
    def test_checkpoint_recovery_requires_exact_parent_and_full_scoped_diff(self):
        allowed = ["daily-practice.html"]
        files = {"daily-practice.html": b"fixture"}
        ae.validate_checkpoint_recovery(
            parent_shas=["a" * 40], expected_base_sha="a" * 40,
            changed_paths=["daily-practice.html"], artifact_paths=["daily-practice.html"],
            committed_files=files, artifact_files=files, allowed_paths=allowed,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_parent_mismatch"):
            ae.validate_checkpoint_recovery(
                parent_shas=["a" * 40, "c" * 40], expected_base_sha="a" * 40,
                changed_paths=["daily-practice.html"], artifact_paths=["daily-practice.html"],
                committed_files=files, artifact_files=files, allowed_paths=allowed,
            )
        with self.assertRaises(ae.UnauthorizedPath):
            ae.validate_checkpoint_recovery(
                parent_shas=["a" * 40], expected_base_sha="a" * 40,
                changed_paths=["daily-practice.html", "README.md"],
                artifact_paths=["daily-practice.html", "README.md"],
                committed_files={**files, "README.md": b"intrusion"},
                artifact_files={**files, "README.md": b"intrusion"}, allowed_paths=allowed,
            )

    def test_recovery_distinguishes_pre_call_uncertain_call_and_artifact_only(self):
        args = dict(max_provider_calls=1, max_recoveries=2, recovery_number=1)
        self.assertEqual(ae.recovery_action(provider_started=False, artifact_available=False,
                         provider_calls=0, **args), "EXECUTE_ONCE")
        self.assertEqual(ae.recovery_action(provider_started=True, artifact_available=False,
                         provider_calls=1, **args), "BLOCKED_MISSING_WORKER_ARTIFACT")
        self.assertEqual(ae.recovery_action(provider_started=True, artifact_available=True,
                         provider_calls=1, **args), "ARTIFACT_ONLY")
        self.assertEqual(ae.recovery_action(provider_started=False, artifact_available=False,
                         provider_calls=0, **{**args, "recovery_number": 3}), "BLOCKED_RECOVERY_LIMIT")

    def test_claim_write_failure_never_creates_branch_and_lease_retry_is_idempotent(self):
        base = "a" * 40
        state = {"claim": False, "branch_sha": None, "create_calls": 0, "claim_calls": 0}

        def read_branch():
            return state["branch_sha"]

        def write_claim_failure():
            state["claim_calls"] += 1
            raise RuntimeError("simulated claim comment failure")

        def create_branch(sha):
            state["create_calls"] += 1
            state["branch_sha"] = sha

        with self.assertRaisesRegex(RuntimeError, "claim comment failure"):
            ae.claim_then_acquire_dispatch_branch_lease(
                expected_base_sha=base, same_run_claim=False, read_branch_sha=read_branch,
                validate_provider=lambda: None, write_claim=write_claim_failure,
                create_branch=create_branch)
        self.assertIsNone(state["branch_sha"])
        self.assertEqual(state["create_calls"], 0)

        def durable_claim():
            state["claim_calls"] += 1
            state["claim"] = True

        def create_then_interrupt(sha):
            state["create_calls"] += 1
            state["branch_sha"] = sha
            raise RuntimeError("simulated interruption after branch creation")

        with self.assertRaisesRegex(RuntimeError, "after branch creation"):
            ae.claim_then_acquire_dispatch_branch_lease(
                expected_base_sha=base, same_run_claim=False, read_branch_sha=read_branch,
                validate_provider=lambda: None, write_claim=durable_claim,
                create_branch=create_then_interrupt)
        self.assertTrue(state["claim"])
        self.assertEqual(state["branch_sha"], base)
        create_calls_before_resume = state["create_calls"]
        action = ae.claim_then_acquire_dispatch_branch_lease(
            expected_base_sha=base, same_run_claim=True, read_branch_sha=read_branch,
            validate_provider=lambda: self.fail("lease recovery must not repeat inference gating"),
            write_claim=lambda: self.fail("lease recovery must not duplicate the claim comment"),
            create_branch=create_branch)
        self.assertEqual(action, "RECOVERED")
        self.assertEqual(state["create_calls"], create_calls_before_resume)
        self.assertEqual(state["claim_calls"], 2)

    def test_existing_branch_cannot_be_reused_by_a_new_claim(self):
        calls = []
        with self.assertRaisesRegex(ae.DuplicateExecution, "dispatch_branch_preexisting"):
            ae.claim_then_acquire_dispatch_branch_lease(
                expected_base_sha="a" * 40, same_run_claim=False,
                read_branch_sha=lambda: "a" * 40,
                validate_provider=lambda: calls.append("validate"),
                write_claim=lambda: calls.append("claim"),
                create_branch=lambda _sha: calls.append("create"),
            )
        self.assertEqual(calls, [])

    def test_local_qwen_eligibility_expires_and_is_rejected_at_point_of_use(self):
        payload = {
            "issue_number": 999, "authorized_actors": ["chinkk-git"],
            "repository": "chinkk-git/aaria-ados-poc", "task_id": "ADOS-POC-999",
            "branch": "experiment/ados-e01-synthetic", "expected_base_sha": "a" * 40,
            "permitted_paths": ["daily-practice.html"], "task_prompt": "Synthetic task",
            "acceptance_criteria": ["No inference"], "cost_ceiling_usd": 0.0,
            "provider": "local", "provider_model": ae.LOCAL_QWEN_MODEL,
            "provider_model_revision": ae.LOCAL_QWEN_REVISION,
            "provider_model_sha256": ae.LOCAL_QWEN_SHA256,
            "inference_authorization_id": "AUTH-SYNTHETIC-999",
            "max_provider_calls": 1, "max_workflow_recoveries": 2, "enabled": True,
        }
        manifest = ae.parse_dispatch_manifest(payload)
        record = {"provider": "local", "model": ae.LOCAL_QWEN_MODEL,
                  "eligibility": "VERIFIED_ZERO_COST", "cost_authorized_usd": 0.0,
                  "quota_status": "NOT_APPLICABLE", "last_verified_at": "1970-01-01T00:16:40Z"}
        ae.validate_local_qwen_record(record, manifest, now_epoch=1500)
        with self.assertRaisesRegex(ae.ExecutorError, "local_qwen_eligibility_stale"):
            ae.validate_local_qwen_record(record, manifest, now_epoch=5000)
        with self.assertRaisesRegex(ae.ExecutorError, "local_qwen_eligibility_stale"):
            ae.validate_local_qwen_record(record, manifest, now_epoch=900)
        with self.assertRaisesRegex(ae.ExecutorError, "local_qwen_eligibility_unverified"):
            ae.validate_local_qwen_record({**record, "eligibility": "DISABLED"}, manifest, now_epoch=1500)

    def test_provider_record_requires_verified_zero_cost_and_available_quota(self):
        good = {"provider": "fixture", "model": "free-fixture", "eligibility": "VERIFIED_ZERO_COST",
                "quota_status": "AVAILABLE", "quota_remaining": 2,
                "cost_authorized_usd": 0.0, "last_verified_epoch": 1000}
        ae.validate_provider_record(good, provider="fixture", model="free-fixture", now_epoch=1100)
        for key, value in (("eligibility", "BILLING_UNVERIFIED"),
                           ("quota_status", "EXHAUSTED"), ("quota_remaining", 0)):
            invalid = dict(good); invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ae.ExecutorError):
                ae.validate_provider_record(invalid, provider="fixture", model="free-fixture", now_epoch=1100)
        stale = dict(good); stale["last_verified_epoch"] = 1
        with self.assertRaisesRegex(ae.ExecutorError, "provider_eligibility_stale"):
            ae.validate_provider_record(stale, provider="fixture", model="free-fixture", now_epoch=5000)

    def test_dispatch_manifest_selects_only_the_pinned_local_qwen_route(self):
        payload = {
            "issue_number": 999, "authorized_actors": ["chinkk-git"],
            "repository": "chinkk-git/aaria-ados-poc", "task_id": "ADOS-POC-999",
            "branch": "experiment/ados-e01-synthetic", "expected_base_sha": "a" * 40,
            "permitted_paths": ["daily-practice.html"], "task_prompt": "Synthetic task",
            "acceptance_criteria": ["Use fake worker"], "cost_ceiling_usd": 0.0,
            "provider": "local", "provider_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            "provider_model_revision": "f86cb2c1fa58255f8052cc32aeede1b7482d4361",
            "provider_model_sha256": "cc324af070c2ecbfd324a30884d2f951a7ff756aba85cb811a6ec436933bb046",
            "inference_authorization_id": "AUTH-SYNTHETIC-999",
            "max_provider_calls": 1, "max_workflow_recoveries": 2, "enabled": True,
        }
        parsed = ae.parse_dispatch_manifest(payload)
        status = {"provider": "local", "model": parsed.provider_model,
                  "eligibility": "VERIFIED_ZERO_COST", "cost_authorized_usd": 0.0,
                  "quota_status": "NOT_APPLICABLE",
                  "last_verified_at": "2026-01-01T00:00:00Z"}
        ae.validate_local_qwen_record(status, parsed, now_epoch=1767225600)
        with self.assertRaisesRegex(ValueError, "dispatch_local_qwen_route_unapproved"):
            ae.parse_dispatch_manifest({**payload, "provider_model_revision": "0" * 40})
        with self.assertRaisesRegex(ValueError, "dispatch_inference_authorization_required"):
            ae.parse_dispatch_manifest({**payload, "inference_authorization_id": None})

    def test_local_qwen_prompt_and_output_are_one_file_bounded_and_scoped(self):
        prompt = ae.build_local_qwen_prompt(
            task_prompt="Update the page safely.",
            acceptance_criteria=["Accessible status text"],
            target_path="daily-practice.html", existing_content="<main>old</main>")
        self.assertIn("Target file: daily-practice.html", prompt)
        self.assertIn("<ADOS_FILE_CONTENT>", prompt)
        self.assertEqual(ae.validate_local_qwen_output(
            returned_path="daily-practice.html", expected_path="daily-practice.html",
            content="<main>new</main>"), b"<main>new</main>")
        with self.assertRaises(ae.UnauthorizedPath):
            ae.validate_local_qwen_output(returned_path="README.md", expected_path="daily-practice.html",
                                          content="no")
        with self.assertRaisesRegex(ValueError, "local_qwen_existing_file_too_large"):
            ae.build_local_qwen_prompt(task_prompt="change", acceptance_criteria=["ok"],
                target_path="daily-practice.html", existing_content="x" * (ae.MAX_LOCAL_QWEN_BASE_FILE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "local_qwen_prompt_too_large"):
            ae.build_local_qwen_prompt(task_prompt="x" * ae.MAX_LOCAL_QWEN_PROMPT_BYTES,
                acceptance_criteria=["ok"], target_path="daily-practice.html", existing_content="")
        with self.assertRaisesRegex(ValueError, "local_qwen_output_invalid"):
            ae.validate_local_qwen_output(returned_path="daily-practice.html", expected_path="daily-practice.html",
                                          content="x" * (ae.MAX_LOCAL_QWEN_OUTPUT_BYTES + 1))

    def test_existing_runner_prompt_override_is_bounded_and_default_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompt_path = Path(tmp) / "prompt.txt"
            self.assertIn("Daily Practice", qwen.load_prompt())
            prompt_path.write_text("synthetic prompt", encoding="utf-8")
            self.assertEqual(qwen.load_prompt(prompt_path), "synthetic prompt")
            prompt_path.write_bytes(b"x" * (qwen.MAX_PROMPT_BYTES + 1))
            with self.assertRaisesRegex(RuntimeError, "local_prompt_size_invalid"):
                qwen.load_prompt(prompt_path)
            prompt_path.write_bytes(b"  ")
            with self.assertRaisesRegex(RuntimeError, "local_prompt_empty"):
                qwen.load_prompt(prompt_path)

    def test_local_qwen_server_environment_excludes_inherited_ci_credentials(self):
        environment = qwen.isolated_server_environment(
            home=Path("/tmp/fake-home"),
            path="/usr/bin:/bin",
        )
        self.assertEqual(set(environment), {"PATH", "HOME", "LANG", "CUDA_VISIBLE_DEVICES"})
        self.assertNotIn("GITHUB_TOKEN", environment)
        self.assertNotIn("ACTIONS_RUNTIME_TOKEN", environment)

    def test_live_local_provider_status_is_blocked_until_separately_authorized(self):
        status = json.loads((Path(__file__).resolve().parents[1] / "providers" / "status.json").read_text())
        payload = {
            "issue_number": 999, "authorized_actors": ["chinkk-git"],
            "repository": "chinkk-git/aaria-ados-poc", "task_id": "ADOS-POC-999",
            "branch": "experiment/ados-e01-synthetic", "expected_base_sha": "a" * 40,
            "permitted_paths": ["daily-practice.html"], "task_prompt": "Synthetic task",
            "acceptance_criteria": ["No inference"], "cost_ceiling_usd": 0.0,
            "provider": "local", "provider_model": ae.LOCAL_QWEN_MODEL,
            "provider_model_revision": ae.LOCAL_QWEN_REVISION,
            "provider_model_sha256": ae.LOCAL_QWEN_SHA256,
            "inference_authorization_id": "AUTH-SYNTHETIC-999",
            "max_provider_calls": 1, "max_workflow_recoveries": 2, "enabled": True,
        }
        manifest = ae.parse_dispatch_manifest(payload)
        with self.assertRaisesRegex(ae.ExecutorError, "local_qwen_eligibility_unverified"):
            ae.validate_local_qwen_record(status["local"], manifest)
        self.assertEqual(status["local"]["eligibility"], "DISABLED")

    def test_repository_provider_status_blocks_current_exhausted_or_unverified_routes(self):
        status = json.loads((Path(__file__).resolve().parents[1] / "providers" / "status.json").read_text())
        with self.assertRaises(ae.ExecutorError):
            ae.validate_provider_record(status["openrouter"], provider="openrouter",
                                        model="openrouter/free", now_epoch=2_000_000_000)
        with self.assertRaises(ae.ExecutorError):
            ae.validate_provider_record(status["local"], provider="local",
                                        model=status["local"]["model"], now_epoch=2_000_000_000)

    def test_durable_ledger_requires_bot_and_associated_main_workflow_run(self):
        event = "b" * 64; run_id = "123"
        marker = f"<!-- ADOS_DISPATCH_CLAIM task=ADOS-POC-999 event={event} run={run_id} -->"
        comment = {"body": marker, "user_login": "github-actions[bot]"}
        run = {"id": 123, "repository": "chinkk-git/aaria-ados-poc",
               "workflow_path": ".github/workflows/ados-dispatch.yml", "event": "issues",
               "head_branch": "main", "head_sha": "a" * 40}
        ledger = ae.dispatch_ledger([comment], {run_id: run}, repository="chinkk-git/aaria-ados-poc",
                                    expected_base_sha="a" * 40)
        self.assertEqual(ledger[2][event], run_id)
        consumed = {"body": f"<!-- ADOS_DISPATCH_CONSUMED task=ADOS-POC-999 event={event} run={run_id} -->",
                    "user_login": "github-actions[bot]"}
        with_claim_and_consumption = ae.dispatch_ledger([comment, consumed], {run_id: run},
            repository="chinkk-git/aaria-ados-poc", expected_base_sha="a" * 40)
        self.assertIn("ADOS-POC-999", with_claim_and_consumption[1])
        forged = dict(comment); forged["user_login"] = "attacker"
        with self.assertRaisesRegex(ValueError, "dispatch_ledger_marker_forged"):
            ae.dispatch_ledger([forged], {run_id: run}, repository="chinkk-git/aaria-ados-poc",
                               expected_base_sha="a" * 40)
        bad_run = dict(run); bad_run["head_sha"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "dispatch_ledger_run_unverified"):
            ae.dispatch_ledger([comment], {run_id: bad_run}, repository="chinkk-git/aaria-ados-poc",
                               expected_base_sha="a" * 40)
        malformed = dict(comment); malformed["body"] += " extra"
        with self.assertRaisesRegex(ValueError, "dispatch_ledger_marker_malformed"):
            ae.dispatch_ledger([malformed], {run_id: run}, repository="chinkk-git/aaria-ados-poc",
                               expected_base_sha="a" * 40)
        # Historical Issue #9 markers are outside this parser's namespace.
        self.assertEqual(ae.dispatch_ledger([{"body": "ADOS_CLAIM historical", "user_login": "user"}],
                         {}, repository="chinkk-git/aaria-ados-poc", expected_base_sha="a" * 40),
                         (set(), set(), {}, set()))


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = ae.CheckpointStore(self.root / "state")

    def tearDown(self):
        self.tmp.cleanup()

    def execute_one(self, name, model, output):
        record = ae.TaskRecord(manifest((name,)))
        registry = ae.ProviderRegistry()
        adapter = FakeAdapter(name, model, output)
        registry.register(adapter, owner_approved=True)
        proof = elig(name, model, license_ok=(name == "local" or True),
                     runner_ok=(name == "local" or True))
        result = ae.Executor(record, registry, {name: proof}, self.store).execute("synthetic")
        return record, adapter, result

    def test_local_gemini_and_openrouter_success(self):
        for name, model in (
            ("local", "qwen-fixture"),
            ("gemini", "gemini-3.5-flash-lite"),
            ("openrouter", "openrouter/free"),
        ):
            with self.subTest(provider=name):
                record, adapter, result = self.execute_one(name, model, "generated fixture")
                self.assertEqual(result.provider, name)
                self.assertEqual(adapter.calls, 1)
                self.assertEqual(record.state, ae.State.BUILDING)
                self.assertEqual(record.attempts[-1].outcome, "success")
                saved = self.store.read(record.manifest.task_id)
                self.assertEqual(saved["output_sha256"], record.output_sha256)

    def test_quota_failure_switches_only_to_next_eligible_provider(self):
        record = ae.TaskRecord(manifest())
        registry = ae.ProviderRegistry()
        local = FakeAdapter("local", "qwen", ae.ProviderFailure(
            "local", ae.FailureKind.QUOTA_EXHAUSTED, outcome_known=True))
        gemini = FakeAdapter("gemini", "gemini-3.5-flash-lite", "valid page")
        openrouter = FakeAdapter("openrouter", "openrouter/free", "should not run")
        for adapter in (local, gemini, openrouter):
            registry.register(adapter, owner_approved=True)
        proofs = {
            "local": elig("local", "qwen"),
            "gemini": elig("gemini", "gemini-3.5-flash-lite"),
            "openrouter": elig("openrouter", "openrouter/free"),
        }
        result = ae.Executor(record, registry, proofs, self.store).execute("synthetic")
        self.assertEqual(result.provider, "gemini")
        self.assertEqual((local.calls, gemini.calls, openrouter.calls), (1, 1, 0))
        self.assertEqual(record.state, ae.State.CHECKPOINTED)
        self.assertEqual(record.attempts[0].error_kind, ae.FailureKind.QUOTA_EXHAUSTED.value)
        self.assertEqual(self.store.read(record.manifest.task_id)["last_successful_step"],
                         "provider_output_received")

    def test_no_eligible_providers_blocks_without_calling(self):
        record = ae.TaskRecord(manifest(("gemini",)))
        registry = ae.ProviderRegistry()
        adapter = FakeAdapter("gemini", "gemini-3.5-flash-lite", "should not run")
        registry.register(adapter, owner_approved=True)
        proof = elig("gemini", "gemini-3.5-flash-lite", billing=ae.BillingState.UNKNOWN)
        with self.assertRaisesRegex(ae.ExecutorError, "no_eligible_provider"):
            ae.Executor(record, registry, {"gemini": proof}, self.store).execute("prompt")
        self.assertEqual(adapter.calls, 0)
        self.assertEqual(record.state, ae.State.BLOCKED)

    def test_auth_failure_does_not_switch(self):
        record = ae.TaskRecord(manifest())
        registry = ae.ProviderRegistry()
        auth = FakeAdapter("local", "qwen", ae.ProviderFailure(
            "local", ae.FailureKind.AUTHENTICATION, outcome_known=True, code="bad_auth"))
        next_provider = FakeAdapter("gemini", "gemini-3.5-flash-lite", "must not run")
        registry.register(auth, owner_approved=True)
        registry.register(next_provider, owner_approved=True)
        proofs = {"local": elig("local", "qwen"),
                  "gemini": elig("gemini", "gemini-3.5-flash-lite")}
        with self.assertRaises(ae.ProviderFailure):
            ae.Executor(record, registry, proofs, self.store).execute("prompt")
        self.assertEqual((auth.calls, next_provider.calls), (1, 0))
        self.assertEqual(record.state, ae.State.BLOCKED)

    def test_ambiguous_timeout_does_not_switch(self):
        record = ae.TaskRecord(manifest())
        registry = ae.ProviderRegistry()
        timed = FakeAdapter("local", "qwen", ae.ProviderFailure(
            "local", ae.FailureKind.TIMEOUT_UNKNOWN, outcome_known=False))
        following = FakeAdapter("gemini", "gemini-3.5-flash-lite", "must not run")
        registry.register(timed, owner_approved=True)
        registry.register(following, owner_approved=True)
        proofs = {"local": elig("local", "qwen"),
                  "gemini": elig("gemini", "gemini-3.5-flash-lite")}
        with self.assertRaises(ae.ProviderFailure):
            ae.Executor(record, registry, proofs, self.store).execute("prompt")
        self.assertEqual((timed.calls, following.calls), (1, 0))

    def test_duplicate_execution_is_rejected(self):
        record, adapter, _ = self.execute_one("local", "qwen", "output")
        registry = ae.ProviderRegistry()
        registry.register(adapter, owner_approved=True)
        executor = ae.Executor(record, registry, {"local": elig("local", "qwen")}, self.store)
        with self.assertRaises(ae.DuplicateExecution):
            executor.execute("same task")
        self.assertEqual(adapter.calls, 1)

    def test_attempt_budget_is_bounded(self):
        task = manifest(max_provider_attempts=1)
        record = ae.TaskRecord(task)
        registry = ae.ProviderRegistry()
        local = FakeAdapter("local", "qwen", ae.ProviderFailure(
            "local", ae.FailureKind.QUOTA_EXHAUSTED, outcome_known=True))
        gemini = FakeAdapter("gemini", "gemini-3.5-flash-lite", "must not run")
        registry.register(local, owner_approved=True)
        registry.register(gemini, owner_approved=True)
        proofs = {"local": elig("local", "qwen"),
                  "gemini": elig("gemini", "gemini-3.5-flash-lite")}
        with self.assertRaisesRegex(ae.ExecutorError, "no_eligible_provider"):
            ae.Executor(record, registry, proofs, self.store).execute("prompt")
        self.assertEqual((local.calls, gemini.calls), (1, 0))
        self.assertEqual(record.state, ae.State.BLOCKED)

    def test_unknown_billing_and_paid_route_never_select(self):
        unknown = elig("openrouter", "openrouter/free", billing=ae.BillingState.UNKNOWN)
        self.assertFalse(unknown.eligible())
        unverified = elig("gemini", "gemini-3.5-flash-lite", free=False)
        self.assertFalse(unverified.eligible())
        registry = ae.ProviderRegistry()
        with self.assertRaises(ae.ExecutorError):
            registry.register(FakeAdapter("openrouter", "paid/model", "no"), owner_approved=True)
        with self.assertRaises(ValueError):
            manifest(cost_ceiling_usd=0.01)

    def test_provider_requires_explicit_owner_approval(self):
        with self.assertRaises(ae.ExecutorError):
            ae.ProviderRegistry().register(FakeAdapter("local", "qwen", "x"), owner_approved=False)

    def test_manifest_json_is_strict_and_cost_is_zero(self):
        path = self.root / "task.json"
        path.write_text(
            '{"task_id":"ADOS-POC-999","branch":"experiment/python-multiprovider-executor",'
            '"scope_paths":["daily-practice.html"],"providers":["local"],'
            '"cost_ceiling_usd":0,"max_provider_attempts":1}', encoding="utf-8"
        )
        loaded = ae.load_manifest(path)
        self.assertEqual(loaded.cost_ceiling_usd, 0.0)
        payload = json.loads(path.read_text())
        payload["unexpected"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest_fields_mismatch"):
            ae.load_manifest(path)

    def test_stuck_lease_operation_fails_closed(self):
        task = manifest()
        lease = ae.TaskLease(self.root / "leases", task, "worker")
        lease.root.mkdir(parents=True)
        lease.operation_path.write_text("interrupted", encoding="utf-8")
        with self.assertRaises(ae.LeaseHeld):
            lease.acquire()
        self.assertFalse(lease.path.exists())

    def test_explicitly_approved_adapter_extension(self):
        adapter = FakeAdapter("approved_lab", "example-free", "result")
        registry = ae.ProviderRegistry(
            approved_extensions=("approved_lab",),
            verified_free_models={"approved_lab": "example-free"},
        )
        registry.register(adapter, owner_approved=True)
        self.assertIs(registry.get("approved_lab"), adapter)
        with self.assertRaises(ae.ExecutorError):
            ae.ProviderRegistry().register(
                FakeAdapter("unapproved_lab", "example-free", "result"),
                owner_approved=True,
            )

    def test_lease_is_exclusive_and_recovery_preserves_old_claim(self):
        task = manifest()
        first = ae.TaskLease(self.root / "leases", task, "worker-1")
        first.acquire()
        concurrent = ae.TaskLease(self.root / "leases", task, "worker-2")
        with self.assertRaises(ae.LeaseHeld):
            concurrent.acquire()
        recovered = ae.TaskLease(self.root / "leases", task, "recovery-worker")
        with self.assertRaises(ae.LeaseHeld):
            recovered.recover(previous_worker_inactive=False)
        recovered.recover(previous_worker_inactive=True)
        retired = list((self.root / "leases").glob("ADOS-POC-999.lease.retired-*"))
        self.assertEqual(len(retired), 1)
        self.assertEqual(recovered.worker, "recovery-worker")
        recovered.release()

    def test_interrupted_checkpoint_is_readable_and_identity_checked(self):
        record = ae.TaskRecord(manifest())
        self.store.write(record)
        checkpoint = self.store.read("ADOS-POC-999")
        self.assertEqual(checkpoint["state"], "READY")
        self.assertEqual(checkpoint["last_successful_step"], "task_validated")
        self.assertIsNone(checkpoint["commit_sha"])

    def test_interruption_after_quota_checkpoint_resumes_next_provider_only(self):
        task = manifest()
        record = ae.TaskRecord(task)
        registry = ae.ProviderRegistry()
        local = FakeAdapter("local", "qwen", ae.ProviderFailure(
            "local", ae.FailureKind.QUOTA_EXHAUSTED, outcome_known=True))
        gemini = FakeAdapter("gemini", "gemini-3.5-flash-lite", "recovered output")
        registry.register(local, owner_approved=True)
        registry.register(gemini, owner_approved=True)
        executor = ae.Executor(
            record, registry,
            {"local": elig("local", "qwen")},
            self.store,
        )
        with self.assertRaisesRegex(ae.ExecutorError, "no_eligible_provider"):
            executor.execute("synthetic")
        checkpoint = self.store.read(task.task_id)
        self.assertEqual(checkpoint["state"], "BLOCKED")
        self.assertEqual(checkpoint["attempts"][0]["error_kind"],
                         ae.FailureKind.QUOTA_EXHAUSTED.value)
        recovered = ae.Executor(
            record, registry,
            {
                "local": elig("local", "qwen"),
                "gemini": elig("gemini", "gemini-3.5-flash-lite"),
            },
            self.store,
        ).execute("synthetic")
        self.assertEqual(recovered.provider, "gemini")
        self.assertEqual((local.calls, gemini.calls), (1, 1))
        self.assertEqual(len(record.action_keys), 2)

    def test_ambiguous_blocked_checkpoint_cannot_be_restarted(self):
        record = ae.TaskRecord(manifest())
        registry = ae.ProviderRegistry()
        timed = FakeAdapter("local", "qwen", ae.ProviderFailure(
            "local", ae.FailureKind.TIMEOUT_UNKNOWN, outcome_known=False))
        registry.register(timed, owner_approved=True)
        with self.assertRaises(ae.ProviderFailure):
            ae.Executor(record, registry, {"local": elig("local", "qwen")}, self.store).execute("prompt")
        with self.assertRaises(ae.InvalidTransition):
            ae.Executor(record, registry, {"local": elig("local", "qwen")}, self.store).execute("prompt")

    def test_unauthorized_paths_and_nonexact_sha_are_rejected(self):
        with self.assertRaises(ae.UnauthorizedPath):
            ae.validate_changed_paths(["daily-practice.html", "../README.md"],
                                      ["daily-practice.html"])
        record, _, result = self.execute_one("local", "qwen", "page")
        executor = ae.Executor(record, ae.ProviderRegistry(),
                               {"local": elig("local", "qwen")}, self.store)
        with self.assertRaisesRegex(ae.ExecutorError, "commit_sha_must_be_exact"):
            executor.mark_verified(result.text, changed_paths=["daily-practice.html"], commit_sha="abc")

    def test_exact_sha_scope_and_review_handoff(self):
        record, _, result = self.execute_one("local", "qwen", "page")
        registry = ae.ProviderRegistry()
        adapter = FakeAdapter("local", "qwen", "unused")
        registry.register(adapter, owner_approved=True)
        executor = ae.Executor(record, registry, {"local": elig("local", "qwen")}, self.store)
        commit = "a" * 40
        executor.mark_verified(result.text, changed_paths=["daily-practice.html"], commit_sha=commit)
        self.assertEqual(record.state, ae.State.VERIFYING)
        self.assertEqual(record.commit_sha, commit)
        executor.mark_review_handoff()
        self.assertEqual(record.state, ae.State.REVIEWING)

    def test_invalid_state_transition_is_rejected(self):
        record = ae.TaskRecord(manifest())
        with self.assertRaises(ae.InvalidTransition):
            record.transition(ae.State.COMPLETE)


if __name__ == "__main__":
    unittest.main()
