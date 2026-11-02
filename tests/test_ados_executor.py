import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import ados_executor as ae


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
            "max_attempts": 1,
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
                             ("max_attempts", 4)):
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
