import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ados_executor as ae
import provider_catalog as pc


class ProviderCatalogTests(unittest.TestCase):
    def test_status_records_fail_closed_and_never_claim_unknown_quota(self):
        records = pc.default_status_records()
        self.assertEqual(records["openrouter"].eligibility, pc.EligibilityState.QUOTA_EXHAUSTED)
        self.assertEqual(records["openrouter"].quota_status, "EXHAUSTED")
        self.assertEqual(records["groq"].eligibility, pc.EligibilityState.BILLING_UNVERIFIED)
        self.assertEqual(records["nvidia_nim"].eligibility, pc.EligibilityState.AUTH_UNAVAILABLE)
        self.assertEqual(records["github_models"].eligibility, pc.EligibilityState.DISABLED)
        serial = json.dumps({k: v.as_json() for k, v in records.items()})
        self.assertNotIn("API_KEY_VALUE", serial)

    def test_verified_free_status_requires_evidence_and_available_quota(self):
        with self.assertRaisesRegex(ValueError, "verified_provider_requires_timestamp"):
            pc.ProviderStatusRecord("x", "m", pc.EligibilityState.VERIFIED_FREE, None,
                "AVAILABLE", "proof", "source", None, True,
                pc.ProviderCapabilities(True, False), 0.0, "https://x", "Bearer", "X_KEY", "ok")

    def test_candidate_registry_map_includes_only_explicit_zero_cost_verified_models(self):
        self.assertEqual(pc.verified_free_model_map(pc.default_status_records()), {})
        verified = pc.ProviderStatusRecord(
            "lab", "lab/coder-v1", pc.EligibilityState.VERIFIED_FREE,
            "2026-11-02T00:00:00Z", "AVAILABLE", "account evidence", "owner-approved source",
            None, True, pc.ProviderCapabilities(True, True, 32768, 4096, "verified fixture"),
            0.0, "https://provider.example/v1", "Bearer", "LAB_API_KEY", "approved")
        self.assertEqual(pc.verified_free_model_map({"lab": verified}), {"lab": "lab/coder-v1"})

    def test_candidate_adapters_use_one_standard_single_call_response(self):
        payload = {"choices": [{"message": {"content": "synthetic result"}}],
                   "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
        for adapter in (pc.GroqCloudAdapter("openai/gpt-oss-20b", lambda: "mock-secret"),
                        pc.NvidiaNimAdapter("openai/gpt-oss-20b", lambda: "mock-secret")):
            with self.subTest(provider=adapter.name), patch.object(ae, "_http_json", return_value=(200, payload)) as call:
                request = pc.ProviderRequest("ADOS-POC-999", adapter.model, "synthetic")
                result = adapter.execute_request(request)
                self.assertEqual(result.provider, adapter.name)
                self.assertEqual(result.request_count, 1)
                self.assertEqual((result.prompt_tokens, result.output_tokens), (3, 4))
                self.assertEqual(call.call_count, 1)
                self.assertEqual(call.call_args.args[1]["Authorization"], "Bearer mock-secret")
                self.assertNotIn("mock-secret", repr(result))

    def test_github_models_adapter_is_permanently_non_invocable(self):
        adapter = pc.GitHubModelsAdapter()
        with self.assertRaises(ae.ProviderFailure) as failure:
            adapter.execute("synthetic")
        self.assertEqual(failure.exception.code, "service_retired")

    def test_provider_rate_limit_is_not_misclassified_as_quota_exhaustion(self):
        adapter = pc.GroqCloudAdapter("openai/gpt-oss-20b", lambda: "mock-secret")
        with patch.object(ae, "_http_json", return_value=(429, {"error": {"message": "limit"}})):
            with self.assertRaises(ae.ProviderFailure) as failure:
                adapter.execute("synthetic")
        self.assertEqual(failure.exception.kind, ae.FailureKind.RATE_LIMIT_UNKNOWN)

    def test_openrouter_quota_switch_to_explicitly_approved_future_free_adapter(self):
        class Fake:
            def __init__(self, name, model, outcome):
                self.name, self.model, self.outcome, self.calls = name, model, outcome, 0
            def execute(self, prompt):
                self.calls += 1
                if isinstance(self.outcome, Exception):
                    raise self.outcome
                return ae.ExecutionResult(self.name, self.model, self.outcome, 1, 0.01)

        task = ae.TaskManifest("ADOS-POC-999", "experiment/python-multiprovider-executor",
                               ("daily-practice.html",), ("openrouter", "approved_lab"),
                               max_provider_attempts=2)
        record, store = ae.TaskRecord(task), ae.CheckpointStore(Path(tempfile.mkdtemp()))
        first = Fake("openrouter", "openrouter/free", ae.ProviderFailure(
            "openrouter", ae.FailureKind.QUOTA_EXHAUSTED, outcome_known=True,
            code="free-models-per-day"))
        second = Fake("approved_lab", "free-model-x", "resumed output")
        registry = ae.ProviderRegistry(approved_extensions=("approved_lab",),
                                       verified_free_models={"approved_lab": "free-model-x"})
        registry.register(first, owner_approved=True)
        registry.register(second, owner_approved=True)
        proofs = {"openrouter": ae.Eligibility("openrouter", "openrouter/free", ae.BillingState.VERIFIED_ZERO_COST,
                  "fixture", True, 1, True),
                  "approved_lab": ae.Eligibility("approved_lab", "free-model-x", ae.BillingState.VERIFIED_ZERO_COST,
                  "fixture", True, 1, True)}
        result = ae.Executor(record, registry, proofs, store).execute("same task context")
        saved = store.read(task.task_id)
        self.assertEqual(result.provider, "approved_lab")
        self.assertEqual((first.calls, second.calls), (1, 1))
        self.assertEqual(saved["task"]["task_id"], task.task_id)
        self.assertEqual(saved["last_successful_step"], "provider_output_received")
        self.assertEqual(saved["attempts"][0]["error_kind"], "QUOTA_EXHAUSTED")

    def test_gemini_quota_exhaustion_resumes_with_openrouter(self):
        class Fake:
            def __init__(self, name, model, outcome):
                self.name, self.model, self.outcome, self.calls = name, model, outcome, 0
            def execute(self, prompt):
                self.calls += 1
                if isinstance(self.outcome, Exception):
                    raise self.outcome
                return ae.ExecutionResult(self.name, self.model, self.outcome, 1, 0.01)

        task = ae.TaskManifest("ADOS-POC-999", "experiment/python-multiprovider-executor",
                               ("daily-practice.html",), ("gemini", "openrouter"),
                               max_provider_attempts=2)
        record, store = ae.TaskRecord(task), ae.CheckpointStore(Path(tempfile.mkdtemp()))
        gemini = Fake("gemini", "gemini-3.5-flash-lite", ae.ProviderFailure(
            "gemini", ae.FailureKind.QUOTA_EXHAUSTED, outcome_known=True,
            code="google_rpc_quota_failure"))
        openrouter = Fake("openrouter", "openrouter/free", "resumed page")
        registry = ae.ProviderRegistry()
        registry.register(gemini, owner_approved=True)
        registry.register(openrouter, owner_approved=True)
        proofs = {"gemini": ae.Eligibility("gemini", gemini.model, ae.BillingState.VERIFIED_ZERO_COST,
                  "fixture", True, 1, True),
                  "openrouter": ae.Eligibility("openrouter", openrouter.model, ae.BillingState.VERIFIED_ZERO_COST,
                  "fixture", True, 1, True)}
        result = ae.Executor(record, registry, proofs, store).execute("same task context")
        self.assertEqual(result.provider, "openrouter")
        self.assertEqual((gemini.calls, openrouter.calls), (1, 1))
        self.assertEqual(record.manifest.task_id, store.read(task.task_id)["task"]["task_id"])

    def test_candidate_cloud_adapters_cannot_register_without_verified_free_model(self):
        adapter = pc.GroqCloudAdapter("openai/gpt-oss-20b", lambda: "never-used")
        with self.assertRaisesRegex(ae.ExecutorError, "provider_requires_exact_verified_free_model"):
            ae.ProviderRegistry().register(adapter, owner_approved=True)
        registry = ae.ProviderRegistry(verified_free_models={"groq": "different-model"})
        with self.assertRaisesRegex(ae.ExecutorError, "provider_requires_exact_verified_free_model"):
            registry.register(adapter, owner_approved=True)


if __name__ == "__main__":
    unittest.main()
