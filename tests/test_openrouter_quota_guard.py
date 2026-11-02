import json
import tempfile
import unittest
from pathlib import Path
from scripts.openrouter_quota_guard import classify


class QuotaGuardTests(unittest.TestCase):
    def test_confirmed_daily_free_limit_is_blocked_with_run_evidence(self):
        status = {
            "task_id": "ADOS-POC-002",
            "state": "BLOCKED_QUOTA",
            "evidence": {
                "provider": "OpenRouter",
                "error_class": "free-models-per-day",
                "workflow_run_id": 41376777156,
            },
        }
        self.assertIn("BLOCKED_QUOTA", classify(status))
        self.assertIn("#41376777156", classify(status))

    def test_monetary_key_fields_never_unlock_request_quota(self):
        status = {
            "task_id": "ADOS-POC-002",
            "state": "UNVERIFIED",
            "key_metadata": {
                "usage_daily": 0,
                "limit_remaining": 0,
                "free_requests_remaining": 9999,
            },
        }
        result = classify(status)
        self.assertIn("BLOCKED_QUOTA", result)
        self.assertIn("monetary", result)

    def test_missing_or_malformed_checkpoint_fails_closed(self):
        self.assertIn("BLOCKED_QUOTA", classify(None))
        self.assertIn("BLOCKED_QUOTA", classify({"task_id": "OTHER"}))

    def test_unverified_evidence_does_not_authorize_a_model_call(self):
        status = {
            "task_id": "ADOS-POC-002",
            "state": "BLOCKED_QUOTA",
            "evidence": {
                "provider": "OpenRouter",
                "error_class": "unknown",
                "workflow_run_id": 41376777156,
            },
        }
        self.assertIn("lacks verifiable provider evidence", classify(status))


if __name__ == "__main__":
    unittest.main()
