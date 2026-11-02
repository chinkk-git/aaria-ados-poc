import sys
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from write_source_provenance import provenance


class SourceProvenanceTests(unittest.TestCase):
    def test_distinguishes_tested_pr_head_from_merge_sha(self):
        record = provenance(
            "37fd9444cc519788f92f665bb3913be0532a95fe",
            "37fd9444cc519788f92f665bb3913be0532a95fe",
            "f804ff5b0d528b817b102dc8c38c86d11ef02703",
            "41474479962",
        )
        self.assertEqual(record["source_commit_sha"], "37fd9444cc519788f92f665bb3913be0532a95fe")
        self.assertEqual(record["github_synthetic_merge_sha"], "f804ff5b0d528b817b102dc8c38c86d11ef02703")
        self.assertNotEqual(record["source_commit_sha"], record["github_synthetic_merge_sha"])

    def test_corrected_inference_record_preserves_head_merge_and_artifact_digests(self):
        record = json.loads((ROOT / "evidence/local-model-attempt-2-provenance-correction.json").read_text())
        raw = (ROOT / "evidence/local-model-attempt-2-raw-output.html").read_bytes()
        cleaned = (ROOT / "evidence/local-model-attempt-2-validated.html").read_bytes()
        self.assertEqual(record["workflow_run_head_sha"], "37fd9444cc519788f92f665bb3913be0532a95fe")
        self.assertEqual(record["checked_out_source_commit_sha"], "37fd9444cc519788f92f665bb3913be0532a95fe")
        self.assertEqual(record["github_synthetic_merge_sha"], "f804ff5b0d528b817b102dc8c38c86d11ef02703")
        self.assertEqual(record["preserved_original_claim_source_sha"], record["github_synthetic_merge_sha"])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), record["raw_generated_output_sha256"])
        self.assertEqual(hashlib.sha256(cleaned).hexdigest(), record["validated_unwrapped_output_sha256"])
        self.assertFalse(record["new_model_invocation"])

    def test_rejects_source_head_mismatch(self):
        with self.assertRaisesRegex(ValueError, "checked_out_sha_does_not_match_pr_head"):
            provenance("0" * 40, "1" * 40, "2" * 40, "123")


if __name__ == "__main__":
    unittest.main()
