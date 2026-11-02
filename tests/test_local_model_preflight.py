import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_model_preflight as preflight


class LocalModelMetadataTests(unittest.TestCase):
    def test_model_and_runtime_sources_are_immutable_and_checksum_pinned(self):
        self.assertEqual(len(preflight.MODEL_REVISION), 40)
        self.assertEqual(len(preflight.MODEL_SHA256), 64)
        self.assertEqual(len(preflight.LLAMA_CPP_COMMIT), 40)
        self.assertEqual(preflight.MODEL_LICENSE, "Apache-2.0")
        self.assertEqual(preflight.LLAMA_CPP_LICENSE, "MIT")
        self.assertEqual(preflight.MODEL_SIZE_BYTES, 491_400_064)

    def test_weight_verifier_rejects_wrong_size_without_network_or_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.gguf"
            path.write_bytes(b"fixture")
            with self.assertRaisesRegex(RuntimeError, "model_size_mismatch"):
                preflight.verify_model(path)

    def test_pinned_metadata_must_match_revision_license_and_filename(self):
        response = MagicMock()
        response.geturl.return_value = "https://huggingface.co/api/models/pinned"
        response.__enter__.return_value = response
        response.read.return_value = (
            '{"sha":"' + preflight.MODEL_REVISION + '",'
            '"cardData":{"license":"apache-2.0"},'
            '"siblings":[{"rfilename":"' + preflight.MODEL_FILENAME + '"}]}'
        ).encode()
        with patch.object(preflight.urllib.request, "urlopen", return_value=response):
            proof = preflight.verify_model_card_metadata()
        self.assertEqual(proof["model_revision"], preflight.MODEL_REVISION)
        self.assertEqual(proof["model_license"], "Apache-2.0")

    def test_license_tag_is_valid_pinned_metadata_when_card_field_is_absent(self):
        response = MagicMock()
        response.geturl.return_value = "https://huggingface.co/api/models/pinned"
        response.__enter__.return_value = response
        response.read.return_value = (
            '{"sha":"' + preflight.MODEL_REVISION + '",'
            '"tags":["text-generation","license:apache-2.0"],'
            '"siblings":[{"rfilename":"' + preflight.MODEL_FILENAME + '"}]}'
        ).encode()
        with patch.object(preflight.urllib.request, "urlopen", return_value=response):
            proof = preflight.verify_model_card_metadata()
        self.assertEqual(proof["model_license"], "Apache-2.0")

    def test_unverified_license_tag_still_blocks(self):
        response = MagicMock()
        response.geturl.return_value = "https://huggingface.co/api/models/pinned"
        response.__enter__.return_value = response
        response.read.return_value = (
            '{"sha":"' + preflight.MODEL_REVISION + '",'
            '"tags":["text-generation"],'
            '"siblings":[{"rfilename":"' + preflight.MODEL_FILENAME + '"}]}'
        ).encode()
        with patch.object(preflight.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "model_license_metadata_mismatch"):
                preflight.verify_model_card_metadata()


if __name__ == "__main__":
    unittest.main()
