import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import download_local_model as download


class LocalModelDownloadPolicyTests(unittest.TestCase):
    def test_untrusted_redirect_is_rejected_before_any_model_bytes_are_read(self):
        response = MagicMock()
        response.geturl.return_value = "https://untrusted.example/model"
        response.__enter__.return_value = response
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "model.gguf"
            with patch.object(download.urllib.request, "urlopen", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "model_redirect_host_not_allowlisted"):
                    download.download(destination)
            response.read.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".gguf.part").exists())


if __name__ == "__main__":
    unittest.main()
