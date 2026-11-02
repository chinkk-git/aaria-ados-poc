"""Single bounded, checksum-verified download of the pinned GGUF file."""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from local_model_preflight import (
    DOWNLOAD_ALLOWED_HOSTS,
    MODEL_SIZE_BYTES,
    model_url,
    verify_model,
)


def download(destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    started = time.monotonic()
    request = urllib.request.Request(
        model_url(), headers={"User-Agent": "ADOS-POC-local-model-fetch/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            host = urllib.parse.urlparse(response.geturl()).hostname or ""
            if host not in DOWNLOAD_ALLOWED_HOSTS:
                raise RuntimeError("model_redirect_host_not_allowlisted")
            announced = response.headers.get("Content-Length")
            if not announced or not announced.isdigit() or int(announced) != MODEL_SIZE_BYTES:
                raise RuntimeError("model_size_metadata_mismatch")
            written = 0
            with temporary.open("wb") as output:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    written += len(block)
                    if written > MODEL_SIZE_BYTES:
                        raise RuntimeError("model_download_over_size_limit")
                    output.write(block)
            if written != MODEL_SIZE_BYTES:
                raise RuntimeError("model_download_truncated")
        os.replace(temporary, destination)
        report = verify_model(destination)
        report["download_duration_seconds"] = round(time.monotonic() - started, 3)
        report["download_bytes"] = written
        return report
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__ == "__main__":
    try:
        evidence = download(Path(sys.argv[1]))
        report_path = Path(sys.argv[2])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8")
        print("LOCAL_MODEL_DOWNLOAD: PASS; size_bytes=" + str(evidence["download_bytes"])
              + "; sha256=" + str(evidence["model_sha256"])
              + "; duration_seconds=" + str(evidence["download_duration_seconds"]))
    except Exception as error:
        print("LOCAL_MODEL_DOWNLOAD: BLOCKED; code=" + str(error))
        raise SystemExit(1)
