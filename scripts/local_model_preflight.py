"""Verify model provenance, immutable source metadata, and runner capacity.

This script reads the pinned model card and sends an unauthenticated HEAD
request to Hugging Face. It does not download weights, call an inference API,
or read repository secrets.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

MODEL_REPO = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF"
MODEL_REVISION = "ebb2015119c907b064c512bf053e945850b5875f"
MODEL_FILENAME = "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
MODEL_SIZE_BYTES = 491_400_064
MODEL_SHA256 = "1d9614638d18024d0fbb36575a15f1302a3adf044df10345688ec4f6e1c4ff32"
MODEL_LICENSE = "Apache-2.0"
LLAMA_CPP_COMMIT = "3f8af15a76ac8675812e856da45454bafc90f12c"
LLAMA_CPP_LICENSE = "MIT"
MAX_PEAK_RSS_BYTES = 4 * 1024**3
MIN_FREE_DISK_BYTES = 6 * 1024**3
MIN_AVAILABLE_RAM_BYTES = 8 * 1024**3
MIN_TOTAL_RAM_BYTES = 12 * 1024**3
DOWNLOAD_ALLOWED_HOSTS = {
    # Current Hugging Face Hub storage and CDN endpoints documented at:
    # https://huggingface.co/docs/hub/models-downloading#downloading-behind-a-proxy-or-firewall
    "huggingface.co",
    "cas-bridge.xethub.hf.co",
    "cas-server.xethub.hf.co",
    "cas-server.xethub-eu.hf.co",
    "transfer.xethub.hf.co",
    "transfer.xethub-eu.hf.co",
    "us.aws.cdn.hf.co",
    "us.gcp.cdn.hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.hf.co",
    "cdn-lfs-eu-1.hf.co",
}


def meminfo_bytes(field: str) -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(field + ":"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("meminfo_field_missing")


def model_url() -> str:
    return (
        f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/"
        f"{MODEL_FILENAME}"
    )


def verify_model_card_metadata(timeout: float = 20.0) -> dict[str, object]:
    url = f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "ADOS-POC-local-model-preflight/1.0"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        final_host = urllib.parse.urlparse(response.geturl()).hostname or ""
        if final_host != "huggingface.co":
            raise RuntimeError("model_metadata_host_not_allowlisted")
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise RuntimeError("model_metadata_response_too_large")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("sha") != MODEL_REVISION:
        raise RuntimeError("model_revision_metadata_mismatch")
    card = data.get("cardData")
    license_tags = data.get("tags")
    tagged_license = isinstance(license_tags, list) and "license:apache-2.0" in license_tags
    card_license = isinstance(card, dict) and card.get("license") == "apache-2.0"
    if not (card_license or tagged_license):
        raise RuntimeError("model_license_metadata_mismatch")
    siblings = data.get("siblings")
    if not isinstance(siblings, list) or not any(
        isinstance(item, dict) and item.get("rfilename") == MODEL_FILENAME
        for item in siblings
    ):
        raise RuntimeError("model_file_missing_from_revision")
    return {
        "model_revision": data["sha"],
        "model_license": MODEL_LICENSE,
        "model_file_present": True,
        "metadata_host": final_host,
    }


def verify_runner_capacity() -> dict[str, int]:
    total = meminfo_bytes("MemTotal")
    available = meminfo_bytes("MemAvailable")
    disk = shutil.disk_usage(os.environ.get("RUNNER_TEMP", "/tmp")).free
    cpus = os.cpu_count() or 0
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError("runner_platform_not_supported")
    if cpus < 4 or total < MIN_TOTAL_RAM_BYTES or available < MIN_AVAILABLE_RAM_BYTES:
        raise RuntimeError("runner_memory_or_cpu_below_preflight_floor")
    if disk < MIN_FREE_DISK_BYTES:
        raise RuntimeError("runner_disk_below_preflight_floor")
    return {
        "cpu_count": cpus,
        "total_ram_bytes": total,
        "available_ram_bytes": available,
        "free_disk_bytes_before_download": disk,
    }


def verify_remote_metadata(timeout: float = 20.0) -> tuple[int, str]:
    request = urllib.request.Request(
        model_url(),
        method="HEAD",
        headers={"User-Agent": "ADOS-POC-local-model-preflight/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        final_host = urllib.parse.urlparse(response.geturl()).hostname or ""
        if final_host not in DOWNLOAD_ALLOWED_HOSTS:
            raise RuntimeError("model_redirect_host_not_allowlisted")
        size = response.headers.get("Content-Length")
        if not size or not size.isdigit():
            raise RuntimeError("model_size_metadata_unavailable")
        if int(size) != MODEL_SIZE_BYTES:
            raise RuntimeError("model_size_metadata_mismatch")
        return int(size), final_host


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model(path: Path) -> dict[str, object]:
    actual_size = path.stat().st_size
    if actual_size != MODEL_SIZE_BYTES:
        raise RuntimeError("model_size_mismatch")
    digest = sha256_file(path)
    if digest != MODEL_SHA256:
        raise RuntimeError("model_sha256_mismatch")
    return {
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "model_filename": MODEL_FILENAME,
        "model_size_bytes": actual_size,
        "model_sha256": digest,
        "model_license": MODEL_LICENSE,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "llama_cpp_license": LLAMA_CPP_LICENSE,
        "expected_peak_rss_limit_bytes": MAX_PEAK_RSS_BYTES,
    }


def preflight() -> dict[str, object]:
    capacity = verify_runner_capacity()
    model_metadata = verify_model_card_metadata()
    size, host = verify_remote_metadata()
    return {
        "preflight": "PASS",
        "weights_downloaded": False,
        "model_source_url": model_url(),
        "model_expected_size_bytes": size,
        "model_sha256": MODEL_SHA256,
        "model_license": MODEL_LICENSE,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "llama_cpp_license": LLAMA_CPP_LICENSE,
        "remote_metadata_host": host,
        **model_metadata,
        "estimated_peak_ram_limit_bytes": MAX_PEAK_RSS_BYTES,
        **capacity,
        "checked_unix_time": int(time.time()),
    }


if __name__ == "__main__":
    try:
        report = preflight()
        destination = Path(sys.argv[1]) if len(sys.argv) > 1 else None
        serialized = json.dumps(report, sort_keys=True)
        if destination:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
    except Exception as error:
        print("LOCAL_MODEL_PREFLIGHT: BLOCKED; code=" + str(error))
        raise SystemExit(1)
