"""Start a pinned local llama.cpp server and issue exactly one synthetic request."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from local_model_preflight import MODEL_REPO, MODEL_REVISION, verify_model


PROMPT = (
    "Return one complete standalone HTML document and nothing else. It must have "
    "the heading Daily Practice, a native accessible Mark Complete button, a "
    "role=status aria-live=polite indicator initially saying Not completed, and "
    "an inline click handler that changes the indicator to Completed. Reloading "
    "must return to Not completed. Use no external resources, libraries, network "
    "calls, or persistent storage. Include a doctype, lang, title, and main."
)
BASE_URL = "http://127.0.0.1:8090"


def rss_kib(pid: int) -> int:
    status_file = Path(f"/proc/{pid}/status")
    if not status_file.exists():
        return 0
    for line in status_file.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return 0


def request_json(url: str, data: bytes | None = None, timeout: float = 3.0) -> dict[str, object]:
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read(2_000_001))
    if not isinstance(payload, dict):
        raise RuntimeError("local_server_invalid_response")
    return payload


def run(model_path: Path, output_path: Path, evidence_path: Path) -> dict[str, object]:
    verified = verify_model(model_path)
    binary = os.environ.get("LLAMA_SERVER_BIN", "")
    expected_commit = os.environ.get("LLAMA_CPP_EXPECTED_COMMIT", "")
    if not binary or not Path(binary).is_file() or len(expected_commit) != 40:
        raise RuntimeError("pinned_local_runtime_missing")
    command = [
        binary, "-m", str(model_path), "--host", "127.0.0.1", "--port", "8090",
        "--ctx-size", "4096", "--threads", "4", "--n-gpu-layers", "0",
    ]
    log_path = evidence_path.parent / "llama-server.log"
    started = time.monotonic()
    process = subprocess.Popen(
        command, stdout=log_path.open("wb"), stderr=subprocess.STDOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        start_new_session=True,
    )
    peak_rss = 0
    stop_monitor = threading.Event()

    def monitor() -> None:
        nonlocal peak_rss
        while not stop_monitor.wait(0.1):
            peak_rss = max(peak_rss, rss_kib(process.pid))

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        ready_at = None
        readiness_deadline = time.monotonic() + 120
        while time.monotonic() < readiness_deadline:
            if process.poll() is not None:
                raise RuntimeError("local_server_exited_before_ready")
            try:
                request_json(BASE_URL + "/health", timeout=2.0)
                ready_at = time.monotonic()
                break
            except (urllib.error.URLError, TimeoutError, ValueError):
                time.sleep(0.25)
        if ready_at is None:
            raise RuntimeError("local_server_startup_timeout")
        startup_seconds = ready_at - started

        body = json.dumps({
            "model": MODEL_REPO,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "max_tokens": 1200,
            "stream": False,
        }).encode("utf-8")
        inference_started = time.monotonic()
        payload = request_json(BASE_URL + "/v1/chat/completions", body, timeout=900)
        inference_seconds = time.monotonic() - inference_started
        try:
            html = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("local_generation_response_shape") from exc
        if not isinstance(html, str) or not html.strip() or len(html.encode("utf-8")) > 100_000:
            raise RuntimeError("local_generation_output_invalid")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        report = {
            "status": "GENERATED_PENDING_FUNCTIONAL_VALIDATION",
            "model": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_file_sha256": verified["model_sha256"],
            "model_file_bytes": verified["model_size_bytes"],
            "llama_cpp_commit": expected_commit,
            "llama_cpp_source_integrity": "exact_git_commit_pin",
            "requests": 1,
            "startup_seconds": round(startup_seconds, 3),
            "inference_seconds": round(inference_seconds, 3),
            "peak_process_rss_kib": peak_rss,
            "prompt_tokens": payload.get("usage", {}).get("prompt_tokens")
            if isinstance(payload.get("usage"), dict) else None,
            "output_tokens": payload.get("usage", {}).get("completion_tokens")
            if isinstance(payload.get("usage"), dict) else None,
            "output_bytes": len(html.encode("utf-8")),
            "output_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
            "external_api_requests": 0,
            "secret_access": False,
        }
        evidence_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
        return report
    finally:
        stop_monitor.set()
        watcher.join(timeout=1)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)


if __name__ == "__main__":
    try:
        report = run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
        print(json.dumps(report, sort_keys=True))
    except Exception as error:
        print("LOCAL_MODEL_RUN: BLOCKED; code=" + str(error))
        raise SystemExit(1)
