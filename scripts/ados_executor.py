"""Minimal deterministic, zero-cost ADOS task executor and provider adapters.

Adapters return data only. They do not own task state, leases, file writes,
commits, reviews, or merge permissions.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, Protocol, Sequence


class State(str, Enum):
    READY = "READY"
    CLAIMED = "CLAIMED"
    BUILDING = "BUILDING"
    CHECKPOINTED = "CHECKPOINTED"
    VERIFYING = "VERIFYING"
    REVIEWING = "REVIEWING"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


ALLOWED_TRANSITIONS: dict[State, frozenset[State]] = {
    State.READY: frozenset({State.CLAIMED, State.BLOCKED, State.FAILED}),
    State.CLAIMED: frozenset({State.BUILDING, State.BLOCKED, State.FAILED}),
    State.BUILDING: frozenset({State.CHECKPOINTED, State.VERIFYING, State.BLOCKED, State.FAILED}),
    State.CHECKPOINTED: frozenset({State.BUILDING, State.VERIFYING, State.BLOCKED, State.FAILED}),
    State.VERIFYING: frozenset({State.REVIEWING, State.BLOCKED, State.FAILED}),
    State.REVIEWING: frozenset({State.COMPLETE, State.BUILDING, State.BLOCKED, State.FAILED}),
    State.COMPLETE: frozenset(),
    State.BLOCKED: frozenset({State.CLAIMED}),
    State.FAILED: frozenset(),
}


class FailureKind(str, Enum):
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    RATE_LIMIT_UNKNOWN = "RATE_LIMIT_UNKNOWN"
    AUTHENTICATION = "AUTHENTICATION"
    BILLING_UNKNOWN = "BILLING_UNKNOWN"
    TIMEOUT_UNKNOWN = "TIMEOUT_UNKNOWN"
    TRANSPORT_UNKNOWN = "TRANSPORT_UNKNOWN"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    POLICY = "POLICY"
    UNAVAILABLE = "UNAVAILABLE"


class BillingState(str, Enum):
    VERIFIED_ZERO_COST = "VERIFIED_ZERO_COST"
    UNKNOWN = "UNKNOWN"
    PAID = "PAID"


class ExecutorError(RuntimeError):
    pass


class LeaseHeld(ExecutorError):
    pass


class InvalidTransition(ExecutorError):
    pass


class DuplicateExecution(ExecutorError):
    pass


class UnauthorizedPath(ExecutorError):
    pass


class ProviderFailure(ExecutorError):
    def __init__(
        self,
        provider: str,
        kind: FailureKind,
        *,
        outcome_known: bool,
        output_committed: bool = False,
        code: str = "",
    ) -> None:
        super().__init__(f"{provider}:{kind.value}:{code}")
        self.provider = provider
        self.kind = kind
        self.outcome_known = outcome_known
        self.output_committed = output_committed
        self.code = code


def validate_task_id(task_id: str) -> None:
    if not re.fullmatch(r"ADOS-POC-[0-9]{3}", task_id):
        raise ValueError("invalid_task_id")


def validate_branch(branch: str) -> None:
    if not re.fullmatch(r"(?:feature|experiment)/[a-z0-9][a-z0-9._/-]*", branch):
        raise ValueError("invalid_branch")
    if branch == "main" or ".." in branch.split("/"):
        raise ValueError("protected_or_invalid_branch")


def normalize_scope_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or not path or "\\" in path or any(p in {"", ".", ".."} for p in path.split("/")):
        raise UnauthorizedPath("unsafe_path")
    normalized = str(candidate)
    if normalized != path:
        raise UnauthorizedPath("noncanonical_path")
    return normalized


@dataclass(frozen=True)
class TaskManifest:
    task_id: str
    branch: str
    scope_paths: tuple[str, ...]
    providers: tuple[str, ...] = ("local", "gemini", "openrouter")
    cost_ceiling_usd: float = 0.0
    max_provider_attempts: int = 3

    def __post_init__(self) -> None:
        validate_task_id(self.task_id)
        validate_branch(self.branch)
        if not self.scope_paths or len(set(self.scope_paths)) != len(self.scope_paths):
            raise ValueError("invalid_scope")
        for path in self.scope_paths:
            normalize_scope_path(path)
        if self.cost_ceiling_usd != 0.0:
            raise ValueError("cost_ceiling_must_be_zero")
        if not 1 <= self.max_provider_attempts <= 3:
            raise ValueError("attempt_limit_out_of_range")
        if not self.providers or len(set(self.providers)) != len(self.providers):
            raise ValueError("invalid_provider_order")
        if any(not re.fullmatch(r"[a-z][a-z0-9_-]{1,31}", p) for p in self.providers):
            raise ValueError("invalid_provider_name")

    def as_json(self) -> dict[str, object]:
        result = asdict(self)
        result["scope_paths"] = list(self.scope_paths)
        result["providers"] = list(self.providers)
        return result


def load_manifest(path: Path) -> TaskManifest:
    """Load the versioned JSON task shape using only the standard library."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "task_id", "branch", "scope_paths", "providers",
        "cost_ceiling_usd", "max_provider_attempts",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("manifest_fields_mismatch")
    if not isinstance(payload["task_id"], str) or not isinstance(payload["branch"], str):
        raise ValueError("manifest_identity_type_invalid")
    if not isinstance(payload["scope_paths"], list) or not all(
        isinstance(value, str) for value in payload["scope_paths"]
    ):
        raise ValueError("manifest_scope_type_invalid")
    if not isinstance(payload["providers"], list) or not all(
        isinstance(value, str) for value in payload["providers"]
    ):
        raise ValueError("manifest_provider_type_invalid")
    if isinstance(payload["cost_ceiling_usd"], bool) or not isinstance(
        payload["cost_ceiling_usd"], (int, float)
    ):
        raise ValueError("manifest_cost_type_invalid")
    attempts = payload["max_provider_attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int):
        raise ValueError("manifest_attempt_type_invalid")
    return TaskManifest(
        task_id=payload["task_id"],
        branch=payload["branch"],
        scope_paths=tuple(payload["scope_paths"]),
        providers=tuple(payload["providers"]),
        cost_ceiling_usd=float(payload["cost_ceiling_usd"]),
        max_provider_attempts=attempts,
    )


@dataclass(frozen=True)
class Eligibility:
    provider: str
    model: str
    billing: BillingState
    evidence_id: str
    approved: bool
    quota_remaining: int | None = None
    model_free_tier: bool = False
    license_verified: bool = False
    runner_capacity_verified: bool = False

    def eligible(self) -> bool:
        if not self.approved or self.billing is not BillingState.VERIFIED_ZERO_COST:
            return False
        if not self.evidence_id.strip():
            return False
        if self.provider == "local":
            return self.license_verified and self.runner_capacity_verified
        if not self.model_free_tier or self.quota_remaining is None:
            return False
        return self.quota_remaining > 0


@dataclass(frozen=True)
class ExecutionResult:
    provider: str
    model: str
    text: str
    request_count: int
    duration_seconds: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.request_count != 1:
            raise ValueError("provider_execution_must_be_single_attempt")
        if not self.text.strip():
            raise ValueError("empty_provider_output")


class ProviderAdapter(Protocol):
    name: str
    model: str

    def execute(self, prompt: str) -> ExecutionResult: ...


@dataclass
class AttemptEvidence:
    provider: str
    model: str
    outcome: str
    error_kind: str | None = None
    error_code: str | None = None
    duration_seconds: float | None = None


@dataclass
class TaskRecord:
    manifest: TaskManifest
    state: State = State.READY
    attempts: list[AttemptEvidence] = field(default_factory=list)
    output_sha256: str | None = None
    commit_sha: str | None = None
    action_keys: set[str] = field(default_factory=set)
    last_successful_step: str = "task_validated"

    def transition(self, new_state: State, *, checkpoint_step: str | None = None) -> None:
        if new_state not in ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.state.value}->{new_state.value}")
        self.state = new_state
        if checkpoint_step:
            self.last_successful_step = checkpoint_step

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "task": self.manifest.as_json(),
            "state": self.state.value,
            "attempts": [asdict(a) for a in self.attempts],
            "output_sha256": self.output_sha256,
            "commit_sha": self.commit_sha,
            "action_keys": sorted(self.action_keys),
            "last_successful_step": self.last_successful_step,
            "cost_ceiling_usd": 0.0,
        }


def exact_sha(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{40}", value))


def validate_changed_paths(changed_paths: Sequence[str], allowed_paths: Sequence[str]) -> None:
    allowed = {normalize_scope_path(p) for p in allowed_paths}
    normalized = {normalize_scope_path(p) for p in changed_paths}
    if not normalized or not normalized <= allowed:
        raise UnauthorizedPath("out_of_scope_change")


class CheckpointStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, record: TaskRecord) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{record.manifest.task_id}.json"
        fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record.checkpoint_payload(), stream, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            dir_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return target

    def read(self, task_id: str) -> dict[str, object] | None:
        validate_task_id(task_id)
        path = self.root / f"{task_id}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task", {}).get("task_id") != task_id:
            raise ExecutorError("checkpoint_identity_mismatch")
        if payload.get("cost_ceiling_usd") != 0.0:
            raise ExecutorError("checkpoint_cost_policy_mismatch")
        return payload


class TaskLease:
    """Exclusive durable lease. Recovery preserves the retired lease record."""

    def __init__(self, root: Path, manifest: TaskManifest, worker: str) -> None:
        self.root = root
        self.manifest = manifest
        self.worker = worker
        self.token = uuid.uuid4().hex
        self.path = root / f"{manifest.task_id}.lease"
        self.operation_path = root / f"{manifest.task_id}.lease-operation"
        self.acquired = False

    @contextmanager
    def _operation_lock(self):
        try:
            fd = os.open(self.operation_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise LeaseHeld("lease_operation_in_progress_or_interrupted") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"pid": os.getpid(), "worker": self.worker}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            yield
        finally:
            try:
                self.operation_path.unlink()
            except FileNotFoundError:
                pass

    def _create_lease(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": self.manifest.task_id,
            "branch": self.manifest.branch,
            "scope_paths": list(self.manifest.scope_paths),
            "worker": self.worker,
            "token": self.token,
            "pid": os.getpid(),
        }
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise LeaseHeld("exclusive_task_lease_held") from exc
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        self.acquired = True

    def acquire(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self._operation_lock():
            self._create_lease()

    def recover(self, *, previous_worker_inactive: bool) -> None:
        if not previous_worker_inactive:
            raise LeaseHeld("recovery_requires_inactive_worker_evidence")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._operation_lock():
            try:
                old = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise LeaseHeld("lease_missing_during_recovery") from exc
            expected = {
                "task_id": self.manifest.task_id,
                "branch": self.manifest.branch,
                "scope_paths": list(self.manifest.scope_paths),
            }
            if any(old.get(k) != v for k, v in expected.items()):
                raise LeaseHeld("existing_lease_scope_mismatch")
            retired = self.root / f"{self.manifest.task_id}.lease.retired-{old.get('token', 'unknown')}"
            try:
                os.rename(self.path, retired)
            except FileNotFoundError as exc:
                raise LeaseHeld("lease_changed_during_recovery") from exc
            self._create_lease()

    def release(self) -> None:
        if not self.acquired:
            return
        with self._operation_lock():
            try:
                current = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                self.acquired = False
                raise LeaseHeld("lease_missing_on_release") from exc
            if current.get("token") != self.token:
                raise LeaseHeld("lease_owner_mismatch")
            self.path.unlink()
            self.acquired = False

    def __enter__(self) -> "TaskLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http_json(url: str, headers: Mapping[str, str], body: dict[str, object], timeout: float) -> tuple[int, dict[str, object]]:
    encoded = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=encoded, headers={**headers, "Content-Type": "application/json"}, method="POST"
    )
    opener = urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ProviderFailure("cloud", FailureKind.INVALID_RESPONSE, outcome_known=True, code="response_too_large")
            try:
                payload = json.loads(raw)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ProviderFailure("cloud", FailureKind.INVALID_RESPONSE, outcome_known=True, code="invalid_json") from exc
            if not isinstance(payload, dict):
                raise ProviderFailure("cloud", FailureKind.INVALID_RESPONSE, outcome_known=True, code="invalid_json_shape")
            return response.status, payload
    except urllib.error.HTTPError as exc:
        raw = exc.read(100_001)
        try:
            payload = json.loads(raw[:100_000]) if raw else {}
        except (ValueError, UnicodeDecodeError):
            payload = {}
        return exc.code, payload if isinstance(payload, dict) else {}
    except TimeoutError as exc:
        raise ProviderFailure("cloud", FailureKind.TIMEOUT_UNKNOWN, outcome_known=False, code="timeout") from exc
    except urllib.error.URLError as exc:
        raise ProviderFailure("cloud", FailureKind.TRANSPORT_UNKNOWN, outcome_known=False, code="transport") from exc


def classify_google_error(status_code: int, payload: Mapping[str, object]) -> tuple[FailureKind, str]:
    error = payload.get("error")
    if not isinstance(error, dict):
        return (FailureKind.INVALID_RESPONSE, "malformed_google_error")
    code = str(error.get("status", ""))
    if status_code in (401, 403) or code in {"UNAUTHENTICATED", "PERMISSION_DENIED"}:
        return (FailureKind.AUTHENTICATION, code or str(status_code))
    details = error.get("details")
    confirmed_quota = isinstance(details, list) and any(
        isinstance(item, dict)
        and str(item.get("@type", "")).endswith("google.rpc.QuotaFailure")
        for item in details
    )
    if status_code == 429 and confirmed_quota:
        return (FailureKind.QUOTA_EXHAUSTED, "google_rpc_quota_failure")
    if status_code == 429:
        return (FailureKind.RATE_LIMIT_UNKNOWN, code or "http_429")
    if status_code >= 500:
        return (FailureKind.TRANSPORT_UNKNOWN, code or f"http_{status_code}")
    return (FailureKind.POLICY, code or f"http_{status_code}")


def classify_openrouter_error(status_code: int, payload: Mapping[str, object]) -> tuple[FailureKind, str]:
    error = payload.get("error")
    if not isinstance(error, dict):
        return (FailureKind.INVALID_RESPONSE, "malformed_openrouter_error")
    code = str(error.get("code", ""))
    if status_code in (401, 403) or code in {"401", "403", "unauthorized", "invalid_api_key"}:
        return (FailureKind.AUTHENTICATION, code or str(status_code))
    if status_code == 429 and code == "free-models-per-day":
        return (FailureKind.QUOTA_EXHAUSTED, code)
    if status_code == 429:
        return (FailureKind.RATE_LIMIT_UNKNOWN, code or "http_429")
    if status_code >= 500:
        return (FailureKind.TRANSPORT_UNKNOWN, code or f"http_{status_code}")
    return (FailureKind.POLICY, code or f"http_{status_code}")


class GeminiAdapter:
    name = "gemini"
    model = "gemini-3.5-flash-lite"

    def __init__(self, key_supplier: Callable[[], str], *, timeout: float = 90.0) -> None:
        self._key_supplier = key_supplier
        self.timeout = timeout

    def execute(self, prompt: str) -> ExecutionResult:
        key = self._key_supplier()
        if not key:
            raise ProviderFailure(self.name, FailureKind.AUTHENTICATION, outcome_known=True, code="missing_secret")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        started = time.monotonic()
        status, payload = _http_json(
            url,
            {"x-goog-api-key": key},
            {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"candidateCount": 1, "maxOutputTokens": 2048}},
            self.timeout,
        )
        if status != 200:
            kind, code = classify_google_error(status, payload)
            raise ProviderFailure(self.name, kind, outcome_known=kind not in {FailureKind.TIMEOUT_UNKNOWN, FailureKind.TRANSPORT_UNKNOWN}, code=code)
        try:
            candidates = payload["candidates"]
            text = "".join(p["text"] for p in candidates[0]["content"]["parts"] if isinstance(p, dict) and isinstance(p.get("text"), str))
            usage = payload.get("usageMetadata", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderFailure(self.name, FailureKind.INVALID_RESPONSE, outcome_known=True, code="response_shape") from exc
        if not text.strip():
            raise ProviderFailure(self.name, FailureKind.INVALID_RESPONSE, outcome_known=True, code="empty_output")
        usage = usage if isinstance(usage, dict) else {}
        return ExecutionResult(self.name, self.model, text, 1, time.monotonic() - started,
            usage.get("promptTokenCount") if isinstance(usage.get("promptTokenCount"), int) else None,
            usage.get("candidatesTokenCount") if isinstance(usage.get("candidatesTokenCount"), int) else None)


class OpenRouterAdapter:
    name = "openrouter"
    model = "openrouter/free"

    def __init__(self, key_supplier: Callable[[], str], *, timeout: float = 90.0) -> None:
        self._key_supplier = key_supplier
        self.timeout = timeout

    def execute(self, prompt: str) -> ExecutionResult:
        key = self._key_supplier()
        if not key:
            raise ProviderFailure(self.name, FailureKind.AUTHENTICATION, outcome_known=True, code="missing_secret")
        started = time.monotonic()
        status, payload = _http_json(
            "https://openrouter.ai/api/v1/chat/completions",
            {"Authorization": f"Bearer {key}"},
            {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048},
            self.timeout,
        )
        if status != 200:
            kind, code = classify_openrouter_error(status, payload)
            raise ProviderFailure(self.name, kind, outcome_known=kind not in {FailureKind.TIMEOUT_UNKNOWN, FailureKind.TRANSPORT_UNKNOWN}, code=code)
        try:
            text = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderFailure(self.name, FailureKind.INVALID_RESPONSE, outcome_known=True, code="response_shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise ProviderFailure(self.name, FailureKind.INVALID_RESPONSE, outcome_known=True, code="empty_output")
        usage = usage if isinstance(usage, dict) else {}
        return ExecutionResult(self.name, self.model, text, 1, time.monotonic() - started,
            usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
            usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None)


class LocalHTTPAdapter:
    name = "local"

    def __init__(self, base_url: str, model: str, *, timeout: float = 600.0) -> None:
        if not base_url.startswith("http://127.0.0.1:") and not base_url.startswith("http://localhost:"):
            raise ValueError("local_model_must_bind_loopback")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def execute(self, prompt: str) -> ExecutionResult:
        started = time.monotonic()
        status, payload = _http_json(
            self.base_url + "/v1/chat/completions",
            {},
            {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048, "temperature": 0},
            self.timeout,
        )
        if status != 200:
            raise ProviderFailure(self.name, FailureKind.UNAVAILABLE, outcome_known=True, code=f"http_{status}")
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderFailure(self.name, FailureKind.INVALID_RESPONSE, outcome_known=True, code="response_shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise ProviderFailure(self.name, FailureKind.INVALID_RESPONSE, outcome_known=True, code="empty_output")
        return ExecutionResult(self.name, self.model, text, 1, time.monotonic() - started)


class ProviderRegistry:
    def __init__(self, *, approved_extensions: Sequence[str] = ()) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}
        self._approved = {"local", "gemini", "openrouter", *approved_extensions}

    def register(self, adapter: ProviderAdapter, *, owner_approved: bool) -> None:
        if not owner_approved:
            raise ExecutorError("provider_not_owner_approved")
        if adapter.name in self._adapters:
            raise ExecutorError("duplicate_provider_adapter")
        if adapter.name not in self._approved:
            raise ExecutorError("provider_not_allowlisted")
        if adapter.name == "openrouter" and adapter.model != "openrouter/free":
            raise ExecutorError("paid_or_unapproved_openrouter_route")
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> ProviderAdapter | None:
        return self._adapters.get(name)


class Executor:
    def __init__(
        self,
        record: TaskRecord,
        registry: ProviderRegistry,
        eligibility: Mapping[str, Eligibility],
        store: CheckpointStore,
    ) -> None:
        self.record = record
        self.registry = registry
        self.eligibility = eligibility
        self.store = store

    def execute(self, prompt: str, *, phase: str = "generate") -> ExecutionResult:
        if self.record.output_sha256 is not None:
            raise DuplicateExecution("task_output_already_exists")
        if self.record.state is State.BLOCKED:
            if self.record.last_successful_step != "no_eligible_provider":
                raise InvalidTransition("blocked_state_requires_manual_reconciliation")
            self.record.transition(State.CLAIMED, checkpoint_step="recovery_reconciled")
        if self.record.state is State.READY:
            self.record.transition(State.CLAIMED, checkpoint_step="lease_acquired")
        if self.record.state is State.CLAIMED:
            self.record.transition(State.BUILDING, checkpoint_step="executor_started")
        if self.record.state not in {State.BUILDING, State.CHECKPOINTED}:
            raise InvalidTransition(f"cannot_execute_from_{self.record.state.value}")
        attempts_used = len(self.record.attempts)
        for provider_name in self.record.manifest.providers:
            if attempts_used >= self.record.manifest.max_provider_attempts:
                break
            adapter = self.registry.get(provider_name)
            proof = self.eligibility.get(provider_name)
            if adapter is None or proof is None or proof.provider != provider_name or not proof.eligible():
                continue
            if adapter.model != proof.model:
                continue
            action_key = f"{self.record.manifest.task_id}:{phase}:{provider_name}:{adapter.model}"
            if action_key in self.record.action_keys:
                previous = next(
                    (a for a in reversed(self.record.attempts)
                     if a.provider == provider_name and a.model == adapter.model),
                    None,
                )
                if previous and previous.error_kind == FailureKind.QUOTA_EXHAUSTED.value:
                    continue
                raise DuplicateExecution("provider_phase_already_claimed")
            self.record.action_keys.add(action_key)
            self.store.write(self.record)
            try:
                result = adapter.execute(prompt)
            except ProviderFailure as exc:
                self.record.attempts.append(AttemptEvidence(
                    provider_name, adapter.model, "failed", exc.kind.value, exc.code
                ))
                self.store.write(self.record)
                attempts_used += 1
                can_switch = (
                    exc.kind is FailureKind.QUOTA_EXHAUSTED
                    and exc.outcome_known
                    and not exc.output_committed
                    and self.record.output_sha256 is None
                )
                if not can_switch:
                    if self.record.state is State.BUILDING:
                        self.record.transition(State.BLOCKED, checkpoint_step="provider_failure_blocked")
                    self.store.write(self.record)
                    raise
                if self.record.state is State.BUILDING:
                    self.record.transition(State.CHECKPOINTED, checkpoint_step="quota_failure_checkpointed")
                self.store.write(self.record)
                continue
            output_hash = hashlib.sha256(result.text.encode("utf-8")).hexdigest()
            self.record.attempts.append(AttemptEvidence(
                provider_name, adapter.model, "success", duration_seconds=result.duration_seconds
            ))
            self.record.output_sha256 = output_hash
            self.record.last_successful_step = "provider_output_received"
            self.store.write(self.record)
            return result
        if self.record.state in {State.BUILDING, State.CHECKPOINTED}:
            self.record.transition(State.BLOCKED, checkpoint_step="no_eligible_provider")
        self.store.write(self.record)
        raise ExecutorError("no_eligible_provider")

    def mark_verified(self, output: str, *, changed_paths: Sequence[str], commit_sha: str) -> None:
        if self.record.state not in {State.BUILDING, State.CHECKPOINTED}:
            raise InvalidTransition("verification_from_invalid_state")
        if self.record.output_sha256 != hashlib.sha256(output.encode("utf-8")).hexdigest():
            raise ExecutorError("output_checkpoint_hash_mismatch")
        validate_changed_paths(changed_paths, self.record.manifest.scope_paths)
        if not exact_sha(commit_sha):
            raise ExecutorError("commit_sha_must_be_exact")
        self.record.commit_sha = commit_sha
        self.record.transition(State.VERIFYING, checkpoint_step="output_and_scope_verified")
        self.store.write(self.record)

    def mark_review_handoff(self) -> None:
        if self.record.state is not State.VERIFYING:
            raise InvalidTransition("review_handoff_requires_verified_output")
        self.record.transition(State.REVIEWING, checkpoint_step="independent_review_requested")
        self.store.write(self.record)


def environment_key(name: str) -> Callable[[], str]:
    """Return a lazy secret supplier; secret values are never serialized or logged."""
    return lambda: os.environ.get(name, "")
