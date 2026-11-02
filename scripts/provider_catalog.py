"""Machine-readable provider candidates and fail-closed eligibility metadata.

This module never treats a candidate endpoint/model as proof of free usage.
Cloud adapters are dormant until status evidence and owner approval authorize them.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

import ados_executor as ae


class EligibilityState(str, Enum):
    VERIFIED_FREE = "VERIFIED_FREE"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    AUTH_UNAVAILABLE = "AUTH_UNAVAILABLE"
    BILLING_UNVERIFIED = "BILLING_UNVERIFIED"
    INCOMPATIBLE = "INCOMPATIBLE"
    DISABLED = "DISABLED"


@dataclass(frozen=True)
class ProviderCapabilities:
    coding: bool
    structured_output: bool
    max_context_tokens: int | None = None
    max_output_tokens: int | None = None
    source: str = "UNKNOWN"


@dataclass(frozen=True)
class ProviderRequest:
    task_id: str
    model: str
    prompt: str
    max_output_tokens: int = 2048
    structured_output: bool = False

    def __post_init__(self) -> None:
        ae.validate_task_id(self.task_id)
        if not self.model or not self.prompt.strip() or not 1 <= self.max_output_tokens <= 8192:
            raise ValueError("invalid_provider_request")


@dataclass(frozen=True)
class ProviderStatusRecord:
    provider: str
    model: str
    eligibility: EligibilityState
    last_verified_at: str | None
    quota_status: str
    quota_evidence: str | None
    quota_source: str | None
    last_failure_classification: str | None
    retry_eligible: bool
    capabilities: ProviderCapabilities
    cost_authorized_usd: float
    api_endpoint: str | None
    auth_method: str | None
    required_secret: str | None
    note: str

    def __post_init__(self) -> None:
        if self.cost_authorized_usd != 0.0:
            raise ValueError("provider_cost_authorization_must_be_zero")
        if self.quota_status not in {"AVAILABLE", "EXHAUSTED", "UNKNOWN", "NOT_APPLICABLE"}:
            raise ValueError("invalid_quota_status")
        if self.eligibility is EligibilityState.VERIFIED_FREE:
            if not self.last_verified_at or not self.capabilities.coding:
                raise ValueError("verified_provider_requires_timestamp_and_coding_capability")
            if self.quota_status not in {"AVAILABLE", "NOT_APPLICABLE"}:
                raise ValueError("verified_provider_requires_available_quota")

    @classmethod
    def unverified(
        cls, provider: str, model: str, state: EligibilityState, *,
        endpoint: str | None, auth_method: str | None, secret: str | None,
        note: str, coding: bool = True, structured: bool = False,
        context: int | None = None, output: int | None = None,
    ) -> "ProviderStatusRecord":
        return cls(provider, model, state, None, "UNKNOWN", None, None, None,
                   False, ProviderCapabilities(coding, structured, context, output),
                   0.0, endpoint, auth_method, secret, note)

    def as_json(self) -> dict[str, object]:
        return asdict(self) | {"eligibility": self.eligibility.value}


class OpenAICompatibleAdapter:
    """Single-request OpenAI-compatible adapter; eligibility is checked elsewhere."""

    def __init__(self, name: str, endpoint: str, model: str,
                 key_supplier: Callable[[], str], *, timeout: float = 60.0) -> None:
        if not endpoint.startswith("https://"):
            raise ValueError("cloud_endpoint_must_use_https")
        self.name, self.endpoint, self.model = name, endpoint.rstrip("/"), model
        self._key_supplier, self.timeout = key_supplier, timeout

    def execute(self, prompt: str) -> ae.ExecutionResult:
        return self.execute_request(ProviderRequest("ADOS-POC-999", self.model, prompt))

    def execute_request(self, request: ProviderRequest) -> ae.ExecutionResult:
        if request.model != self.model:
            raise ValueError("request_model_does_not_match_adapter")
        key = self._key_supplier()
        if not key:
            raise ae.ProviderFailure(self.name, ae.FailureKind.AUTHENTICATION,
                                     outcome_known=True, code="missing_secret")
        started = ae.time.monotonic()
        status, payload = ae._http_json(
            self.endpoint + "/chat/completions",
            {"Authorization": f"Bearer {key}"},
            {"model": self.model, "messages": [{"role": "user", "content": request.prompt}],
             "max_tokens": request.max_output_tokens, "temperature": 0},
            self.timeout,
        )
        if status != 200:
            kind, code = _classify_openai_compatible_error(status, payload)
            raise ae.ProviderFailure(self.name, kind,
                outcome_known=kind not in {ae.FailureKind.TIMEOUT_UNKNOWN,
                                           ae.FailureKind.TRANSPORT_UNKNOWN}, code=code)
        try:
            text = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage", {})
        except (KeyError, IndexError, TypeError) as exc:
            raise ae.ProviderFailure(self.name, ae.FailureKind.INVALID_RESPONSE,
                                     outcome_known=True, code="response_shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise ae.ProviderFailure(self.name, ae.FailureKind.INVALID_RESPONSE,
                                     outcome_known=True, code="empty_output")
        usage = usage if isinstance(usage, dict) else {}
        return ae.ExecutionResult(self.name, self.model, text, 1,
            ae.time.monotonic() - started,
            usage.get("prompt_tokens") if isinstance(usage.get("prompt_tokens"), int) else None,
            usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None)


def _classify_openai_compatible_error(status: int, payload: dict[str, object]):
    error = payload.get("error")
    if not isinstance(error, dict):
        return ae.FailureKind.INVALID_RESPONSE, "malformed_provider_error"
    code = str(error.get("code", error.get("type", "")))
    if status in (401, 403):
        return ae.FailureKind.AUTHENTICATION, code or f"http_{status}"
    if status == 429:
        # Without provider-specific structured quota semantics, fail closed.
        return ae.FailureKind.RATE_LIMIT_UNKNOWN, code or "http_429"
    if status >= 500:
        return ae.FailureKind.TRANSPORT_UNKNOWN, code or f"http_{status}"
    return ae.FailureKind.POLICY, code or f"http_{status}"


class GroqCloudAdapter(OpenAICompatibleAdapter):
    def __init__(self, model: str, key_supplier: Callable[[], str], *, timeout: float = 60.0):
        super().__init__("groq", "https://api.groq.com/openai/v1", model, key_supplier, timeout=timeout)


class NvidiaNimAdapter(OpenAICompatibleAdapter):
    def __init__(self, model: str, key_supplier: Callable[[], str], *, timeout: float = 60.0):
        super().__init__("nvidia_nim", "https://integrate.api.nvidia.com/v1", model,
                         key_supplier, timeout=timeout)


class GitHubModelsAdapter:
    """Retired service marker. It is intentionally impossible to invoke."""
    name = "github_models"
    model = "RETIRED"

    def execute(self, prompt: str) -> ae.ExecutionResult:
        raise ae.ProviderFailure(self.name, ae.FailureKind.UNAVAILABLE,
                                 outcome_known=True, code="service_retired")

    def execute_request(self, request: ProviderRequest) -> ae.ExecutionResult:
        return self.execute(request.prompt)


def default_status_records() -> dict[str, ProviderStatusRecord]:
    """Public docs identify candidates; live account/billing evidence remains absent."""
    rows = [
        ProviderStatusRecord.unverified("local", "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
            EligibilityState.DISABLED, endpoint="http://127.0.0.1:8090/v1",
            auth_method="none_loopback", secret=None,
            note="Awaiting one-shot local feasibility run."),
        ProviderStatusRecord.unverified("gemini", "gemini-3.5-flash-lite",
            EligibilityState.BILLING_UNVERIFIED, endpoint="https://generativelanguage.googleapis.com/v1beta",
            auth_method="x-goog-api-key", secret="GEMINI_API_KEY",
            note="Public pricing lists free and paid tiers; repository key project billing is not observable here."),
        ProviderStatusRecord("openrouter", "openrouter/free", EligibilityState.QUOTA_EXHAUSTED,
            None, "EXHAUSTED", "used=52; limit=50; remaining=0",
            "workflow run 41448677323 in chinkk-git/aaria-ados-poc",
            "QUOTA_EXHAUSTED", False,
            ProviderCapabilities(True, False, None, 2048, "existing route and failed task evidence"),
            0.0, "https://openrouter.ai/api/v1", "Bearer", "OPENROUTER_API_KEY",
            "No reset time was reported; do not retry until a new authoritative reading exists."),
        ProviderStatusRecord.unverified("groq", "openai/gpt-oss-20b",
            EligibilityState.BILLING_UNVERIFIED, endpoint="https://api.groq.com/openai/v1",
            auth_method="Bearer", secret="GROQ_API_KEY",
            note="Official API/model documented; model has published paid pricing and account free-tier eligibility is unverified."),
        ProviderStatusRecord.unverified("nvidia_nim", "openai/gpt-oss-20b",
            EligibilityState.AUTH_UNAVAILABLE, endpoint="https://integrate.api.nvidia.com/v1",
            auth_method="Bearer", secret="NVIDIA_API_KEY",
            note="Official free API endpoint documented; no authorized credential/account verification is available."),
        ProviderStatusRecord.unverified("github_models", "RETIRED",
            EligibilityState.DISABLED, endpoint=None, auth_method=None, secret=None,
            note="GitHub Models inference service retired July 30, 2026."),
    ]
    return {row.provider: row for row in rows}


def verified_free_model_map(records: dict[str, ProviderStatusRecord]) -> dict[str, str]:
    """Return only exact models whose recorded evidence allows zero-cost use."""
    return {
        name: row.model
        for name, row in records.items()
        if row.eligibility is EligibilityState.VERIFIED_FREE
        and row.cost_authorized_usd == 0.0
        and row.quota_status in {"AVAILABLE", "NOT_APPLICABLE"}
        and row.last_verified_at is not None
        and row.capabilities.coding
    }


def write_status_file(path, records: dict[str, ProviderStatusRecord]) -> None:
    payload = {name: record.as_json() for name, record in sorted(records.items())}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

