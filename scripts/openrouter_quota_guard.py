#!/usr/bin/env python3
"""Fail-closed OpenRouter free-request budget gate. No secrets are logged."""
import json, os, sys, urllib.request, urllib.error
key = os.environ.get("OPENROUTER_API_KEY", "")
if not key:
    print("QUOTA_BLOCKED: missing API key")
    sys.exit(2)
request = urllib.request.Request(
    "https://openrouter.ai/api/v1/key",
    headers={"Authorization": "Bearer " + key, "Accept": "application/json"},
)
try:
    with urllib.request.urlopen(request, timeout=15) as response:
        data = json.load(response).get("data", {})
except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as error:
    print("QUOTA_BLOCKED: key quota unavailable (" + type(error).__name__ + ")")
    sys.exit(2)
# The key endpoint's usage_daily and limit_remaining are monetary values,
# NOT free-model request counts. Never mistake them for request quota.
# Fail closed until a verified request-count source is integrated.
request_remaining = data.get("free_requests_remaining")
if not isinstance(request_remaining, int) or isinstance(request_remaining, bool) or request_remaining < 0:
    print("QUOTA_BLOCKED: verified daily free-request counter unavailable; no model call made")
    sys.exit(2)
reserve = int(os.environ.get("ADOS_QUOTA_RESERVE", "10"))
if request_remaining <= reserve:
    print("QUOTA_BLOCKED: free request reserve reached")
    sys.exit(2)
print("QUOTA_OK: free request budget available")
