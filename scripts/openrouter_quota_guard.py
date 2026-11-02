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
# Free-tier daily cap: conservative upper bound; never assume a larger quota.
# A provider may report fractional usage; round up to protect the reserve.
import math
used = data.get("usage_daily")
limit = data.get("limit")
remaining = data.get("limit_remaining")
is_free = data.get("is_free_tier")
if is_free is not True:
    print("QUOTA_BLOCKED: cannot verify free-tier account")
    sys.exit(2)
if not isinstance(used, (int, float)) or isinstance(used, bool) or not math.isfinite(used) or used < 0:
    print("QUOTA_BLOCKED: daily usage counter unavailable")
    sys.exit(2)
if remaining is not None and (not isinstance(remaining, (int, float)) or isinstance(remaining, bool) or not math.isfinite(remaining)):
    print("QUOTA_BLOCKED: invalid remaining counter")
    sys.exit(2)
daily_remaining = max(0, 50 - math.ceil(used))
if remaining is not None:
    daily_remaining = min(daily_remaining, max(0, math.floor(remaining)))
reserve = int(os.environ.get("ADOS_QUOTA_RESERVE", "10"))
if daily_remaining <= reserve:
    print("QUOTA_BLOCKED: insufficient free daily budget; retry only after verified quota recovery")
    sys.exit(2)
print("QUOTA_OK: sufficient free daily budget with reserved review capacity")
