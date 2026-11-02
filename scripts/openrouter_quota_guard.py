#!/usr/bin/env python3
"""Fail-closed gate based on durable provider-error evidence.

OpenRouter's documented key endpoint exposes monetary usage/limits, not a
free-request counter. This gate never calls that endpoint and never infers
request availability from monetary fields.
"""
import argparse
import json
from pathlib import Path
import sys


def classify(status):
    if not isinstance(status, dict) or status.get("task_id") != "ADOS-POC-002":
        return "BLOCKED_QUOTA: task quota checkpoint missing or invalid"
    if status.get("state") == "BLOCKED_QUOTA":
        evidence = status.get("evidence")
        if (
            isinstance(evidence, dict)
            and evidence.get("provider") == "OpenRouter"
            and evidence.get("error_class") == "free-models-per-day"
            and isinstance(evidence.get("workflow_run_id"), int)
            and not isinstance(evidence.get("workflow_run_id"), bool)
            and evidence["workflow_run_id"] > 0
        ):
            return (
                "BLOCKED_QUOTA: OpenRouter confirmed free-model daily limit "
                f"in workflow run #{evidence['workflow_run_id']}; no retry or paid fallback"
            )
        return "BLOCKED_QUOTA: quota checkpoint lacks verifiable provider evidence"
    return (
        "BLOCKED_QUOTA: no supported authoritative free-request counter is available; "
        "monetary key limits are not request allowances"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-file", required=True)
    args = parser.parse_args(argv)
    try:
        status = json.loads(Path(args.status_file).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        print("BLOCKED_QUOTA: durable quota checkpoint unreadable; no model call made")
        return 2
    decision = classify(status)
    print(decision)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
