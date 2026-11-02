"""Build a checked provenance record for a pull-request validation run."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_SHA = re.compile(r"^[0-9a-f]{40}$")


def provenance(checked_out_sha: str, expected_head_sha: str, merge_sha: str, run_id: str) -> dict[str, str]:
    if not _SHA.fullmatch(checked_out_sha) or not _SHA.fullmatch(expected_head_sha) or not _SHA.fullmatch(merge_sha):
        raise ValueError("source_provenance_sha_invalid")
    if checked_out_sha != expected_head_sha:
        raise ValueError("checked_out_sha_does_not_match_pr_head")
    if not run_id.isdigit():
        raise ValueError("source_provenance_run_id_invalid")
    return {
        "source_commit_sha": checked_out_sha,
        "expected_pr_head_sha": expected_head_sha,
        "github_synthetic_merge_sha": merge_sha,
        "workflow_run_id": run_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checked-out-sha", required=True)
    parser.add_argument("--expected-head-sha", required=True)
    parser.add_argument("--merge-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = provenance(args.checked_out_sha, args.expected_head_sha, args.merge_sha, args.run_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("SOURCE_PROVENANCE: PASS; source=" + args.checked_out_sha + "; merge=" + args.merge_sha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
