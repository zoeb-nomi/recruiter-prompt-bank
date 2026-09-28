#!/usr/bin/env python3
"""
validate.py — schema + integrity check for data/prompts.jsonl

Per spec §3/§8, this script checks (no network calls):
  - every required key is present and every enum value is legal
  - source_url is a well-formed https URL (actually resolving the URL is the
    curator's job upstream, not this script's)
  - no duplicate ids
  - no near-duplicate records (difflib ratio >= 0.92 on normalised text)

It tolerates records that lack the optional `id` / `url_ok` keys (raw,
pre-curation collector output) so it can also be run against a fixture built
straight from the collectors' output — those two keys just aren't checked
when absent.

Usage:
    python scripts/validate.py --in data/prompts.jsonl

Exit code: 0 if every check passes, 1 otherwise. A human-readable
pass/fail report with counts is printed to stdout either way.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Schema, per spec §3. `id` and the curator-added `url_ok` / `text_found` /
# `has_placeholder` fields are treated as optional: they are present in the
# curated data/prompts.jsonl but absent from raw collector output, and this
# script is also run against a raw fixture for testing.
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {
    "kind",
    "text",
    "verbatim",
    "source_url",
    "source_type",
    "author_role",
    "role_family",
    "date_observed",
    "collector",
}

# Keys that must be present in the schema but may hold null.
NULLABLE_KEYS = {"tool", "seniority_hint", "geo_hint", "date_published", "notes"}

OPTIONAL_KEYS = {"id", "url_ok", "text_found", "has_placeholder"}

ALL_KNOWN_KEYS = REQUIRED_KEYS | NULLABLE_KEYS | OPTIONAL_KEYS

ENUMS = {
    "kind": {"sourcing_query", "screening_rubric", "lookup_prompt", "ranking_prompt"},
    "source_type": {
        "vendor_docs",
        "vendor_blog",
        "recruiter_post",
        "community",
        "article",
        "prompt_library",
        "newsletter",
    },
    "author_role": {"vendor", "recruiter", "hiring_manager", "sourcer", "unknown"},
    "role_family": {"pm", "engineering", "generic", "other"},
}

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

NEAR_DUP_THRESHOLD = 0.92


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding quotes — for
    near-duplicate comparison only. Does not mutate the stored record."""
    t = text.strip().lower()
    t = t.strip("\"'“”‘’")
    t = re.sub(r"\s+", " ", t)
    return t


def is_https_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(parsed.netloc)


def load_records(path: str) -> tuple[list[dict], list[str]]:
    """Read a JSONL file. Returns (records, parse_errors)."""
    records = []
    errors = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {lineno}: invalid JSON ({exc})")
                continue
            if not isinstance(obj, dict):
                errors.append(f"line {lineno}: record is not a JSON object")
                continue
            obj["_lineno"] = lineno
            records.append(obj)
    return records, errors


def check_schema(records: list[dict]) -> list[str]:
    problems = []
    for rec in records:
        lineno = rec["_lineno"]
        rid = rec.get("id", f"line {lineno}")

        missing = REQUIRED_KEYS - rec.keys()
        if missing:
            problems.append(f"{rid}: missing required key(s): {sorted(missing)}")

        unknown = set(rec.keys()) - ALL_KNOWN_KEYS - {"_lineno"}
        if unknown:
            problems.append(f"{rid}: unknown key(s): {sorted(unknown)}")

        for key, allowed in ENUMS.items():
            if key in rec and rec[key] not in allowed:
                problems.append(
                    f"{rid}: {key}={rec[key]!r} not in allowed set {sorted(allowed)}"
                )

        if "verbatim" in rec and not isinstance(rec["verbatim"], bool):
            problems.append(f"{rid}: verbatim must be a bool, got {rec['verbatim']!r}")

        if "text" in rec and (not isinstance(rec["text"], str) or not rec["text"].strip()):
            problems.append(f"{rid}: text must be a non-empty string")

        if "source_url" in rec:
            if not isinstance(rec["source_url"], str) or not is_https_url(rec["source_url"]):
                problems.append(
                    f"{rid}: source_url is not a well-formed https URL: {rec['source_url']!r}"
                )

        if "date_observed" in rec:
            if not isinstance(rec["date_observed"], str) or not ISO_DATE_RE.match(
                rec["date_observed"]
            ):
                problems.append(
                    f"{rid}: date_observed is not an ISO date: {rec['date_observed']!r}"
                )

        if rec.get("date_published") is not None and "date_published" in rec:
            if not isinstance(rec["date_published"], str) or not ISO_DATE_RE.match(
                rec["date_published"]
            ):
                problems.append(
                    f"{rid}: date_published is not null or an ISO date: {rec['date_published']!r}"
                )

    return problems


def check_duplicate_ids(records: list[dict]) -> list[str]:
    ids = [r["id"] for r in records if "id" in r]
    counts = Counter(ids)
    dupes = [i for i, c in counts.items() if c > 1]
    return [f"duplicate id {i!r} appears {counts[i]} times" for i in sorted(dupes)]


def check_near_duplicates(records: list[dict]) -> list[str]:
    """O(n^2) SequenceMatcher comparison. Fine at this dataset's scale
    (hundreds, not tens of thousands, of records); if the bank grows a lot
    this should bucket by length or a cheap hash first."""
    problems = []
    normed = []
    for rec in records:
        text = rec.get("text", "")
        normed.append(normalise(text) if isinstance(text, str) else "")

    n = len(records)
    for i in range(n):
        if not normed[i]:
            continue
        for j in range(i + 1, n):
            if not normed[j]:
                continue
            # Cheap length pre-filter before the expensive ratio() call.
            li, lj = len(normed[i]), len(normed[j])
            if li == 0 or lj == 0:
                continue
            if min(li, lj) / max(li, lj) < NEAR_DUP_THRESHOLD - 0.05:
                continue
            ratio = SequenceMatcher(None, normed[i], normed[j]).ratio()
            if ratio >= NEAR_DUP_THRESHOLD:
                a = records[i].get("id", f"line {records[i]['_lineno']}")
                b = records[j].get("id", f"line {records[j]['_lineno']}")
                problems.append(f"near-duplicate ({ratio:.3f}): {a} ~ {b}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="path to prompts.jsonl")
    args = parser.parse_args()

    records, parse_errors = load_records(args.in_path)

    schema_problems = check_schema(records) if records else []
    dup_id_problems = check_duplicate_ids(records) if records else []
    near_dup_problems = check_near_duplicates(records) if records else []

    kind_counts = Counter(r.get("kind", "?") for r in records)
    source_type_counts = Counter(r.get("source_type", "?") for r in records)
    has_placeholder_counts = Counter(
        r.get("kind", "?") for r in records if r.get("has_placeholder") is True
    )

    total_problems = (
        len(parse_errors)
        + len(schema_problems)
        + len(dup_id_problems)
        + len(near_dup_problems)
    )

    print("=" * 70)
    print(f"validate.py — {args.in_path}")
    print("=" * 70)
    print(f"records read:            {len(records)}")
    print(f"by kind:                 {dict(kind_counts)}")
    print(f"by source_type:          {dict(source_type_counts)}")
    print(f"has_placeholder=true by kind: {dict(has_placeholder_counts)}")
    print()
    print(f"parse errors:            {len(parse_errors)}")
    print(f"schema problems:         {len(schema_problems)}")
    print(f"duplicate ids:           {len(dup_id_problems)}")
    print(f"near-duplicate pairs:    {len(near_dup_problems)}")
    print()

    if parse_errors:
        print("-- parse errors --")
        for p in parse_errors[:50]:
            print(f"  {p}")
        if len(parse_errors) > 50:
            print(f"  ... and {len(parse_errors) - 50} more")

    if schema_problems:
        print("-- schema problems --")
        for p in schema_problems[:50]:
            print(f"  {p}")
        if len(schema_problems) > 50:
            print(f"  ... and {len(schema_problems) - 50} more")

    if dup_id_problems:
        print("-- duplicate ids --")
        for p in dup_id_problems[:50]:
            print(f"  {p}")

    if near_dup_problems:
        print("-- near-duplicates --")
        for p in near_dup_problems[:50]:
            print(f"  {p}")
        if len(near_dup_problems) > 50:
            print(f"  ... and {len(near_dup_problems) - 50} more")

    print()
    if total_problems == 0:
        print("RESULT: PASS")
    else:
        print(f"RESULT: FAIL ({total_problems} problem(s))")
    print("=" * 70)

    return 1 if total_problems > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
