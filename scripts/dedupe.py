#!/usr/bin/env python3
"""
dedupe.py -- recruiter-prompt-bank deduplication (curation step, spec §5/§7).

Normalises record `text`, then drops exact and near-duplicates
(token Jaccard >= 0.9 OR difflib.SequenceMatcher ratio >= 0.92),
keeping the record with the earliest `date_published` (else the
first one encountered in input order). The kept record gets a
`notes` addition: "also at <other urls>" listing the source_url(s)
of the record(s) it absorbed.

Usage:
    python3 scripts/dedupe.py --in in1.jsonl [in2.jsonl ...] \
        --out out_kept.jsonl --rejects out_dupe_rejects.jsonl

Each line of an --in file is a JSON object with at least: text, source_url,
date_published (optional, ISO date or null). Records are read in the order
given (file order, then line order within each file); that order should
already reflect any prior scope-filter / relabel pass, since this script
only performs deduplication. This is the curation step (`make dedupe`); it
is not part of `make all`.
"""
import argparse
import sys
import json
import re
import difflib
from datetime import date


def normalise(text: str) -> str:
    """Lowercase, collapse whitespace, strip quotes/punctuation for comparison."""
    t = text.lower()
    t = re.sub(r"[‘’“”'\"`]", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def is_near_duplicate(a: str, b: str) -> bool:
    if a == b:
        return True
    if token_jaccard(a, b) >= 0.9:
        return True
    if difflib.SequenceMatcher(None, a, b).ratio() >= 0.92:
        return True
    return False


def sort_key_for_keeper(rec):
    """Earlier date_published wins; records without a date sort last
    among candidates but ties are broken by original input order
    (index), which is what 'else the first seen' means here."""
    dp = rec.get("date_published")
    if dp:
        try:
            y, m, d = [int(x) for x in dp.split("-")]
            return (0, date(y, m, d))
        except Exception:
            pass
    return (1, date.max)


def dedupe(records):
    """records: list of dicts, each with '_orig_idx' set to its
    original input order. Returns (kept, dropped) where dropped is a
    list of (record, keeper_record) pairs."""
    n = len(records)
    norm = [normalise(r.get("text", "")) for r in records]

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    # naive O(n^2) comparison -- fine at this scale (hundreds of records)
    for i in range(n):
        for j in range(i + 1, n):
            if is_near_duplicate(norm[i], norm[j]):
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    kept = []
    dropped = []
    for root, members in clusters.items():
        if len(members) == 1:
            kept.append(records[members[0]])
            continue
        # pick keeper: earliest date_published, tie-break by original order
        ordered = sorted(
            members,
            key=lambda idx: (sort_key_for_keeper(records[idx]), records[idx]["_orig_idx"]),
        )
        keeper_idx = ordered[0]
        keeper = dict(records[keeper_idx])
        others = [records[idx] for idx in ordered[1:]]
        other_urls = sorted({o.get("source_url", "") for o in others if o.get("source_url")})
        if other_urls:
            note_add = "also at " + ", ".join(other_urls)
            existing = keeper.get("notes")
            keeper["notes"] = f"{existing}; {note_add}" if existing else note_add
        kept.append(keeper)
        for o in others:
            dropped.append((o, keeper))
    return kept, dropped


def load_records(in_paths):
    records = []
    idx = 0
    for in_path in in_paths:
        with open(in_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_orig_idx"] = idx
                records.append(rec)
                idx += 1
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_paths", nargs="+", required=True, help="one or more input JSONL files (read in order given)")
    parser.add_argument("--out", dest="out_path", required=True, help="path to write kept (deduplicated) records to")
    parser.add_argument("--rejects", dest="rejects_path", required=True, help="path to write dropped duplicate records to")
    args = parser.parse_args()

    records = load_records(args.in_paths)

    kept, dropped = dedupe(records)
    # restore original relative order among kept records
    kept.sort(key=lambda r: r["_orig_idx"])

    with open(args.out_path, "w") as f:
        for r in kept:
            r = {k: v for k, v in r.items() if k != "_orig_idx"}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(args.rejects_path, "w") as f:
        for rec, keeper in dropped:
            out = {k: v for k, v in rec.items() if k != "_orig_idx"}
            out["reject_reason"] = f"duplicate_of: {keeper.get('id', keeper.get('source_url'))}"
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"kept: {len(kept)}  dropped as duplicates: {len(dropped)}")


if __name__ == "__main__":
    main()
