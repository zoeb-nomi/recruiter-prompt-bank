#!/usr/bin/env python3
"""
mine.py — deterministic mining over data/prompts.jsonl (spec §6).

No LLM judge anywhere in this script. Everything here is either a regex/
keyword-list extractor, a frequency count, or an embedding + clustering
pipeline; clusters are described (top TF-IDF terms + example ids), never
scored or labelled by a model's opinion.

Pipeline:
  (a) slot extraction — job titles, skills/keywords, companies,
      years-of-experience, locations, education, seniority, must/should/
      exclude operators, requested output shape, evidence demand.
  (b) frequency tables — slot x kind, operator x kind, output-shape x kind,
      evidence-demand rate by kind and by author_role, verbatim rate by
      source_type.
  (c) clustering — sentence-transformers all-MiniLM-L6-v2 embeddings if the
      package/model are available, else a TF-IDF fallback; HDBSCAN if
      installable, else agglomerative clustering with a fixed distance
      threshold. Fixed random seeds throughout. Each cluster is labelled by
      its top-5 TF-IDF terms and three example record ids.
  (d) writes reports/frequency_tables.md, reports/clusters.md,
      reports/summary.json.

Usage:
    python scripts/mine.py --in data/prompts.jsonl --out reports/

`--offline` defaults to True: by default this script never touches the
network and goes straight to the TF-IDF embedding fallback. Pass
`--no-offline` to let it try sentence-transformers/all-MiniLM-L6-v2 (one
outbound call to huggingface.co unless the model is already cached), pinned
to MINILM_REVISION below when that constant is set; it still falls back to
TF-IDF automatically if the package or the download is unavailable.

Tolerates records missing the optional `id` / `url_ok` keys (raw,
pre-curation collector output) — a positional fallback id (`row<N>`) is used
for reporting when `id` is absent.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Pinned revision (commit SHA) for the sentence-transformers model
# "all-MiniLM-L6-v2" on the Hugging Face Hub. None means "unpinned": a
# --no-offline run fetches whatever revision the Hub currently serves as
# that model's default branch, which is not reproducible across time. To
# pin it, look up a commit SHA at
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/commits/main
# and set this constant to that SHA (e.g. "abcd1234...").
MINILM_REVISION = None

try:
    import numpy as np

    np.random.seed(RANDOM_SEED)
except ImportError:
    np = None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.setdefault("id", f"row{lineno}")
            obj["_lineno"] = lineno
            records.append(obj)
    return records


# ---------------------------------------------------------------------------
# (a) Slot extraction — regex / keyword-list heuristics.
#
# These are deliberately simple, auditable heuristics, not a trained NER
# model, per spec §6. Each returns the list of surface matches found in one
# record's text; frequency tables downstream care mainly about presence
# (did this record mention a company at all?) rather than exhaustively
# cataloguing every mention.
# ---------------------------------------------------------------------------

JOB_TITLE_SEEDS = [
    "product manager", "software engineer", "data scientist", "machine learning engineer",
    "ml engineer", "backend engineer", "front end engineer", "frontend engineer",
    "full stack engineer", "fullstack engineer", "devops engineer", "site reliability engineer",
    "data engineer", "sales engineer", "solutions engineer", "engineering manager",
    "technical program manager", "program manager", "project manager", "product designer",
    "ux designer", "ui designer", "designer", "recruiter", "sourcer", "talent acquisition",
    "hr business partner", "account executive", "sales representative", "customer success manager",
    "marketing manager", "growth marketer", "financial analyst", "business analyst",
    "operations manager", "general manager", "vp of engineering", "cto", "cfo", "ceo",
    "director of engineering", "head of product", "staff engineer", "principal engineer",
    "security engineer", "qa engineer", "test engineer", "mobile engineer", "ios engineer",
    "android engineer", "cloud engineer", "platform engineer", "infrastructure engineer",
    "data analyst", "business development", "customer support", "technical writer",
    "content writer", "research scientist", "applied scientist", "solutions architect",
    "systems administrator", "network engineer", "database administrator",
]
JOB_TITLE_SUFFIX_RE = re.compile(
    r"\b([A-Z][a-zA-Z]*(?:[\s/&-][A-Z][a-zA-Z]*){0,3}\s"
    r"(?:Engineer|Manager|Developer|Designer|Scientist|Analyst|Director|Specialist|"
    r"Lead|Architect|Recruiter|Sourcer|Administrator))\b"
)


def extract_job_titles(text: str) -> list[str]:
    lower = text.lower()
    found = {seed for seed in JOB_TITLE_SEEDS if seed in lower}
    found |= {m.strip() for m in JOB_TITLE_SUFFIX_RE.findall(text)}
    return sorted(found)


SKILL_SEEDS = [
    "python", "java", "javascript", "typescript", "sql", "nosql", "aws", "azure", "gcp",
    "kubernetes", "docker", "react", "node.js", "nodejs", "go", "golang", "rust", "c++",
    "machine learning", "deep learning", "nlp", "llm", "llms", "roadmapping", "agile",
    "scrum", "saas", "b2b", "b2c", "crm", "ats", "salesforce", "figma", "a/b testing",
    "excel", "tableau", "power bi", "powerbi", "spark", "hadoop", "airflow", "terraform",
    "ci/cd", "rest api", "graphql", "swift", "kotlin", "django", "flask", "postgresql",
    "mongodb", "redis", "elasticsearch", "boolean search", "sourcing", "negotiation",
    "stakeholder management", "data analysis", "product strategy", "user research",
]


def extract_skills(text: str) -> list[str]:
    lower = text.lower()
    return sorted({seed for seed in SKILL_SEEDS if seed in lower})


COMPANY_SEEDS = [
    "google", "meta", "amazon", "microsoft", "apple", "netflix", "linkedin", "salesforce",
    "stripe", "uber", "airbnb", "openai", "anthropic", "greenhouse", "ashby", "lever",
    "workable", "eightfold", "hiredscore", "workday", "paradox", "manatal", "teamtailor",
    "zoho", "hireez", "seekout", "apollo", "gem", "findem", "fetcher", "juicebox", "dover",
]
COMPANY_AT_RE = re.compile(
    r"\bat\s+([A-Z][\w&.'-]*(?:\s[A-Z][\w&.'-]*){0,3})\b"
)
COMPANY_SUFFIX_RE = re.compile(
    r"\b([A-Z][\w&.'-]*(?:\s[A-Z][\w&.'-]*){0,3}\s"
    r"(?:Inc|Corp|LLC|Ltd|Co|Technologies|Systems|Labs|Group))\b"
)


def extract_companies(text: str) -> list[str]:
    lower = text.lower()
    found = {seed for seed in COMPANY_SEEDS if re.search(rf"\b{re.escape(seed)}\b", lower)}
    found |= {m.strip() for m in COMPANY_AT_RE.findall(text)}
    found |= {m.strip() for m in COMPANY_SUFFIX_RE.findall(text)}
    return sorted(found)


YEARS_EXP_RE = re.compile(
    r"\b(\d{1,2})\+?\s*(?:-|to)\s*(\d{1,2})\+?\s*years?\b|\b(\d{1,2})\+?\s*years?"
    r"(?:\s+of\s+experience)?\b",
    re.IGNORECASE,
)


def extract_years_experience(text: str) -> list[str]:
    matches = []
    for m in YEARS_EXP_RE.finditer(text):
        matches.append(m.group(0))
    return matches


LOCATION_SEEDS = [
    "remote", "hybrid", "onsite", "united states", "usa", "us", "uk", "united kingdom",
    "canada", "india", "germany", "france", "singapore", "australia", "new york",
    "san francisco", "bay area", "austin", "seattle", "boston", "chicago", "los angeles",
    "london", "berlin", "toronto", "bengaluru", "bangalore", "mumbai", "delhi", "remote-us",
    "emea", "apac", "latam",
]
LOCATION_IN_RE = re.compile(r"\bin\s+([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+){0,2})\b")


def extract_locations(text: str) -> list[str]:
    lower = text.lower()
    found = {seed for seed in LOCATION_SEEDS if re.search(rf"\b{re.escape(seed)}\b", lower)}
    found |= {m.strip() for m in LOCATION_IN_RE.findall(text)}
    return sorted(found)


EDUCATION_WORDS = [
    "bachelor's", "bachelor", "master's", "master", "mba", "phd", "ph.d.", "degree",
    "university", "college", "b.s.", "m.s.", "b.a.", "m.a.", "computer science",
    "undergraduate", "graduate degree",
]


def extract_education(text: str) -> list[str]:
    lower = text.lower()
    return sorted({w for w in EDUCATION_WORDS if w in lower})


SENIORITY_WORDS = [
    "entry-level", "entry level", "junior", "mid-level", "mid level", "senior", "staff",
    "principal", "lead", "director", "vp", "vice president", "head of", "chief",
    "executive", "associate", "intern",
]


def extract_seniority(text: str) -> list[str]:
    lower = text.lower()
    return sorted({w for w in SENIORITY_WORDS if w in lower})


OPERATOR_WORDS = {
    "must": ["must", "required", "mandatory", "need to have", "has to have", "non-negotiable"],
    "should": ["should", "preferred", "nice to have", "ideally", "bonus", "plus if"],
    "exclude": ["exclude", "excluding", "without", "must not", "no experience with", "not include"],
}


def extract_operators(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for op, words in OPERATOR_WORDS.items():
        if any(w in lower for w in words):
            found.append(op)
    return found


OUTPUT_SHAPE_WORDS = {
    "list": ["list", "bullet", "bulleted"],
    "score": ["score", "rate 1", "rating", "rank ", "ranked", "out of 10", "out of 5"],
    "summary": ["summarize", "summarise", "summary", "tl;dr", "overview"],
    "yes-no": ["yes or no", "yes/no", "true or false", "true/false"],
    "table": ["table", "columns", "spreadsheet", "grid"],
    "json": ["json", "structured output", "key-value"],
}


def extract_output_shape(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for shape, words in OUTPUT_SHAPE_WORDS.items():
        if any(w in lower for w in words):
            found.append(shape)
    return found


EVIDENCE_WORDS = [
    "source", "cite", "citation", "link", "verify", "verification", "evidence", "url",
    "proof", "confirm", "double-check", "double check", "fact-check", "fact check",
]


def extract_evidence_demand(text: str) -> bool:
    lower = text.lower()
    return any(w in lower for w in EVIDENCE_WORDS)


SLOT_EXTRACTORS = {
    "job_title": extract_job_titles,
    "skill": extract_skills,
    "company": extract_companies,
    "years_experience": extract_years_experience,
    "location": extract_locations,
    "education": extract_education,
    "seniority": extract_seniority,
}


def extract_all_slots(text: str) -> dict[str, list[str]]:
    return {slot: fn(text) for slot, fn in SLOT_EXTRACTORS.items()}


# ---------------------------------------------------------------------------
# (b) Frequency tables
# ---------------------------------------------------------------------------


def build_frequency_tables(records: list[dict]) -> dict:
    kinds = sorted({r.get("kind", "?") for r in records})
    author_roles = sorted({r.get("author_role", "?") for r in records})
    source_types = sorted({r.get("source_type", "?") for r in records})

    # slot x kind: presence rate (records in that kind with >=1 match) and
    # top-5 matched surface values overall.
    slot_kind_presence = {slot: {k: 0 for k in kinds} for slot in SLOT_EXTRACTORS}
    slot_value_counts = {slot: Counter() for slot in SLOT_EXTRACTORS}
    kind_totals = Counter(r.get("kind", "?") for r in records)

    operator_kind = {op: {k: 0 for k in kinds} for op in OPERATOR_WORDS}
    shape_kind = {shape: {k: 0 for k in kinds} for shape in OUTPUT_SHAPE_WORDS}

    evidence_by_kind = {k: [0, 0] for k in kinds}  # [with evidence demand, total]
    evidence_by_role = {r: [0, 0] for r in author_roles}
    verbatim_by_source_type = {s: [0, 0] for s in source_types}  # [verbatim true, total]

    for rec in records:
        text = rec.get("text", "")
        if not isinstance(text, str):
            text = ""
        kind = rec.get("kind", "?")
        role = rec.get("author_role", "?")
        source_type = rec.get("source_type", "?")

        slots = extract_all_slots(text)
        for slot, values in slots.items():
            if values:
                slot_kind_presence[slot][kind] += 1
                slot_value_counts[slot].update(v.lower() for v in values)

        for op in extract_operators(text):
            operator_kind[op][kind] += 1

        for shape in extract_output_shape(text):
            shape_kind[shape][kind] += 1

        has_evidence = extract_evidence_demand(text)
        evidence_by_kind[kind][1] += 1
        evidence_by_role[role][1] += 1
        if has_evidence:
            evidence_by_kind[kind][0] += 1
            evidence_by_role[role][0] += 1

        verbatim_by_source_type[source_type][1] += 1
        if rec.get("verbatim") is True:
            verbatim_by_source_type[source_type][0] += 1

    return {
        "kinds": kinds,
        "author_roles": author_roles,
        "source_types": source_types,
        "kind_totals": dict(kind_totals),
        "slot_kind_presence": slot_kind_presence,
        "slot_value_counts": slot_value_counts,
        "operator_kind": operator_kind,
        "shape_kind": shape_kind,
        "evidence_by_kind": evidence_by_kind,
        "evidence_by_role": evidence_by_role,
        "verbatim_by_source_type": verbatim_by_source_type,
    }


def render_frequency_tables_md(freq: dict) -> str:
    kinds = freq["kinds"]
    lines = ["# Frequency tables", ""]
    lines.append(
        "Generated by `scripts/mine.py`. All counts come from deterministic "
        "regex/keyword-list slot extraction (spec §6) — nothing here is model-scored."
    )
    lines.append("")

    lines.append("## Slot presence x kind")
    lines.append("")
    lines.append("Share of records in each `kind` where the slot extractor found >=1 match.")
    lines.append("")
    header = "| slot | " + " | ".join(kinds) + " |"
    sep = "|---|" + "---|" * len(kinds)
    lines.append(header)
    lines.append(sep)
    for slot, per_kind in freq["slot_kind_presence"].items():
        row = [slot]
        for k in kinds:
            total = freq["kind_totals"].get(k, 0)
            n = per_kind.get(k, 0)
            pct = f"{n}/{total} ({100 * n / total:.0f}%)" if total else "0/0"
            row.append(pct)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Top matched values per slot (top 8)")
    lines.append("")
    for slot, counter in freq["slot_value_counts"].items():
        top = counter.most_common(8)
        lines.append(f"- **{slot}**: " + (
            ", ".join(f"{v} ({c})" for v, c in top) if top else "_no matches found_"
        ))
    lines.append("")

    lines.append("## Operator x kind (must / should / exclude)")
    lines.append("")
    header = "| operator | " + " | ".join(kinds) + " |"
    lines.append(header)
    lines.append(sep)
    for op, per_kind in freq["operator_kind"].items():
        row = [op] + [str(per_kind.get(k, 0)) for k in kinds]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Requested output shape x kind")
    lines.append("")
    lines.append(header.replace("operator", "shape"))
    lines.append(sep)
    for shape, per_kind in freq["shape_kind"].items():
        row = [shape] + [str(per_kind.get(k, 0)) for k in kinds]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Evidence-demand rate by kind")
    lines.append("")
    lines.append("| kind | with evidence demand | total | rate |")
    lines.append("|---|---|---|---|")
    for k, (n, total) in freq["evidence_by_kind"].items():
        rate = f"{100 * n / total:.0f}%" if total else "n/a"
        lines.append(f"| {k} | {n} | {total} | {rate} |")
    lines.append("")

    lines.append("## Evidence-demand rate by author_role")
    lines.append("")
    lines.append("| author_role | with evidence demand | total | rate |")
    lines.append("|---|---|---|---|")
    for role, (n, total) in freq["evidence_by_role"].items():
        rate = f"{100 * n / total:.0f}%" if total else "n/a"
        lines.append(f"| {role} | {n} | {total} | {rate} |")
    lines.append("")

    lines.append("## Verbatim rate by source_type")
    lines.append("")
    lines.append("| source_type | verbatim=true | total | rate |")
    lines.append("|---|---|---|---|")
    for st, (n, total) in freq["verbatim_by_source_type"].items():
        rate = f"{100 * n / total:.0f}%" if total else "n/a"
        lines.append(f"| {st} | {n} | {total} | {rate} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# (c) Clustering
# ---------------------------------------------------------------------------


def get_embeddings(texts: list[str], offline: bool = True) -> tuple["np.ndarray", str]:
    """Try sentence-transformers all-MiniLM-L6-v2; fall back to TF-IDF if
    `offline` is set (the default — no network call at all), the package is
    missing, or the model can't be downloaded."""
    if offline:
        print(
            "[mine.py] --offline set (default); skipping sentence-transformers, using TF-IDF",
            file=sys.stderr,
        )
    else:
        try:
            from sentence_transformers import SentenceTransformer

            kwargs = {"revision": MINILM_REVISION} if MINILM_REVISION else {}
            model = SentenceTransformer("all-MiniLM-L6-v2", **kwargs)
            embeddings = model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
            return embeddings, "minilm"
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a fallback path
            print(f"[mine.py] sentence-transformers unavailable ({exc}); falling back to TF-IDF", file=sys.stderr)

    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), "tfidf"


AGGLOMERATIVE_DISTANCE_THRESHOLD = 0.7  # cosine distance; fixed, not tuned per run


def run_clustering(embeddings, n_records: int) -> tuple[list[int], str]:
    """Try HDBSCAN, fall back to agglomerative clustering with a fixed
    cosine-distance threshold. Returns (labels, backend_name). Label -1 means
    "noise" (HDBSCAN) — agglomerative never emits noise."""
    if n_records < 4:
        # Too few records for either algorithm to do anything meaningful.
        return [0] * n_records, "trivial"

    try:
        import hdbscan

        min_cluster_size = max(3, n_records // 100)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
        )
        labels = clusterer.fit_predict(embeddings)
        return list(int(x) for x in labels), "hdbscan"
    except Exception as exc:  # noqa: BLE001
        print(f"[mine.py] HDBSCAN unavailable ({exc}); falling back to agglomerative clustering", file=sys.stderr)
        from sklearn.cluster import AgglomerativeClustering

        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=AGGLOMERATIVE_DISTANCE_THRESHOLD,
            metric="cosine",
            linkage="average",
        )
        labels = clusterer.fit_predict(embeddings)
        return list(int(x) for x in labels), "agglomerative"


def label_clusters(records: list[dict], labels: list[int]) -> list[dict]:
    """For each non-noise cluster, compute top-5 TF-IDF terms (fit on that
    cluster's texts against the whole corpus's IDF) and 3 example record ids."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [r.get("text", "") or "" for r in records]
    corpus_vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    corpus_vectorizer.fit(texts)
    vocab = corpus_vectorizer.get_feature_names_out()

    clusters_by_label = defaultdict(list)
    for rec, label in zip(records, labels):
        clusters_by_label[label].append(rec)

    results = []
    for label, members in sorted(clusters_by_label.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        member_texts = [m.get("text", "") or "" for m in members]
        try:
            tfidf = corpus_vectorizer.transform(member_texts)
            mean_scores = tfidf.mean(axis=0)
            mean_scores = np.asarray(mean_scores).ravel() if np is not None else list(mean_scores)
            top_idx = sorted(range(len(mean_scores)), key=lambda i: -mean_scores[i])[:5]
            top_terms = [vocab[i] for i in top_idx if mean_scores[i] > 0]
        except Exception:
            top_terms = []

        example_ids = [m["id"] for m in members[:3]]
        results.append(
            {
                "cluster": "noise" if label == -1 else label,
                "size": len(members),
                "top_terms": top_terms,
                "example_ids": example_ids,
                "kinds": dict(Counter(m.get("kind", "?") for m in members)),
            }
        )
    return results


def render_clusters_md(cluster_summaries: list[dict], embedding_backend: str, clustering_backend: str) -> str:
    lines = ["# Clusters", ""]
    lines.append(
        f"Embedding backend: **{embedding_backend}**. Clustering backend: **{clustering_backend}**. "
        "Clusters describe shapes seen in the text, not a quality score."
    )
    lines.append("")
    for c in cluster_summaries:
        title = f"Cluster {c['cluster']}" if c["cluster"] != "noise" else "Noise (unclustered)"
        lines.append(f"## {title} — {c['size']} record(s)")
        lines.append("")
        lines.append(f"- top terms: {', '.join(c['top_terms']) if c['top_terms'] else '_none_'}")
        lines.append(f"- kinds: {c['kinds']}")
        lines.append(f"- example ids: {', '.join(str(i) for i in c['example_ids'])}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="in_path", required=True, help="path to prompts.jsonl")
    parser.add_argument("--out", dest="out_dir", required=True, help="directory to write reports/ into")
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "skip sentence-transformers and go straight to the TF-IDF fallback "
            "(default: True, no network call). Pass --no-offline to let mine.py "
            "try one outbound call to huggingface.co for the MiniLM embedding model."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    records = load_records(args.in_path)
    if not records:
        print("[mine.py] no records found, nothing to mine", file=sys.stderr)
        return 1

    texts = [r.get("text", "") or "" for r in records]

    # (a) + (b)
    freq = build_frequency_tables(records)
    freq_md = render_frequency_tables_md(freq)
    with open(os.path.join(args.out_dir, "frequency_tables.md"), "w", encoding="utf-8") as fh:
        fh.write(freq_md)

    # which extractors produced zero matches across the whole corpus
    empty_slots = [
        slot for slot, counter in freq["slot_value_counts"].items() if not counter
    ]

    # (c)
    embeddings, embedding_backend = get_embeddings(texts, offline=args.offline)
    labels, clustering_backend = run_clustering(embeddings, len(records))
    cluster_summaries = label_clusters(records, labels)
    clusters_md = render_clusters_md(cluster_summaries, embedding_backend, clustering_backend)
    with open(os.path.join(args.out_dir, "clusters.md"), "w", encoding="utf-8") as fh:
        fh.write(clusters_md)

    n_clusters = len({c["cluster"] for c in cluster_summaries if c["cluster"] != "noise"})
    n_noise = sum(c["size"] for c in cluster_summaries if c["cluster"] == "noise")

    # (d) summary.json
    summary = {
        "n_records": len(records),
        "kind_totals": freq["kind_totals"],
        "embedding_backend": embedding_backend,
        "clustering_backend": clustering_backend,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "empty_slot_extractors": empty_slots,
        "random_seed": RANDOM_SEED,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 70)
    print("mine.py — done")
    print("=" * 70)
    print(f"records mined:        {len(records)}")
    print(f"embedding backend:    {embedding_backend}")
    print(f"clustering backend:   {clustering_backend}")
    print(f"clusters found:       {n_clusters} (+ {n_noise} noise records)")
    print(f"empty slot extractors: {empty_slots if empty_slots else 'none'}")
    print(f"reports written to:   {args.out_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
