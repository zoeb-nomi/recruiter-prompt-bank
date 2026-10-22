# recruiter-prompt-bank

Zoeb Nomi — [zoebnomi.com](https://www.zoebnomi.com) · part of the [AI PM eval-quality work](https://www.zoebnomi.com/evals/)

Pass 1 · collected and curated in a single pass on 2026-09-28 · 239 records · 79 sources

## What this is

This repository is a small, public dataset of prompts, search queries, and
screening criteria that hiring-side people and software actually use with
AI tools — recruiters typing a search into an AI sourcing tool, an ATS
describing how it scores a resume against a job description, or someone
pasting a candidate's resume into ChatGPT and asking for a verdict. Every
record is quoted from a public, attributable web page: vendor documentation,
vendor blogs, recruiter-authored posts, community forums, articles, prompt
libraries, and newsletters. Records are also grouped by rough shape
(clusters) and counted by the patterns they contain (frequency tables), all
by deterministic code — no model was asked to judge, rank, or improve any
record.

Use this if you're building an eval or rubric and want real prompts instead
of invented ones — or if you're just curious what recruiters and their
tools actually ask an AI to do.

## Why it exists

Two earlier eval projects (mirror-eval v1 and screener-eval) needed to know
how hiring-side prompts and rubrics are actually worded, and used invented
ones instead because no such dataset existed. This bank replaces the
invented prompts with real ones, so that later work — a prompt set for an
agentic dossier harness, or rubrics for a screening eval — can be built on
what recruiters and ATS vendors actually say, not on a guess. The dataset
and a short pattern read are also published as a standalone artifact: a
plain look at what recruiters ask AI to do.

## What we found, in three lines

Recruiters and their tools almost never ask an AI to show its work (12–16%
of prompts ask for evidence). Screening runs on a numeric fit score, not a
yes/no. And the ATSs large tech companies actually use — Workday/HiredScore,
Rippling — are the least publicly documented of the bunch. Full findings
with record ids below.

## How the data was collected

Six collectors (labelled C1–C6), each targeting a different slice of the
space, searched and read public web pages only — vendor documentation and
blogs, help centres, changelogs, recruiter-authored posts, community forums
(e.g. r/recruiting), prompt libraries, and newsletters. Nothing behind a
login or paywall, no private messages, no scraping of proprietary tool
output.

Every record carries a `verbatim` flag: `true` means the text is quoted
exactly as the source wrote it; `false` means the source only paraphrased
("recruiters can search for things like...") and the record preserves that
paraphrase rather than inventing a verbatim version. Where a target vendor
had no public example at all, that absence is itself recorded in
`data/sources.md` rather than silently skipped.

Every record's `source_url` was fetched once at curation time (2026-09-28)
and the record's `text` checked against the page; the result is stored per
record as `url_ok` and `text_found`. `scripts/validate.py` checks schema and
duplicates only and makes no network calls; re-verification of URLs is a
planned opt-in pass.

| collector | targets |
|---|---|
| C1 sourcing-linkedin-gem | LinkedIn Recruiter AI-Assisted Search docs, LinkedIn Talent Blog, Gem, Dover, Findem |
| C2 sourcing-tools | PeopleGPT (Juicebox), HireEZ, SeekOut, Apollo, Fetcher, Hiretual legacy docs |
| C3 ats-screening | Ashby, Greenhouse, Lever, Workable, Eightfold, HiredScore (Workday), Paradox, Manatal, Teamtailor, Zoho Recruit |
| C4 recruiter-prompts-posts | recruiter-authored ChatGPT/Claude prompts on LinkedIn, blogs, newsletters, SHRM/ERE/Recruiting Daily |
| C5 community-libraries | r/recruiting, r/humanresources, recruiter-focused prompt libraries/packs, GitHub repos |
| C6 candidate-lookup | prompts for researching a named candidate or resume (summarise, verify, red-flag) |

## Schema

One JSON object per line in `data/prompts.jsonl`:

| field | type | notes |
|---|---|---|
| `id` | string | `pb-0001` … |
| `kind` | enum | `sourcing_query` \| `screening_rubric` \| `lookup_prompt` \| `ranking_prompt` |
| `text` | string | the prompt/query/criterion, verbatim where the source gives it verbatim |
| `verbatim` | bool | `true` = quoted as written; `false` = the source paraphrases it |
| `source_url` | string | resolves to the page the text came from; one record → one URL |
| `source_type` | enum | `vendor_docs` \| `vendor_blog` \| `recruiter_post` \| `community` \| `article` \| `prompt_library` \| `newsletter` |
| `author_role` | enum | `vendor` \| `recruiter` \| `hiring_manager` \| `sourcer` \| `unknown` |
| `tool` | string? | product the prompt is tied to, if any (e.g. "LinkedIn Recruiter") |
| `role_family` | enum | `pm` \| `engineering` \| `generic` \| `other` |
| `seniority_hint` | string? | e.g. "senior", "mid", "director", if stated |
| `geo_hint` | string? | e.g. "US", "remote", if stated |
| `date_published` | string? | ISO date if the source shows one |
| `date_observed` | string | ISO date the collector fetched it |
| `collector` | string | agent label that collected it |
| `notes` | string? | anything a curator should know |
| `url_ok` | bool | set once at curation time (2026-09-28): did `source_url` resolve |
| `text_found` | bool | set once at curation time (2026-09-28): did the fetched page contain `text` |
| `has_placeholder` | bool | `true` if `text` contains a bracketed placeholder like `[Job Title]` |

`data/criteria.json` holds structured screening criteria pulled from
`screening_rubric` sources (per-tool: what fields it says it extracts, how
it weighs them, whether it says it uses an LLM) — descriptive only.

`data/rejects.jsonl` holds records dropped in curation, with the reason, so
the drop is auditable.

## Running the checks

```bash
pip install -r requirements.txt

# schema check, duplicate-id check, near-duplicate check
python scripts/validate.py --in data/prompts.jsonl

# slot extraction, frequency tables, clustering -> reports/
python scripts/mine.py --in data/prompts.jsonl --out reports/
```

Or, using the Makefile: `make validate`, `make mine`, `make all` (validate +
mine), `make test` (runs validate.py against `tests/fixture_sample.jsonl`).
Curation itself — deduplicating raw collector output — is `make dedupe`
(`scripts/dedupe.py`); it is not part of `make all`.

`validate.py` does not make network calls — it checks that `source_url` is
a well-formed https URL, not that it resolves; URL resolution and
text-presence matching happen once, upstream, during curation.

`mine.py` defaults to `--offline` (`--offline` is `True` by default), which
skips `sentence-transformers` entirely and goes straight to TF-IDF vectors
with agglomerative clustering — no network call. If `sentence-transformers`
is installed and you pass `--no-offline`, `mine.py` will try one outbound
call to huggingface.co to fetch the `all-MiniLM-L6-v2` embedding model and
cluster with HDBSCAN; if that call or the package is unavailable, it falls
back to TF-IDF + agglomerative clustering automatically. Either way, results
are reproducible (fixed random seeds).

## Findings (first pass, 2026-09-28)

239 records from 79 public URLs after curation (444 collected; 144 dropped as out of scope, 41 duplicates, 20 where the fetched page did not confirm the text — all listed with reasons in `data/rejects.jsonl`). "Out of scope" mostly means interview-question generation, job-description writing, or outreach-message drafting — real hiring-AI uses, but not the sourcing/screening/lookup prompts this bank tracks; full breakdown with reasons in data/rejects.jsonl. Counts come from `scripts/mine.py`; nothing below is model-scored. Record ids are cited so every claim can be checked.

1. **Candidate-lookup prompts rarely ask for evidence.** Only 7 of 57 `lookup_prompt` records (12%) ask the model to cite sources, link, or verify; the rate is similar for ranking prompts (16%) and vendor rubrics (12%). Recruiters and hiring managers mostly ask for a summary or a list and accept what comes back. Read on this: the gap is where candidate-facing work should focus — on what the model's summary of a person says, since the summary itself is rarely checked. (`reports/frequency_tables.md`, evidence-demand tables.)

2. **Ranking is numeric.** 27 of 58 `ranking_prompt` records ask for a score; none ask for a yes/no, and none ask for JSON. Screening happens as a fit score against a job description, which is why a paired A/B résumé design (the delta is the unit) is the right way to test it.

3. **Sourcing queries are slot-shaped; lookups are not.** Sourcing queries name a title (43%), skills (41%), seniority (27%) and location (22%). Lookup prompts almost never do (0–5%): they name a person or paste a résumé. Read on this: being findable is about having those slots as plain text on public surfaces; being read correctly is about what the surfaces say once found.

4. **Vendor rubric transparency is thin, and thinnest where it matters.** 43 `screening_rubric` records in total. The most specific public disclosures come from mid-market tools (Workable Agent's 14 matching-criteria categories; Zoho Recruit's proficiency/duration/recency/rarity signals; Eightfold's embedding match score). Workday/HiredScore, Rippling, Fetcher and Hiretual publish no concrete criteria (`data/sources.md`, "No public examples"). The ATSs large tech companies actually run on are the least documented.

5. **The bank speaks in the vendor's voice.** 173 of 239 records are vendor-authored; 36 are recruiter-authored, 3 hiring-manager. Vendor examples are cleaner but idealised. A second collection pass should target recruiter- and hiring-manager-authored prompts specifically.

Limitations. Slot extraction is regex/keyword based and has known false positives (the skill list matches the word "go"; company matching is capitalised-multiword heuristics). Clustering ran on TF-IDF vectors because the MiniLM model could not be downloaded in the build environment; HDBSCAN found 3 clusters and left 185 records as noise, so `reports/clusters.md` is indicative only. Six Eightfold blog Q&A lines (pb-0118 to pb-0123) are labelled `screening_rubric` but are not a formal published rubric and are excluded from `criteria.json`.


## How to cite

> Nomi, Z. (2026). recruiter-prompt-bank: a public dataset of hiring-side AI prompts,
> search queries, and screening criteria, collected from public sources.
> https://github.com/zoeb-nomi/recruiter-prompt-bank

Cite individual records by their `id` (e.g. `pb-0001`) alongside the
dataset citation.

## Removal requests

Every record carries its source URL. If you are the author of a quoted
prompt, query, or criterion and want it removed, open an issue with the
record's `id` and it will be removed.

## Regenerating and publishing

`reports/` is committed and is regenerated with `make mine` — it is never
edited by hand. A second collection pass appends new records and bumps the
"Pass" line near the top of this README. The repo is published by its
author through the GitHub web UI; nothing here pushes on its own.

## Licence

Data (`data/`) is licensed under [CC BY 4.0](LICENSE-DATA). Code
(everything else) is licensed under [MIT](LICENSE).
