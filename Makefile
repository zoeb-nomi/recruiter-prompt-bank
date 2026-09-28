.PHONY: validate mine all test dedupe

DATA ?= data/prompts.jsonl
REPORTS ?= reports/
FIXTURE ?= tests/fixture_sample.jsonl

# Curation step (spec §5): dedupe raw collector output. Override
# DEDUPE_IN/DEDUPE_OUT/DEDUPE_REJECTS to point at a real collector run;
# defaults here operate on the already-curated data files.
DEDUPE_IN ?= data/prompts.jsonl
DEDUPE_OUT ?= data/prompts.jsonl
DEDUPE_REJECTS ?= data/rejects.jsonl

validate:
	python3 scripts/validate.py --in $(DATA)

mine:
	python3 scripts/mine.py --in $(DATA) --out $(REPORTS)

all: validate mine

# Runs validate.py against the small committed fixture, not the full dataset.
test:
	python3 scripts/validate.py --in $(FIXTURE)

# Curation step, not part of `all`: dedupes raw collector output into
# data/prompts.jsonl + data/rejects.jsonl. Run this before `make all` when
# curating a new collection pass.
dedupe:
	python3 scripts/dedupe.py --in $(DEDUPE_IN) --out $(DEDUPE_OUT) --rejects $(DEDUPE_REJECTS)
