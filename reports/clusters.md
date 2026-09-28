# Clusters

Embedding backend: **tfidf**. Clustering backend: **hdbscan**. Clusters describe shapes seen in the text, not a quality score.

## Noise (unclustered) — 185 record(s)

- top terms: experience, candidate, candidates, skills, role
- kinds: {'lookup_prompt': 49, 'ranking_prompt': 32, 'screening_rubric': 42, 'sourcing_query': 62}
- example ids: pb-0001, pb-0003, pb-0004

## Cluster 0 — 39 record(s)

- top terms: job, resume, title, candidate, description
- kinds: {'lookup_prompt': 8, 'ranking_prompt': 26, 'screening_rubric': 1, 'sourcing_query': 4}
- example ids: pb-0002, pb-0009, pb-0010

## Cluster 1 — 12 record(s)

- top terms: search, boolean, string, linkedin, strings
- kinds: {'sourcing_query': 12}
- example ids: pb-0159, pb-0164, pb-0178

## Cluster 2 — 3 record(s)

- top terms: developer, stack, android, js, node
- kinds: {'sourcing_query': 3}
- example ids: pb-0181, pb-0193, pb-0214
