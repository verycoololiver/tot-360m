# Calibration snapshot: provenance clarification

This note applies only to `production_corpus.jsonl` with SHA256
`81d680336d36dc3e168ac5e05f0ee757ea37fa732a7a952018b76910170f7223`
and the packed manifest SHA256
`be9bf24c7ce842dec675c516586e1d1dfb2ef3364eca4efbccd717cdb978c41c`.
It does not change the frozen corpus, tokenizer, shards or training identities.

The corpus builder requests Hugging Face's **unversioned rows API**. Its
hardcoded per-record `revision` fields are observed upstream reference values,
not proof that those requests fetched that commit. Do not describe this collection
as a commit-pinned upstream download. Reproduction currently depends on retaining
the collected, checksummed local snapshot. A future corpus must bind source
revision, configuration, split, row/source identity and retrieval method explicitly.

The independent corpus audit records 4,959 unique documents: 3,059 FineWeb-Edu,
1,254 code documents, 641 Korean Wikipedia articles and 5 authored Iris examples.
It verifies exact content hashes, exact deduplication, held-out hash isolation,
tokenizer re-encoding and packed shard identities. It does **not** independently
verify every source's license notice, author attribution, repository revision or
near-duplicate exclusion.

Stored labels are `ODC-By` for FineWeb-Edu, accepted permissive repository labels
for code, and `CC-BY-SA-3.0 / GFDL` for the Wikipedia source. Code records retain
repository and path, and web/article records retain available source URLs. These
labels and source identifiers require their corresponding notices and attribution
to be preserved; they are not a basis for calling the corpus wholly proprietary.
The five Iris templates are authored examples, not authentic user transcripts.

This is a small calibration snapshot: 7,206,912 packed tokens, with finite
rank-disjoint usable capacity. It is not the proposed billion-token pretraining
corpus and must not be silently repeated or filled with random tokens to meet
the 100M smoke-test ceiling. No model quality or trained personality is established
by the corpus audit. Consult `../DATA-PIPELINE-REVIEW.md` and
`../qa/corpus-audit.json` for the evidence and remaining provenance work.
