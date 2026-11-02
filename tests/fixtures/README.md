# Test fixtures

All content here is synthetic. "Example Corp" is a made-up company.

`corpus/` is a tiny two-corpus tree for ingestion tests:

- `corpus_a/handbook.md` — Markdown with a heading; spans several chunks.
- `corpus_a/policy.pdf` — one-page PDF (Helvetica text, hand-built so pypdf can extract it).
- `corpus_b/notes.txt` — plain text with one paragraph longer than the chunk size.
