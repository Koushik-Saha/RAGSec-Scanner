# Changelog

## 0.1.0 — 2026-11-03

First release. A Python tool that security-tests the document and retrieval
layer of a retrieval-augmented generation (RAG) system; the working
implementation of the RAGSec paper (DOI 10.20944/preprints202609.2192.v1).

- **RAG pipeline:** ingestion of `.md`, `.txt`, and `.pdf` files into
  ~800-character overlapping chunks with stable IDs; `all-MiniLM-L6-v2`
  embeddings; a persistent Chroma index with corpus-filtered retrieval; Claude
  answers that cite chunk IDs, with citation validation and a fixed refusal
  phrase.
- **10 probes**, each with a deterministic verdict (no LLM judge): indirect
  injection in a document body, injection via title, planted-secret
  extraction, cross-document leakage, poisoning / answer flipping,
  system-prompt extraction via retrieval, ungrounded-answer trap, citation
  integrity, context bleed between corpora, and a benign over-refusal control.
- **Scan and reports:** score = probes passed / 10; `report.json` with full
  evidence and a self-contained, escaped `report.html`.
- **Interfaces:** `ragsec` CLI (`ingest`, `ask`, `scan`, `report`); FastAPI
  service (`/health`, `/ask`, `/scan`, `/report`, `/report/html`) with a
  static demo page.
- **Packaging:** Dockerfile with the embedding model baked in; deployment
  guide for Render and Fly.io.
- **Synthetic demo corpus** for a fictional company, with planted attacks and
  a canary token.
