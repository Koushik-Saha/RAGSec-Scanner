# Verification record — v0.1.0

Commands that were actually run, with their actual output, so the state of
the release can be checked later. Local paths are shown as `<scratch>`.

- **Date:** 2026-11-03
- **Code under test:** commit `4baf40a` plus the v0.1.0 docs/test changes
  committed with this file
- **Python:** 3.12.9 · **Platform:** macOS (Apple silicon)

> **Not yet verified:** the steps that call Claude (`ask`, `scan`, `report`
> with a real answer, and the live tests). No `ANTHROPIC_API_KEY` was
> available on the verification machine. Those commands were run and their
> real output (a clean configuration error) is recorded below; the section
> "Pending: run with an API key" lists what to run and record once a key is
> set.

## 1. Secret sweep

Run over the working tree (excluding `.venv/`) and over every commit on `main`.

| Check | Result |
|---|---|
| `sk-ant` anywhere | Only documentation placeholders: `sk-ant-...` (README), `sk-ant-your-key-here` (DEPLOY.md) |
| `ANTHROPIC_API_KEY=` with a value | Only placeholders: `.env.example` (`your-anthropic-api-key-here`), DEPLOY.md |
| Key-shaped strings `sk-[A-Za-z0-9_-]{20,}` | None (tests use `sk-test-not-real`) |
| Absolute local paths (`/Users/`, `/home/`, `/private/tmp`, …) | None |
| Same patterns in added lines across all commits | 0 matches each |
| `.env` git-ignored | Yes (`.gitignore` line 2) |
| `.env` ever committed (all refs) | Never: 0 commits touch it |
| Docker build context | `.env`, `.venv/`, `chroma/`, `reports/`, `.git/` excluded by `.dockerignore`; image checked to contain no `.env` |

## 2. Tests and lint

Default suite on the development machine, no API key present:

```
$ pytest -q
271 passed, 4 deselected
$ pytest -q -m live -rs
SKIPPED [1] tests/test_live.py:51: no ANTHROPIC_API_KEY in environment or .env
SKIPPED [1] tests/test_live.py:64: no ANTHROPIC_API_KEY in environment or .env
SKIPPED [1] tests/test_live.py:76: no ANTHROPIC_API_KEY in environment or .env
3 skipped
$ pytest -q -m slow
1 passed
$ ruff check .
All checks passed!
```

Clean-clone check: the files to be committed were copied into an empty
folder, a fresh venv was created, and `pip install -e ".[dev]"` was run. Tests
ran with no `.env`, no key, an empty Hugging Face cache, and
`HF_HUB_OFFLINE=1`:

```
$ pytest -q
271 passed, 4 deselected
$ pytest -q -m live -rs
3 skipped        (no ANTHROPIC_API_KEY in environment or .env)
$ pytest -q -m slow -rs
SKIPPED [1] tests/test_retrieve.py:169: embedding model unavailable: OSError
1 skipped
$ ruff check . && ruff format --check .
All checks passed!
54 files already formatted
```

Tests that need the outside world, and how they behave without it:

| Marker | Tests | Needs | Without it |
|---|---|---|---|
| *(none)* | 271 | Nothing. Network access and the real embedding model are blocked by a fixture and fail the test if touched. | Runs |
| `slow` | 1 (`test_real_embedder_ranking_and_filter`) | The `all-MiniLM-L6-v2` model (downloaded on first use) | Skips: "embedding model unavailable" |
| `live` | 3 (`tests/test_live.py`) | `ANTHROPIC_API_KEY` and `RAGSEC_MODEL`, plus the embedding model; costs money | Skips: "no ANTHROPIC_API_KEY" |

`slow` and `live` are deselected by default; run them with `pytest -m slow`
or `pytest -m live`.

## 3. End-to-end run (local)

`RAGSEC_CHROMA_DIR` and `RAGSEC_REPORTS_DIR` pointed at a scratch folder; no
`.env`.

```
$ ragsec ingest data
Indexed 13 chunks into <scratch>/chroma
  handbook: 8
  handbook_poisoned: 3
  restricted: 2
(exit 0)

$ ragsec ask "Which port does the VPN use?" --corpus handbook
Configuration error: RAGSEC_MODEL is not set (see .env.example)
(exit 2)

$ ragsec scan
Configuration error: RAGSEC_MODEL is not set (see .env.example)
(exit 2)

$ ragsec report
Configuration error: RAGSEC_MODEL is not set (see .env.example)
(exit 2; no report files written)

$ uvicorn ragsec.api:app --port 8765
$ curl localhost:8765/health
{"status":"ok"}  [HTTP 200]
$ curl localhost:8765/
HTTP 200 (demo page)
```

Retrieval over the same index with the real embedding model (no Claude call):

```
chunks indexed: 13 corpora: ['handbook', 'handbook_poisoned', 'restricted']
'Which port does the VPN use?' [corpus=handbook]
    -> ['handbook:vpn-setup.md:0', 'handbook:vpn-setup.md:1', 'handbook:onboarding.md:0']
'What is the HR records system access token?' [corpus=handbook]
    -> ['handbook:support-contacts.md:1', 'handbook:password-policy.md:1', 'handbook:support-contacts.md:0']
```

The VPN question retrieves the VPN setup document first; the HR-token question
scoped to the handbook returns no restricted chunks.

## 4. Docker (recorded during the packaging step, same day)

```
$ docker build -t ragsec-scanner:dev .
Successfully built   (image 2.54 GB; PyTorch CPU build is 688 MB of it)
$ docker run -p 8000:8000 -e ANTHROPIC_API_KEY=<deliberately invalid> -e RAGSEC_MODEL=claude-opus-5-5 ragsec-scanner:dev
GET  /health -> 200 {"status":"ok"}        healthy after ~11 s
GET  /       -> 200 text/html (demo page)
POST /ask    -> 503 {"error":"not_configured","detail":"The model API rejected the server's credentials."}
container user: ragsec (non-root); no .env in image; embedding model present in /opt/hf-cache
memory: ~830 MiB after startup indexing, ~880 MiB after requests
```

The `/ask` result shows the key passed through the environment reached the
Anthropic API (which rejected the invalid key); a real answer needs a valid key.

## Pending: run with an API key

Once `.env` holds a real `ANTHROPIC_API_KEY` and `RAGSEC_MODEL`, run and
record here:

```
ragsec ask "Which port does the VPN use?" --corpus handbook
ragsec ask "Who won the 1998 football World Cup?"
ragsec scan
ragsec report
pytest -m live
```

and copy the `ragsec scan` summary into the README's "Sample result" section.
