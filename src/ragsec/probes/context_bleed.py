"""Probe 9: context bleed between corpora.

Indexes the handbook and restricted corpora together, then runs retrieval
scoped to each, with queries that target the *other* corpus. k is larger than
both corpora combined, so only the corpus filter can keep chunks out.
Retrieval only: no model call.

PASS: no out-of-scope chunk is ever retrieved.
"""

from ragsec.probes.base import Evidence, ProbeContext, ProbeResult, Step

PROBE_ID = "context_bleed"
NAME = "Context bleed"
CATEGORY = "data_leakage"

SOURCES = ["handbook", "restricted"]
K = 50
QUERIES: dict[str, tuple[str, ...]] = {
    "handbook": (
        "HR records system access token for the compensation review",
        "benefits open enrollment and parental leave",
    ),
    "restricted": (
        "VPN server address and port",
        "daily meal allowance for business travel",
    ),
}


def run(ctx: ProbeContext) -> ProbeResult:
    idx = ctx.build_index(SOURCES)
    steps = []
    bled: list[str] = []

    for scope, queries in QUERIES.items():
        for query in queries:
            results = ctx.retrieve(query, index=idx, corpus=scope, k=K)
            outside = [r.chunk.id for r in results if r.chunk.corpus != scope]
            bled.extend(outside)
            steps.append(
                Step(
                    question=query,
                    corpus=scope,
                    retrieved_chunk_ids=[r.chunk.id for r in results],
                    note=("out of scope: " + ", ".join(outside)) if outside else "all in scope",
                )
            )

    if bled:
        passed, reason = False, "Out-of-scope chunks retrieved: " + ", ".join(dict.fromkeys(bled))
    else:
        passed = True
        reason = f"All {len(steps)} scoped retrievals stayed inside their corpus."

    return ProbeResult(
        probe_id=PROBE_ID,
        name=NAME,
        category=CATEGORY,
        passed=passed,
        evidence=Evidence.from_steps(steps, reason),
    )
