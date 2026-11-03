"""Security probes. Each probe is a separate module exposing run(ctx) -> ProbeResult.

Use run_probe(probe_id, ctx) to run one probe by ID.
"""

from types import ModuleType

from ragsec.probes import (
    cross_document_leakage,
    indirect_injection_body,
    injection_via_title,
    planted_secret_extraction,
    poisoning_answer_flip,
)
from ragsec.probes.base import Evidence, ProbeContext, ProbeResult, ProbeSetupError

# In probe-number order.
_MODULES: list[ModuleType] = [
    indirect_injection_body,
    injection_via_title,
    planted_secret_extraction,
    cross_document_leakage,
    poisoning_answer_flip,
]

PROBES: dict[str, ModuleType] = {m.PROBE_ID: m for m in _MODULES}


def probe_ids() -> list[str]:
    return list(PROBES)


def run_probe(probe_id: str, ctx: ProbeContext) -> ProbeResult:
    """Run a single probe by its ID."""
    try:
        module = PROBES[probe_id]
    except KeyError:
        known = ", ".join(PROBES)
        raise KeyError(f"unknown probe {probe_id!r}; known probes: {known}") from None
    return module.run(ctx)


__all__ = [
    "PROBES",
    "Evidence",
    "ProbeContext",
    "ProbeResult",
    "ProbeSetupError",
    "probe_ids",
    "run_probe",
]
