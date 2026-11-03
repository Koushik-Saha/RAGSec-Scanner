"""Security probes. Each probe is a separate module exposing run(ctx) -> ProbeResult.

Use run_probe(probe_id, ctx) to run one probe by ID.
"""

from types import ModuleType

from ragsec.probes import (
    benign_control,
    citation_integrity,
    context_bleed,
    cross_document_leakage,
    indirect_injection_body,
    injection_via_title,
    planted_secret_extraction,
    poisoning_answer_flip,
    system_prompt_extraction,
    ungrounded_answer_trap,
)
from ragsec.probes.base import Evidence, ProbeContext, ProbeResult, ProbeSetupError, Step

# In probe-number order.
_MODULES: list[ModuleType] = [
    indirect_injection_body,
    injection_via_title,
    planted_secret_extraction,
    cross_document_leakage,
    poisoning_answer_flip,
    system_prompt_extraction,
    ungrounded_answer_trap,
    citation_integrity,
    context_bleed,
    benign_control,
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
    "Step",
    "probe_ids",
    "run_probe",
]
