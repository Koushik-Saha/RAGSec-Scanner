"""JSON and HTML security report generation.

report.json is the full ScanResult. report.html is one self-contained file
(inline CSS, no scripts, no external assets). All values rendered into the
HTML, model output included, are escaped by Jinja2 autoescaping.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, StrictUndefined
from pydantic import BaseModel

from ragsec import __version__
from ragsec.probes import ProbeResult
from ragsec.scan import ScanResult

TOOL_NAME = "ragsec-scanner"
ANSWER_PREVIEW_CHARS = 800
TEMPLATE_NAME = "report.html.j2"

_env = Environment(
    loader=PackageLoader("ragsec", "templates"),
    autoescape=True,  # always on, whatever the template's file extension
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


class ReportPaths(BaseModel):
    json_path: Path
    html_path: Path


def write_reports(scan_result: ScanResult, out_dir: str | Path) -> ReportPaths:
    """Write report.json and report.html into `out_dir` (created if needed)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "report.json"
    json_path.write_text(scan_result.model_dump_json(indent=2) + "\n", encoding="utf-8")

    html_path = out / "report.html"
    html_path.write_text(render_html(scan_result), encoding="utf-8")
    return ReportPaths(json_path=json_path, html_path=html_path)


def render_html(scan_result: ScanResult) -> str:
    template = _env.get_template(TEMPLATE_NAME)
    return template.render(
        tool_name=TOOL_NAME,
        version=__version__,
        run_at=_human_time(scan_result.timestamp),
        scan=scan_result,
        probes=[_probe_view(n, r) for n, r in enumerate(scan_result.results, start=1)],
        probe_count=scan_result.total,
        errors=len(scan_result.errors),
    )


def report_dir_name(timestamp: str) -> str:
    """Filesystem-safe folder name for a scan timestamp, e.g. 20261103T191203Z."""
    moment = datetime.fromisoformat(timestamp)
    return moment.strftime("%Y%m%dT%H%M%SZ")


def truncate(text: str, limit: int = ANSWER_PREVIEW_CHARS) -> tuple[str, bool]:
    """Cut `text` to about `limit` characters, at a word boundary when possible."""
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.8:
        cut = cut[:space]
    return cut.rstrip() + " …", True


def _probe_view(number: int, result: ProbeResult) -> dict[str, Any]:
    status = "error" if result.error else ("pass" if result.passed else "fail")
    answer, truncated = truncate(result.evidence.answer_text)
    steps = []
    for step in result.evidence.steps:
        step_answer, step_truncated = truncate(step.answer_text)
        steps.append({"step": step, "answer": step_answer, "truncated": step_truncated})
    return {
        "number": number,
        "result": result,
        "status": status,
        "answer": answer,
        "truncated": truncated,
        "steps": steps,
    }


def _human_time(timestamp: str) -> str:
    moment = datetime.fromisoformat(timestamp)
    return moment.strftime("%Y-%m-%d %H:%M:%S %Z").strip() or timestamp
