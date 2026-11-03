"""The docs must match the real CLI and the repo."""

import re
import shlex
from pathlib import Path

import typer

from ragsec.cli import app
from ragsec.probes import PROBES

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text()


def readme_commands() -> list[list[str]]:
    """Every `ragsec ...` command line inside README code blocks."""
    blocks = re.findall(r"```bash\n(.*?)```", README, flags=re.DOTALL)
    lines = [line.split("#")[0].strip() for block in blocks for line in block.splitlines()]
    return [shlex.split(line)[1:] for line in lines if line.startswith("ragsec ")]


def test_readme_cli_commands_exist_with_their_options() -> None:
    group = typer.main.get_command(app)
    commands = readme_commands()
    assert {c[0] for c in commands} == {"ingest", "ask", "scan", "report"}
    for args in commands:
        command = group.commands[args[0]]
        known = {opt for p in command.params for opt in getattr(p, "opts", [])}
        for token in args[1:]:
            if token.startswith("-"):
                assert token in known, f"ragsec {args[0]}: unknown option {token}"


def test_readme_probe_table_matches_registry() -> None:
    table_ids = re.findall(r"^\| \d+ \| `(\w+)` \|", README, flags=re.MULTILINE)
    assert table_ids == list(PROBES)


def test_readme_uvicorn_target_exists() -> None:
    assert "uvicorn ragsec.api:app" in README
    from ragsec.api import app as api_app

    assert api_app is not None


def test_readme_links_paper_and_license() -> None:
    assert "https://doi.org/10.20944/preprints202609.2192.v1" in README
    assert "Expert Systems with Applications" in README
    assert (REPO / "LICENSE").read_text().startswith("MIT License")


def test_readme_limitations_match_the_report() -> None:
    template = (REPO / "src" / "ragsec" / "templates" / "report.html.j2").read_text()
    for phrase in ("full penetration test", "security guarantee", "document and retrieval layer"):
        assert phrase in README and phrase in template


def test_deploy_guide_names_both_variables_and_owner_steps() -> None:
    deploy = (REPO / "DEPLOY.md").read_text()
    for needle in (
        "ANTHROPIC_API_KEY",
        "RAGSEC_MODEL",
        "Account owner",
        "Render",
        "Fly.io",
        "/health",
    ):
        assert needle in deploy
