import importlib

import pytest
from typer.testing import CliRunner

from ragsec.cli import app

MODULES = [
    "ragsec",
    "ragsec.config",
    "ragsec.ingest",
    "ragsec.index",
    "ragsec.retrieve",
    "ragsec.answer",
    "ragsec.probes",
    "ragsec.scan",
    "ragsec.report",
    "ragsec.cli",
    "ragsec.api",
]

COMMANDS = ["ingest", "ask", "scan", "report"]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_cli_help_lists_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in COMMANDS:
        assert command in result.output
