from pathlib import Path

from ragsec.config import EMBEDDING_MODEL, load_settings


def test_defaults_when_env_empty() -> None:
    s = load_settings({})
    assert s.anthropic_api_key is None
    assert s.model is None
    assert s.data_dir == Path("./data")
    assert s.chroma_dir == Path("./chroma")
    assert s.reports_dir == Path("./reports")
    assert s.embedding_model == EMBEDDING_MODEL


def test_reads_values_from_env() -> None:
    s = load_settings(
        {
            "ANTHROPIC_API_KEY": "sk-test-not-real",
            "RAGSEC_MODEL": "some-model",
            "RAGSEC_DATA_DIR": "/tmp/d",
            "RAGSEC_CHROMA_DIR": "/tmp/c",
            "RAGSEC_REPORTS_DIR": "/tmp/r",
        }
    )
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == "sk-test-not-real"
    assert s.model == "some-model"
    assert s.data_dir == Path("/tmp/d")
    assert s.chroma_dir == Path("/tmp/c")
    assert s.reports_dir == Path("/tmp/r")


def test_api_key_not_exposed_in_repr() -> None:
    s = load_settings({"ANTHROPIC_API_KEY": "sk-test-not-real"})
    assert "sk-test-not-real" not in repr(s)
    assert "sk-test-not-real" not in s.model_dump_json()
