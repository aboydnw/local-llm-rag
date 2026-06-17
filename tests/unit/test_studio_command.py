from typer.testing import CliRunner

from rag_lab.cli import app


def test_studio_command_invokes_streamlit(monkeypatch):
    called = {}

    def _fake_run(cmd, *args, **kwargs):
        called["cmd"] = cmd
        return 0

    monkeypatch.setattr("subprocess.call", _fake_run)
    result = CliRunner().invoke(app, ["studio"])
    assert result.exit_code == 0
    assert "streamlit" in called["cmd"]
    assert "run" in called["cmd"]
