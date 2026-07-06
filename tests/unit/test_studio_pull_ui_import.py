import importlib


def test_pull_ui_exposes_run_pull():
    module = importlib.import_module("rag_lab.studio.pull_ui")
    assert callable(getattr(module, "run_pull", None))
