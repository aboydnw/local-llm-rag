import importlib


def test_config_panel_module_imports():
    module = importlib.import_module("rag_lab.studio.config_panel")
    assert callable(getattr(module, "render", None))
