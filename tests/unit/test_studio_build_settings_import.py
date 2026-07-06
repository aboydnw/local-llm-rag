import importlib


def test_build_settings_exposes_render():
    module = importlib.import_module("rag_lab.studio.build_settings")
    assert callable(getattr(module, "render", None))
