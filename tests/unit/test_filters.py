from rag_lab.filters import is_api_stub


def test_bare_directive_is_stub():
    assert is_api_stub("::: titiler.core.errors") is True


def test_directive_with_indented_options_is_stub():
    assert is_api_stub("::: titiler.core.dependencies\n    options:\n      show_source: true") is True


def test_directive_with_surrounding_blank_lines_is_stub():
    assert is_api_stub("\n::: titiler.core.factory\n\n") is True


def test_prose_is_not_stub():
    assert is_api_stub("TiTiler is a dynamic tile server built on FastAPI.") is False


def test_prose_after_directive_is_not_stub():
    assert is_api_stub("::: titiler.core.errors\nThis module defines errors.") is False


def test_admonition_with_content_is_not_stub():
    assert is_api_stub(":::note\nThis is an important note about tiles.") is False


def test_empty_text_is_not_stub():
    assert is_api_stub("") is False
    assert is_api_stub("   \n  ") is False
