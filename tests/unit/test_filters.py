from rag_lab.filters import is_api_stub, strip_markup


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


def test_spaced_admonition_directive_is_not_stub():
    assert is_api_stub("::: note") is False
    assert is_api_stub(":::note") is False


def test_admonition_with_indented_content_is_not_stub():
    assert is_api_stub("::: note\n    Indented note body.") is False


def test_empty_text_is_not_stub():
    assert is_api_stub("") is False
    assert is_api_stub("   \n  ") is False


def test_strip_markup_removes_html_only_lines():
    text = '<p align="center">\n<img src="logo.png"/>\nTiTiler is a dynamic tile server.'
    assert strip_markup(text) == "TiTiler is a dynamic tile server."


def test_strip_markup_removes_standalone_image():
    assert strip_markup("![logo](logo.png)\nReal prose here.") == "Real prose here."


def test_strip_markup_removes_badge_links():
    text = "[![Build](https://img.shields.io/x.svg)](https://ci.example.com)\nProse."
    assert strip_markup(text) == "Prose."


def test_strip_markup_keeps_plain_prose():
    assert strip_markup("A normal sentence.\nAnother line.") == "A normal sentence.\nAnother line."


def test_strip_markup_all_markup_returns_empty():
    assert strip_markup('<p align="center">\n<img src="x.png"/>') == ""
