import re

_MKDOCSTRINGS_API_RE = re.compile(
    r"^:::\s*[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)+\s*$"
)


def is_api_stub(text: str) -> bool:
    """Return True if ``text`` is only mkdocstrings directives and indented options."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    has_directive = False
    for line in lines:
        if _MKDOCSTRINGS_API_RE.match(line.strip()):
            has_directive = True
            continue
        if line[:1].isspace():
            continue
        return False
    return has_directive


_HTML_ONLY_RE = re.compile(r"^(?:<[^>]+>\s*)+$")
_IMAGE_ONLY_RE = re.compile(r"^!\[[^\]]*\]\([^)]*\)$")
_BADGE_ONLY_RE = re.compile(r"^\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)$")
_BLANK_RUN_RE = re.compile(r"\n{3,}")


def strip_markup(text: str) -> str:
    """Remove markup-only lines (HTML, standalone images, badges); keep prose."""
    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if (
            _HTML_ONLY_RE.match(stripped)
            or _IMAGE_ONLY_RE.match(stripped)
            or _BADGE_ONLY_RE.match(stripped)
        ):
            continue
        kept.append(line)
    return _BLANK_RUN_RE.sub("\n\n", "\n".join(kept)).strip()
