def is_api_stub(text: str) -> bool:
    """Return True if ``text`` is only mkdocstrings directives and indented options."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    has_directive = False
    for line in lines:
        if line.lstrip().startswith(":::"):
            has_directive = True
            continue
        if line[:1].isspace():
            continue
        return False
    return has_directive
