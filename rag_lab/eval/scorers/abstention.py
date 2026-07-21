def detected_abstention(answer: str, markers: list[str]) -> bool:
    lower = answer.casefold()
    normalized = (marker.strip().casefold() for marker in markers)
    return any(marker and marker in lower for marker in normalized)
