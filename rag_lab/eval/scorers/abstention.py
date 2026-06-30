def detected_abstention(answer: str, markers: list[str]) -> bool:
    lower = answer.lower()
    return any(marker.lower() in lower for marker in markers)
