def gate_failures(
    current: dict[str, float],
    baseline: dict[str, float],
    gates: dict[str, float],
) -> list[str]:
    failures: list[str] = []
    for metric, max_drop in gates.items():
        if metric not in current or metric not in baseline:
            continue
        if current[metric] < baseline[metric] - max_drop:
            failures.append(
                f"{metric}: {current[metric]:.3f} < baseline {baseline[metric]:.3f} "
                f"- {max_drop:.3f}"
            )
    return failures
