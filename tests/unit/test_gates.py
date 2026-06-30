from rag_lab.eval.gates import gate_failures


def test_gate_failure_when_metric_drops_past_threshold() -> None:
    failures = gate_failures(
        current={"recall@k": 0.40},
        baseline={"recall@k": 0.50},
        gates={"recall@k": 0.05},
    )
    assert len(failures) == 1


def test_no_failure_within_allowed_drop() -> None:
    failures = gate_failures(
        current={"recall@k": 0.47},
        baseline={"recall@k": 0.50},
        gates={"recall@k": 0.05},
    )
    assert failures == []


def test_missing_metric_is_skipped() -> None:
    assert gate_failures(current={}, baseline={"recall@k": 0.5}, gates={"recall@k": 0.05}) == []
