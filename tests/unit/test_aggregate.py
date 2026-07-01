from rag_lab.eval.aggregate import aggregate_metrics
from rag_lab.eval.runner import EvalResult


def _r(recall, ndcg, ap, cite) -> EvalResult:
    return EvalResult(
        item_id="x", question="q", actual_answer="a",
        recall_at_k=recall, mrr=recall, keyword_coverage=recall,
        ndcg_at_k=ndcg, average_precision=ap, citation_validity=cite,
    )


def test_aggregate_metrics_means_each_metric() -> None:
    agg = aggregate_metrics([_r(1.0, 1.0, 1.0, 1.0), _r(0.0, 0.0, 0.0, None)])
    assert agg["recall@k"] == 0.5
    assert agg["ndcg@k"] == 0.5
    assert agg["map"] == 0.5
    assert agg["citation_validity"] == 1.0


def test_aggregate_metrics_omits_citation_when_never_cited() -> None:
    agg = aggregate_metrics([_r(1.0, 1.0, 1.0, None)])
    assert "citation_validity" not in agg


def _agent_r(agent_metrics: dict[str, float]) -> EvalResult:
    return EvalResult(
        item_id="x", question="q", actual_answer="a",
        recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0,
        agent_metrics=agent_metrics,
    )


def test_aggregate_includes_mean_agent_metrics() -> None:
    results = [
        _agent_r({"tool_calls": 2.0, "recall@k_seen": 1.0}),
        _agent_r({"tool_calls": 4.0, "recall@k_seen": 0.0}),
    ]
    agg = aggregate_metrics(results)
    assert agg["tool_calls"] == 3.0
    assert agg["recall@k_seen"] == 0.5


def test_aggregate_without_agent_metrics_unchanged() -> None:
    agg = aggregate_metrics([_agent_r({})])
    assert "tool_calls" not in agg
