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


def test_abstention_accuracy_is_rate_over_flagged_items() -> None:
    flagged_hit = EvalResult(
        item_id="a", question="q", actual_answer="a",
        recall_at_k=0.0, mrr=0.0, keyword_coverage=0.0,
        expected_abstention=True, abstained=True,
    )
    flagged_miss = EvalResult(
        item_id="b", question="q", actual_answer="a",
        recall_at_k=0.0, mrr=0.0, keyword_coverage=0.0,
        expected_abstention=True, abstained=False,
    )
    not_flagged = EvalResult(
        item_id="c", question="q", actual_answer="a",
        recall_at_k=0.0, mrr=0.0, keyword_coverage=0.0,
    )
    agg = aggregate_metrics([flagged_hit, flagged_miss, not_flagged])
    assert agg["abstention_accuracy"] == 0.5


def test_abstention_accuracy_absent_when_no_flagged_items() -> None:
    agg = aggregate_metrics([_r(1.0, 1.0, 1.0, None)])
    assert "abstention_accuracy" not in agg
