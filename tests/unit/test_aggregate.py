from rag_lab.eval.aggregate import aggregate_metrics, aggregate_perf
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


def _perf_r(prompt_tokens, prompt_eval_ms, output_tokens, generation_ms) -> EvalResult:
    return EvalResult(
        item_id="x", question="q", actual_answer="a",
        recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0,
        generation_stats={
            "prompt_tokens": prompt_tokens,
            "prompt_eval_ms": prompt_eval_ms,
            "output_tokens": output_tokens,
            "generation_ms": generation_ms,
        },
    )


def test_aggregate_perf_computes_throughput_and_latency() -> None:
    results = [
        _perf_r(1000, 2000.0, 100, 10000.0),
        _perf_r(2000, 2000.0, 200, 10000.0),
    ]
    perf = aggregate_perf(results)
    assert perf["prompt_eval_tps_mean"] == 750.0
    assert perf["generation_tps_p50"] == 10.0
    assert perf["generation_tps_p95"] == 20.0
    assert perf["total_ms_mean"] == 12000.0


def test_aggregate_perf_empty_when_no_stats_captured() -> None:
    assert aggregate_perf([_agent_r({})]) == {}


def test_aggregate_includes_mean_parse_failures() -> None:
    results = [
        _agent_r({"parse_failures": 0.0}),
        _agent_r({"parse_failures": 2.0}),
    ]
    agg = aggregate_metrics(results)
    assert agg["parse_failures"] == 1.0
