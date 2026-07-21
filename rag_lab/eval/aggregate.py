import math
import statistics

from rag_lab.eval.runner import EvalResult


def aggregate_metrics(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {}
    agg = {
        "recall@k": statistics.mean(r.recall_at_k for r in results),
        "ndcg@k": statistics.mean(r.ndcg_at_k for r in results),
        "map": statistics.mean(r.average_precision for r in results),
        "mrr": statistics.mean(r.mrr for r in results),
        "keyword_coverage": statistics.mean(r.keyword_coverage for r in results),
    }
    cited = [r.citation_validity for r in results if r.citation_validity is not None]
    if cited:
        agg["citation_validity"] = statistics.mean(cited)
    flagged = [r for r in results if r.expected_abstention]
    if flagged:
        agg["abstention_accuracy"] = statistics.mean(
            1.0 if r.abstained else 0.0 for r in flagged
        )
    for key in sorted({k for r in results for k in r.deepeval_scores}):
        vals = [
            r.deepeval_scores[key]
            for r in results
            if key in r.deepeval_scores and not math.isnan(r.deepeval_scores[key])
        ]
        if vals:
            agg[key] = statistics.mean(vals)
    for key in sorted({k for r in results for k in r.agent_metrics}):
        vals = [r.agent_metrics[key] for r in results if key in r.agent_metrics]
        if vals:
            agg[key] = statistics.mean(vals)
    return agg


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(q * len(ordered)) - 1)
    return ordered[index]


def _tps(tokens: float, ms: float) -> float:
    return tokens / (ms / 1000.0) if ms else 0.0


def aggregate_perf(results: list[EvalResult]) -> dict[str, float]:
    """Mean/p50/p95 throughput and latency across results that captured stats."""
    captured = [r.generation_stats for r in results if r.generation_stats]
    if not captured:
        return {}
    series = {
        "prompt_eval_tps": [
            _tps(s["prompt_tokens"], s["prompt_eval_ms"]) for s in captured
        ],
        "generation_tps": [_tps(s["output_tokens"], s["generation_ms"]) for s in captured],
        "total_ms": [s["prompt_eval_ms"] + s["generation_ms"] for s in captured],
    }
    agg: dict[str, float] = {}
    for name, values in series.items():
        agg[f"{name}_mean"] = statistics.mean(values)
        agg[f"{name}_p50"] = _percentile(values, 0.50)
        agg[f"{name}_p95"] = _percentile(values, 0.95)
    return agg
