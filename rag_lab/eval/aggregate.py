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
    return agg
