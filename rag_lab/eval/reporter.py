import math
from datetime import UTC, datetime
from pathlib import Path

from rag_lab.eval.aggregate import aggregate_metrics, aggregate_perf
from rag_lab.eval.runner import EvalResult


class MarkdownReporter:
    def write(
        self,
        results: list[EvalResult],
        config_summary: str,
        out_path: Path,
        previous_run: Path | None = None,
    ) -> None:
        out_path.write_text(self._render(results, config_summary, previous_run))

    def _render(
        self,
        results: list[EvalResult],
        config_summary: str,
        previous_run: Path | None,
    ) -> str:
        aggregates = self._aggregates(results)
        lines: list[str] = []
        lines.append("# rag-lab eval report")
        lines.append("")
        lines.append(f"_Generated {datetime.now(UTC).isoformat(timespec='seconds')}_")
        lines.append("")
        lines.append(f"**Config:** {config_summary}")
        lines.append(f"**Questions evaluated:** {len(results)}")
        lines.append("")
        lines.append("## Aggregates")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for k, v in aggregates.items():
            lines.append(f"| {k} | {v:.2f} |")
        lines.append("")

        if previous_run is not None and previous_run.exists():
            from rag_lab.eval.run_artifact import read_run

            try:
                data = read_run(previous_run)
                prev = data.get("aggregates", {}) if isinstance(data, dict) else {}
            except (OSError, ValueError):
                prev = {}
            if prev:
                lines.append(f"## Diff vs `{previous_run.name}`")
                lines.append("")
                lines.append("| metric | previous | current | delta |")
                lines.append("|---|---|---|---|")
                for k, v in aggregates.items():
                    p = prev.get(k)
                    if p is None:
                        delta_str = "n/a"
                        prev_str = "n/a"
                    else:
                        delta = v - p
                        sign = "+" if delta >= 0 else ""
                        delta_str = f"{sign}{delta:.2f}"
                        prev_str = f"{p:.2f}"
                    lines.append(f"| {k} | {prev_str} | {v:.2f} | {delta_str} |")
                lines.append("")

        lines.extend(self._perf_section(results))

        lines.append("## Per-question detail")
        lines.append("")
        deepeval_keys = sorted({k for r in results for k in r.deepeval_scores})
        header = "| id | recall@k | ndcg@k | mrr | keyword |" + "".join(
            f" {k} |" for k in deepeval_keys
        )
        sep = "|---|---|---|---|---|" + "---|" * len(deepeval_keys)
        lines.append(header)
        lines.append(sep)
        for r in results:
            row = (
                f"| {r.item_id} | {r.recall_at_k:.2f} | {r.ndcg_at_k:.2f} | "
                f"{r.mrr:.2f} | {r.keyword_coverage:.2f} |"
            )
            for key in deepeval_keys:
                if key in r.deepeval_scores and not math.isnan(r.deepeval_scores[key]):
                    row += f" {r.deepeval_scores[key]:.2f} |"
                else:
                    row += " n/a |"
            lines.append(row)
        lines.append("")

        return "\n".join(lines)

    def _aggregates(self, results: list[EvalResult]) -> dict[str, float]:
        return aggregate_metrics(results)

    def _perf_section(self, results: list[EvalResult]) -> list[str]:
        lines = ["## Performance", ""]
        perf = aggregate_perf(results)
        if not perf:
            lines.append("_Generation stats not captured for this run._")
            lines.append("")
            return lines
        labels = {
            "prompt_eval_tps": "prompt-eval tok/s",
            "generation_tps": "generation tok/s",
            "total_ms": "total latency (ms)",
        }
        lines.append("| metric | mean | p50 | p95 |")
        lines.append("|---|---|---|---|")
        for key, label in labels.items():
            lines.append(
                f"| {label} | {perf[f'{key}_mean']:.1f} | "
                f"{perf[f'{key}_p50']:.1f} | {perf[f'{key}_p95']:.1f} |"
            )
        lines.append("")
        return lines
