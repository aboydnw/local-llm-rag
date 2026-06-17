import re
import statistics
from datetime import UTC, datetime
from pathlib import Path

from rag_lab.eval.runner import EvalResult


class MarkdownReporter:
    def write(
        self,
        results: list[EvalResult],
        config_summary: str,
        out_path: Path,
        previous_report: Path | None = None,
    ) -> None:
        out_path.write_text(self._render(results, config_summary, previous_report))

    def _render(
        self,
        results: list[EvalResult],
        config_summary: str,
        previous_report: Path | None,
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

        if previous_report is not None and previous_report.exists():
            prev = self._parse_aggregates(previous_report.read_text())
            if prev:
                lines.append(f"## Diff vs `{previous_report.name}`")
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

        lines.append("## Per-question detail")
        lines.append("")
        has_judge = any(r.judge_score is not None for r in results)
        if has_judge:
            header = "| id | recall@k | mrr | keyword | judge | judge reason |"
            sep = "|---|---|---|---|---|---|"
        else:
            header = "| id | recall@k | mrr | keyword |"
            sep = "|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for r in results:
            row = f"| {r.item_id} | {r.recall_at_k:.2f} | {r.mrr:.2f} | {r.keyword_coverage:.2f} |"
            if has_judge:
                judge_val = str(r.judge_score) if r.judge_score is not None else "n/a"
                reason = re.sub(r"\s+", " ", (r.judge_reason or "").replace("|", "/")).strip()
                row += f" {judge_val} | {reason} |"
            lines.append(row)
        lines.append("")

        return "\n".join(lines)

    def _aggregates(self, results: list[EvalResult]) -> dict[str, float]:
        if not results:
            return {}
        agg = {
            "recall@k": statistics.mean(r.recall_at_k for r in results),
            "mrr": statistics.mean(r.mrr for r in results),
            "keyword_coverage": statistics.mean(r.keyword_coverage for r in results),
        }
        judge_scores = [r.judge_score for r in results if r.judge_score is not None]
        if judge_scores:
            agg["judge_score"] = statistics.mean(judge_scores)
        return agg

    @staticmethod
    def _parse_aggregates(content: str) -> dict[str, float]:
        rows = re.findall(r"^\|\s*([\w@]+)\s*\|\s*([0-9.]+)\s*\|\s*$", content, re.MULTILINE)
        return {name: float(value) for name, value in rows}
