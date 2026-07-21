"""Eval MCP server: rag-lab's eval harness as tools any MCP client can call.

Requires the optional ``mcp`` extra (``uv sync --extra mcp``); started via
``rag-lab mcp``. Never imported by modules the core CLI loads at startup.

The point of these tools is a self-verifying loop: an agent edits a RAG knob in
``rag.yml`` (or passes ``config_overrides``), calls ``run_eval``, reads the
baseline deltas, and keeps or reverts the change based on measured evidence.
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from rag_lab.eval import mcp_tools

CONFIG_PATH = Path("rag.yml")
DB_PATH = Path("rag.db")
GOLDEN_PATH = Path("golden.yml")
RUNS_DIR = Path(".rag-lab") / "runs"

server = FastMCP(
    "rag-lab",
    instructions=(
        "Eval harness for a local RAG pipeline. run_eval scores the golden set "
        "and compares against the pinned baseline; use it to prove a config "
        "change helped before keeping it."
    ),
)


@server.tool()
def run_eval(config_overrides: dict | None = None, repeat: int = 1) -> dict:
    """Run the golden-set eval; returns run_id, scores, and deltas vs the baseline.

    config_overrides deep-merges into rag.yml (e.g. {"retriever": {"k": 8}}).
    repeat > 1 reruns the set N times and reports mean scores with stdev.
    """
    try:
        return mcp_tools.run_eval(
            config_path=CONFIG_PATH,
            db_path=DB_PATH,
            golden_path=GOLDEN_PATH,
            runs_dir=RUNS_DIR,
            config_overrides=config_overrides,
            repeat=repeat,
        )
    except Exception as exc:  # noqa: BLE001 - structured error beats a dead server
        return {"error": f"{type(exc).__name__}: {exc}"}


@server.tool()
def list_runs() -> list[dict] | dict:
    """List saved eval runs, newest first; the pinned baseline is flagged."""
    try:
        return mcp_tools.list_runs(runs_dir=RUNS_DIR)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


@server.tool()
def compare_runs(run_a: str, run_b: str) -> dict:
    """Changed config knobs and per-metric deltas (run_b minus run_a)."""
    try:
        return mcp_tools.compare_runs(run_a, run_b, runs_dir=RUNS_DIR)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


@server.tool()
def get_failures(run_id: str) -> dict:
    """Questions in a run whose gated metrics regressed past eval.gates thresholds."""
    try:
        return mcp_tools.get_failures(run_id, runs_dir=RUNS_DIR)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


@server.tool()
def add_golden_case(
    question: str,
    ideal_docs: list[str] | None = None,
    must_mention: list[str] | None = None,
    ideal_answer: str = "",
    expect_abstention: bool = False,
    case_id: str | None = None,
) -> dict:
    """Append a case to the golden set (id slugged from the question if omitted)."""
    try:
        return mcp_tools.add_golden_case(
            question,
            golden_path=GOLDEN_PATH,
            ideal_docs=ideal_docs,
            must_mention=must_mention,
            ideal_answer=ideal_answer,
            expect_abstention=expect_abstention,
            case_id=case_id,
        )
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
