"""Shared eval execution for the CLI and MCP server (Studio has its own index path)."""

from pathlib import Path

from rag_lab import pipeline
from rag_lab.config import Config
from rag_lab.eval import golden_set as golden_set_mod
from rag_lab.eval.runner import EvalResult, EvalRunner


def build_runner(cfg: Config, db: Path, *, use_agent: bool = False) -> EvalRunner:
    """Assemble an EvalRunner from an existing index and the given config."""
    store = pipeline.build_store(cfg, db)
    embedder = pipeline.build_embedder(cfg)
    retriever = pipeline.build_retriever(store, embedder, cfg)
    llm = pipeline.build_llm(cfg)
    scorer = None
    if cfg.eval.deepeval:
        from rag_lab.eval.scorers.deepeval_scorer import DeepEvalScorer

        scorer = DeepEvalScorer(model=cfg.eval.deepeval_model or cfg.llm.model)
    agent = (
        pipeline.build_agent(store, embedder, cfg)
        if (use_agent or cfg.agent.enabled)
        else None
    )
    return EvalRunner(
        retriever=retriever,
        llm=llm,
        k=cfg.retriever.k,
        deepeval_scorer=scorer,
        prompt_builder=pipeline.build_prompt_builder(cfg),
        abstention_markers=cfg.eval.abstention_markers,
        agent=agent,
    )


def run_eval(
    cfg: Config, db: Path, golden: Path, *, repeat: int = 1, use_agent: bool = False
) -> list[list[EvalResult]]:
    """Run the golden set `repeat` times and return per-repeat result lists."""
    runner = build_runner(cfg, db, use_agent=use_agent)
    items = golden_set_mod.load_golden_set(golden)
    return [runner.run(items) for _ in range(max(1, repeat))]
