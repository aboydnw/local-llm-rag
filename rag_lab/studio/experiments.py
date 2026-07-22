from pathlib import Path

from rag_lab.config import Config
from rag_lab.eval import golden_set as golden_set_mod
from rag_lab.eval import run_store
from rag_lab.eval.run_store import RunRecord
from rag_lab.eval.runner import EvalRunner
from rag_lab.prompts import PromptBuilder
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import components
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio import presets as presets_mod
from rag_lab.studio.corpora import Corpus
from rag_lab.studio.workspace import Workspace


def run_eval(
    workspace: Workspace,
    corpus: Corpus,
    config: Config,
    golden_path: Path,
    run_id: str,
    created_at: str,
    *,
    name: str | None = None,
    repeat: int = 1,
    loader=None,
    embedder=None,
    llm=None,
    extra_provenance: dict[str, str] | None = None,
) -> RunRecord:
    """Build (or reuse) the corpus index, run the golden set, and persist the run."""
    db_path = indexer_mod.ensure_index(
        workspace, corpus, config, loader=loader, embedder=embedder
    )
    if embedder is None:
        embedder = components.build_embedder(config)
    store = SqliteVecStore(db_path, dimension=embedder.dimension)
    retriever = components.build_retriever(store, embedder, config)
    if llm is None:
        llm = components.build_llm(config)
    agent = None
    if config.agent.enabled:
        agent = components.build_agent(store, embedder, config)
        agent.llm = llm
    scorer = None
    if config.eval.deepeval:
        from rag_lab.eval.scorers.deepeval_scorer import DeepEvalScorer

        scorer = DeepEvalScorer(model=config.eval.deepeval_model or config.llm.model)
    runner = EvalRunner(
        retriever=retriever,
        llm=llm,
        k=config.retriever.k,
        deepeval_scorer=scorer,
        prompt_builder=PromptBuilder(system_instructions=config.prompt.system_instructions),
        abstention_markers=config.eval.abstention_markers,
        agent=agent,
    )

    items = golden_set_mod.load_golden_set(golden_path)
    repeats = [runner.run(items) for _ in range(max(1, repeat))]
    return run_store.save_run(
        workspace.runs_dir,
        run_id=run_id,
        name=name,
        created_at=created_at,
        corpus=corpus.label,
        config=config,
        repeats=repeats,
        golden_hash=run_store.golden_hash(golden_path),
        corpus_snapshot=corpus.to_dict(),
        extra_provenance=extra_provenance,
    )


def run_base_sweep(
    workspace: Workspace,
    corpus,
    config: Config,
    golden_path: Path,
    sweep_id: str,
    created_at: str,
    *,
    on_progress=None,
    loader=None,
    embedder=None,
    llm=None,
) -> list[RunRecord]:
    """Run every base preset against the corpus as one tagged sweep."""
    records: list[RunRecord] = []
    for i, preset in enumerate(presets_mod.PRESETS):
        if on_progress is not None:
            on_progress(i, preset.name)
        cfg = presets_mod.apply_preset(config, preset)
        cfg.eval.deepeval = False
        records.append(
            run_eval(
                workspace, corpus, cfg, golden_path,
                run_id=f"{sweep_id}-{preset.name}",
                created_at=created_at,
                name=f"base: {preset.name}",
                repeat=1,
                loader=loader, embedder=embedder, llm=llm,
                extra_provenance={"sweep_id": sweep_id, "preset": preset.name},
            )
        )
    return records


def list_runs(workspace: Workspace) -> list[RunRecord]:
    return run_store.list_runs(workspace.runs_dir)


def load_run(workspace: Workspace, run_id: str) -> RunRecord:
    return run_store.load_run(workspace.runs_dir, run_id)


def load_run_items(workspace: Workspace, run_id: str) -> list[dict]:
    return run_store.load_run_items(workspace.runs_dir, run_id)


def rename_run(workspace: Workspace, run_id: str, name: str) -> None:
    run_store.rename_run(workspace.runs_dir, run_id, name)


def delete_run(workspace: Workspace, run_id: str) -> None:
    run_store.delete_run(workspace.runs_dir, run_id)


def set_baseline(workspace: Workspace, run_id: str) -> None:
    run_store.set_baseline(workspace.runs_dir, run_id)


def get_baseline(workspace: Workspace, corpus: str) -> str | None:
    return run_store.get_baseline(workspace.runs_dir, corpus)


def diff(run_a: RunRecord, run_b: RunRecord) -> dict:
    return run_store.diff(run_a, run_b)
