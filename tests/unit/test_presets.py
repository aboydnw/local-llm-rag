from rag_lab.config import Config
from rag_lab.studio import presets


def test_eight_presets_with_unique_names():
    names = [p.name for p in presets.PRESETS]
    assert len(names) == 8
    assert len(set(names)) == 8


def test_apply_preset_overrides_only_retrieval_knobs():
    base = Config()
    base.chunker.max_tokens = 256
    base.llm.model = "qwen3:4b"
    cfg = presets.apply_preset(base, presets.PRESETS[3])
    assert cfg.retriever.type == "hybrid"
    assert cfg.retriever.vector_weight == 0.75
    assert cfg.retriever.bm25_weight == 0.25
    assert cfg.chunker == base.chunker
    assert cfg.embedder == base.embedder
    assert cfg.llm == base.llm


def test_apply_preset_pins_k_to_5():
    base = Config()
    base.retriever.k = 10
    for preset in presets.PRESETS:
        assert presets.apply_preset(base, preset).retriever.k == 5


def test_apply_preset_does_not_mutate_base():
    base = Config()
    presets.apply_preset(base, presets.PRESETS[0])
    assert base.retriever.type == "hybrid"
    assert base.retriever.k == 5


def test_rerank_presets_use_llm_reranker():
    reranked = [p for p in presets.PRESETS if p.reranker == "llm"]
    assert {p.name for p in reranked} == {"vector-rerank", "bm25-rerank", "hybrid-rerank"}
