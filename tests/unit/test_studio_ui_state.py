from rag_lab.studio import ui_state


def test_normalized_weights_complements_to_one():
    assert ui_state.normalized_weights(0.7) == (0.7, 0.3)


def test_normalized_weights_clamps():
    assert ui_state.normalized_weights(1.4) == (1.0, 0.0)
    assert ui_state.normalized_weights(-0.2) == (0.0, 1.0)


def test_knob_cost_sets_are_disjoint_and_complete():
    assert ui_state.EXPENSIVE_KNOBS == {"chunker", "embedder"}
    assert ui_state.CHEAP_KNOBS == {"retriever", "llm"}
    assert ui_state.EXPENSIVE_KNOBS & ui_state.CHEAP_KNOBS == set()


def test_init_state_seeds_missing_keys():
    session = {}
    from rag_lab.config import Config

    ui_state.init_state(session, Config(), "corpus", "golden.yml")
    assert "config" in session
    assert session["corpus"] == "corpus"
    assert session["golden"] == "golden.yml"


def test_init_state_seeds_corpus_name_none():
    from rag_lab.config import Config

    session = {}
    ui_state.init_state(session, Config(), ".", "golden.yml")
    assert session["corpus_name"] is None
