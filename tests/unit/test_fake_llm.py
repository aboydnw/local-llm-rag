from rag_lab.llms.base import LLM
from rag_lab.llms.fake import FakeLLM, ScriptedLLM


def test_generate_returns_configured_reply():
    assert FakeLLM("hello there").generate("anything") == "hello there"


def test_stream_yields_pieces_that_join_to_reply():
    pieces = list(FakeLLM("titiler is a tile server").stream("q"))
    assert len(pieces) > 1
    assert "".join(pieces) == "titiler is a tile server"


def test_scripted_llm_returns_replies_in_order():
    llm = ScriptedLLM(["one", "two"])
    assert llm.generate("a") == "one"
    assert llm.generate("b") == "two"


def test_scripted_llm_records_prompts():
    llm = ScriptedLLM(["x"])
    llm.generate("the prompt")
    assert llm.prompts == ["the prompt"]


def test_fakes_satisfy_llm_protocol():
    fake: LLM = FakeLLM("x")
    scripted: LLM = ScriptedLLM(["x"])
    assert fake.last_stats() is None
    assert scripted.last_stats() is None


def test_fake_llm_stats_are_deterministic():
    llm = FakeLLM("one two three")
    llm.generate("a four word prompt")
    stats = llm.last_stats()
    assert stats is not None
    assert stats.prompt_tokens == 4
    assert stats.output_tokens == 3
    assert llm.generate("a four word prompt") and llm.last_stats() == stats


def test_scripted_llm_stats_track_last_call():
    llm = ScriptedLLM(["one two", "one two three"])
    llm.generate("p1")
    assert llm.last_stats().output_tokens == 2
    llm.generate("p2 p2")
    assert llm.last_stats().output_tokens == 3
    assert llm.last_stats().prompt_tokens == 2
