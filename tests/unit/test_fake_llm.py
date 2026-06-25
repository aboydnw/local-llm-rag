from rag_lab.llms.fake import FakeLLM


def test_generate_returns_configured_reply():
    assert FakeLLM("hello there").generate("anything") == "hello there"


def test_stream_yields_pieces_that_join_to_reply():
    pieces = list(FakeLLM("titiler is a tile server").stream("q"))
    assert len(pieces) > 1
    assert "".join(pieces) == "titiler is a tile server"
