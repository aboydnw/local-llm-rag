from rag_lab.studio import feedback


def test_log_event_callable_without_recorder():
    assert feedback.log_event("query", question="hi") is None


def test_instrument_usable_as_context_manager():
    with feedback.instrument("retrieval", k=5):
        pass
