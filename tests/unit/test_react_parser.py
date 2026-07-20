import pytest

from rag_lab.agent.parser import ParseError, ReActParser


def test_parses_action_and_input():
    text = "Thought: I need docs.\nAction: vector_search\nAction Input: how to configure"
    step = ReActParser().parse(text)
    assert step.is_final is False
    assert step.action == "vector_search"
    assert step.action_input == "how to configure"
    assert step.thought == "I need docs."


def test_parses_final_answer_marker():
    text = "Thought: I have enough now.\nFinal Answer"
    step = ReActParser().parse(text)
    assert step.is_final is True
    assert step.action is None


def test_multiline_action_input_is_captured():
    text = "Action: fetch_document\nAction Input: docs/guide.md"
    step = ReActParser().parse(text)
    assert step.action == "fetch_document"
    assert step.action_input == "docs/guide.md"


def test_final_answer_mention_in_thought_does_not_end_step():
    text = (
        "Thought: I don't have a final answer yet, let me search.\n"
        "Action: vector_search\nAction Input: q"
    )
    step = ReActParser().parse(text)
    assert step.is_final is False
    assert step.action == "vector_search"


def test_unparseable_text_raises():
    with pytest.raises(ParseError):
        ReActParser().parse("I will just chat without any action.")


def test_parses_json_tool_call():
    text = '{"thought": "need docs", "action": "vector_search", "action_input": "how to"}'
    step = ReActParser().parse(text)
    assert step.is_final is False
    assert step.action == "vector_search"
    assert step.action_input == "how to"
    assert step.thought == "need docs"


def test_parses_json_final_answer():
    text = '{"thought": "done", "action": "final_answer", "action_input": ""}'
    step = ReActParser().parse(text)
    assert step.is_final is True
    assert step.action is None


def test_json_without_action_input_defaults_empty():
    step = ReActParser().parse('{"thought": "t", "action": "list_documents"}')
    assert step.action == "list_documents"
    assert step.action_input == ""


def test_malformed_json_falls_back_to_freetext():
    text = '{"thought": broken json\nAction: vector_search\nAction Input: q'
    step = ReActParser().parse(text)
    assert step.action == "vector_search"
    assert step.action_input == "q"
