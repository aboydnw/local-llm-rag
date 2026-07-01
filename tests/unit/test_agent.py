from pathlib import Path

from rag_lab.agent.agent import Agent, AgentResult
from rag_lab.agent.tools import ToolResult
from rag_lab.llms.fake import ScriptedLLM
from rag_lab.types import Chunk


def _chunk(text: str, doc: str = "d.md", position: int = 0) -> Chunk:
    return Chunk(text=text, doc_path=Path(doc), heading_path=("H",), position=position)


class _RecordingTool:
    def __init__(self, name: str, result: ToolResult) -> None:
        self.name = name
        self.description = "test tool"
        self.result = result
        self.calls: list[str] = []

    def run(self, arg: str) -> ToolResult:
        self.calls.append(arg)
        return self.result


def test_agent_dispatches_tool_then_synthesizes():
    tool = _RecordingTool(
        "vector_search", ToolResult(observation="found it", chunks=[_chunk("hit")])
    )
    llm = ScriptedLLM(
        [
            "Thought: search\nAction: vector_search\nAction Input: my query",
            "Thought: done\nFinal Answer",
            "The synthesized answer [1].",
        ]
    )
    agent = Agent(llm=llm, tools=[tool])
    result = agent.run("a question")
    assert isinstance(result, AgentResult)
    assert tool.calls == ["my query"]
    assert result.answer == "The synthesized answer [1]."
    assert [c.text for c in result.chunks_seen] == ["hit"]
    assert [c.text for c in result.final_context] == ["hit"]


def test_agent_caps_at_max_steps_then_synthesizes():
    tool = _RecordingTool(
        "vector_search", ToolResult(observation="x", chunks=[_chunk("c")])
    )
    loop_reply = "Thought: again\nAction: vector_search\nAction Input: q"
    llm = ScriptedLLM([loop_reply, loop_reply, "final answer text"])
    agent = Agent(llm=llm, tools=[tool], max_steps=2)
    result = agent.run("q")
    assert len(tool.calls) == 2
    assert result.answer == "final answer text"


def test_agent_recovers_from_unknown_tool():
    good = _RecordingTool(
        "vector_search", ToolResult(observation="ok", chunks=[_chunk("c")])
    )
    llm = ScriptedLLM(
        [
            "Thought: typo\nAction: nope_tool\nAction Input: q",
            "Thought: retry\nAction: vector_search\nAction Input: q",
            "Thought: done\nFinal Answer",
            "answer",
        ]
    )
    agent = Agent(llm=llm, tools=[good], max_steps=6)
    result = agent.run("q")
    assert good.calls == ["q"]
    assert result.answer == "answer"
    assert any(
        s.observation is not None and s.observation.startswith("Error:")
        for s in result.steps
    )


def test_agent_dedupes_chunks_seen():
    same = _chunk("dup", "d.md", 0)
    tool = _RecordingTool("vector_search", ToolResult(observation="x", chunks=[same]))
    llm = ScriptedLLM(
        [
            "Action: vector_search\nAction Input: q",
            "Action: vector_search\nAction Input: q",
            "Final Answer",
            "answer",
        ]
    )
    agent = Agent(llm=llm, tools=[tool], max_steps=6)
    result = agent.run("q")
    assert len(result.chunks_seen) == 1


def test_agent_survives_nonconsecutive_parse_failures():
    tool = _RecordingTool(
        "vector_search", ToolResult(observation="ok", chunks=[_chunk("c")])
    )
    llm = ScriptedLLM(
        [
            "I will just chat.",
            "Action: vector_search\nAction Input: q",
            "I will just chat again.",
            "Thought: done\nFinal Answer",
            "answer",
        ]
    )
    agent = Agent(llm=llm, tools=[tool], max_steps=6)
    result = agent.run("q")
    assert tool.calls == ["q"]
    assert result.answer == "answer"


def test_agent_uses_custom_instructions_in_prompt():
    tool = _RecordingTool(
        "vector_search", ToolResult(observation="ok", chunks=[_chunk("c")])
    )
    llm = ScriptedLLM(["Thought: done\nFinal Answer", "answer"])
    agent = Agent(llm=llm, tools=[tool], instructions="CUSTOM AGENT PROMPT")
    agent.run("q")
    assert "CUSTOM AGENT PROMPT" in llm.prompts[0]


def test_agent_counts_llm_calls_including_retries_and_synthesis():
    tool = _RecordingTool(
        "vector_search", ToolResult(observation="ok", chunks=[_chunk("c")])
    )
    llm = ScriptedLLM(
        [
            "I will just chat.",
            "Action: vector_search\nAction Input: q",
            "Thought: done\nFinal Answer",
            "answer",
        ]
    )
    agent = Agent(llm=llm, tools=[tool], max_steps=6)
    result = agent.run("q")
    assert result.llm_calls == 4


def test_agent_records_step_prompt_and_synthesis_prompt():
    tool = _RecordingTool(
        "vector_search", ToolResult(observation="found", chunks=[_chunk("hit")])
    )
    llm = ScriptedLLM(
        [
            "Thought: search\nAction: vector_search\nAction Input: my query",
            "Thought: done\nFinal Answer",
            "The answer [1].",
        ]
    )
    agent = Agent(llm=llm, tools=[tool])
    result = agent.run("a question")
    assert result.steps[0].prompt == llm.prompts[0]
    assert "a question" in result.synthesis_prompt
    assert "hit" in result.synthesis_prompt


def test_trace_dict_omits_chunks():
    from rag_lab.agent.agent import AgentStep, trace_dict

    step = AgentStep(
        thought="t", action="vector_search", action_input="q",
        observation="o", chunks=[_chunk("c")], prompt="P",
    )
    d = trace_dict(step)
    assert d == {
        "thought": "t", "action": "vector_search", "action_input": "q",
        "observation": "o", "prompt": "P",
    }


def test_agent_final_context_capped_at_final_k():
    chunks = [_chunk(f"c{i}", "d.md", i) for i in range(5)]
    tool = _RecordingTool("vector_search", ToolResult(observation="x", chunks=chunks))
    llm = ScriptedLLM(
        ["Action: vector_search\nAction Input: q", "Final Answer", "answer"]
    )
    agent = Agent(llm=llm, tools=[tool], max_steps=6, final_k=2)
    result = agent.run("q")
    assert len(result.final_context) == 2
    assert len(result.chunks_seen) == 5
