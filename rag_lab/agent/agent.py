from dataclasses import dataclass, field

from rag_lab.agent.parser import ParseError, ReActParser
from rag_lab.agent.tools import Tool
from rag_lab.llms.base import LLM
from rag_lab.prompts import PromptBuilder
from rag_lab.retrievers.base import RetrievalResult
from rag_lab.types import Chunk

_REACT_INSTRUCTIONS = """\
You are a retrieval agent. Answer the user's question by gathering evidence with the \
tools below. Work in steps. Each step, output exactly:

Thought: <your reasoning>
Action: <one tool name>
Action Input: <input for the tool>

When you have gathered enough evidence, output:

Thought: <why you are ready>
Final Answer

Do not write the answer itself — stop at "Final Answer" and the system will compose it.

Available tools:
"""


@dataclass
class AgentStep:
    thought: str
    action: str | None
    action_input: str | None
    observation: str | None
    chunks: list[Chunk] = field(default_factory=list)


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep]
    chunks_seen: list[Chunk]
    final_context: list[Chunk]


def _render_prompt(question: str, tools: list[Tool], scratchpad: str) -> str:
    parts = [_REACT_INSTRUCTIONS]
    for tool in tools:
        parts.append(f"- {tool.name}: {tool.description}")
    parts.append("")
    parts.append(f"Question: {question}")
    if scratchpad:
        parts.append(scratchpad)
    return "\n".join(parts)


def _dedupe(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[tuple[str, int]] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        key = (str(chunk.doc_path), chunk.position)
        if key not in seen:
            seen.add(key)
            out.append(chunk)
    return out


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool],
        prompt_builder: PromptBuilder | None = None,
        *,
        max_steps: int = 6,
        final_k: int = 5,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if final_k <= 0:
            raise ValueError("final_k must be positive")
        self.llm = llm
        self.tools = list(tools)
        self._by_name = {tool.name: tool for tool in tools}
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.max_steps = max_steps
        self.final_k = final_k
        self.parser = ReActParser()

    def run(self, question: str) -> AgentResult:
        scratchpad = ""
        steps: list[AgentStep] = []
        seen: list[Chunk] = []
        parse_failures = 0

        for _ in range(self.max_steps):
            prompt = _render_prompt(question, self.tools, scratchpad)
            text = self.llm.generate(prompt)
            try:
                parsed = self.parser.parse(text)
            except ParseError:
                parse_failures += 1
                if parse_failures >= 2:
                    break
                reminder = (
                    "Observation: Could not parse your step. Use 'Action: <tool>' and "
                    "'Action Input: <text>', or 'Final Answer' when ready."
                )
                scratchpad += f"\n{reminder}\n"
                continue

            parse_failures = 0

            if parsed.is_final:
                steps.append(
                    AgentStep(
                        thought=parsed.thought,
                        action=None,
                        action_input=None,
                        observation=None,
                    )
                )
                break

            tool = self._by_name.get(parsed.action)
            if tool is None:
                observation = (
                    f"Error: unknown tool '{parsed.action}'. "
                    f"Available: {', '.join(self._by_name)}."
                )
                step_chunks: list[Chunk] = []
            else:
                result = tool.run(parsed.action_input)
                observation = result.observation
                step_chunks = result.chunks
                seen.extend(step_chunks)

            steps.append(
                AgentStep(
                    thought=parsed.thought,
                    action=parsed.action,
                    action_input=parsed.action_input,
                    observation=observation,
                    chunks=step_chunks,
                )
            )
            scratchpad += (
                f"\nThought: {parsed.thought}"
                f"\nAction: {parsed.action}"
                f"\nAction Input: {parsed.action_input}"
                f"\nObservation: {observation}\n"
            )

        chunks_seen = _dedupe(seen)
        final_context = chunks_seen[: self.final_k]
        answer = self._synthesize(question, final_context)
        return AgentResult(
            answer=answer,
            steps=steps,
            chunks_seen=chunks_seen,
            final_context=final_context,
        )

    def _synthesize(self, question: str, chunks: list[Chunk]) -> str:
        results = [
            RetrievalResult(chunk=chunk, score=0.0, source="agent") for chunk in chunks
        ]
        prompt = self.prompt_builder.build(question=question, results=results)
        return self.llm.generate(prompt)
