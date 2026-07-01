import re
from dataclasses import dataclass

_THOUGHT = re.compile(r"Thought:\s*(.*?)(?=\n(?:Action|Final Answer)|\Z)", re.DOTALL)
_ACTION = re.compile(r"Action:\s*([A-Za-z_]+)")
_INPUT = re.compile(r"Action Input:\s*(.*)", re.DOTALL)


class ParseError(Exception):
    """Raised when model output has no action and no final-answer marker."""


@dataclass(frozen=True)
class ParsedStep:
    thought: str
    action: str | None
    action_input: str
    is_final: bool


class ReActParser:
    def parse(self, text: str) -> ParsedStep:
        thought_match = _THOUGHT.search(text)
        thought = thought_match.group(1).strip() if thought_match else ""

        if "Final Answer" in text:
            return ParsedStep(
                thought=thought, action=None, action_input="", is_final=True
            )

        action_match = _ACTION.search(text)
        if action_match is None:
            raise ParseError("no action or final-answer marker found")

        input_match = _INPUT.search(text)
        action_input = input_match.group(1).strip() if input_match else ""
        return ParsedStep(
            thought=thought,
            action=action_match.group(1).strip(),
            action_input=action_input,
            is_final=False,
        )
