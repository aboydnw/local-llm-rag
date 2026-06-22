import os
import re
from dataclasses import dataclass
from functools import lru_cache

from rag_lab.llms.base import LLM

JUDGE_PROMPT = """\
You are grading a documentation-assistant answer.

QUESTION:
{question}

REFERENCE ANSWER (ground truth):
{ideal_answer}

ACTUAL ANSWER (to be graded):
{actual_answer}

Grade the ACTUAL ANSWER on a 1-5 scale for how well it matches the reference:
  5 = accurate and complete
  4 = accurate, missing minor detail
  3 = mostly accurate, missing significant detail or has minor error
  2 = partially accurate, significant gaps or errors
  1 = inaccurate or off-topic

Respond in this exact format on two lines:
SCORE: <integer 1-5>
Reason: <one sentence>
"""


@dataclass(frozen=True, slots=True)
class JudgeResult:
    score: int
    reason: str


@lru_cache(maxsize=1)
def _client():
    import anthropic

    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


_SCORE_RE = re.compile(r"SCORE:\s*([1-5])", re.IGNORECASE)
_REASON_RE = re.compile(r"Reason:\s*(.+)", re.IGNORECASE)


def parse_judge_response(text: str) -> JudgeResult:
    """Extract the 1-5 score and one-line reason from a judge model's reply."""
    score_match = _SCORE_RE.search(text)
    reason_match = _REASON_RE.search(text)
    return JudgeResult(
        score=int(score_match.group(1)) if score_match else 0,
        reason=reason_match.group(1).strip() if reason_match else text.strip(),
    )


class LLMJudge:
    """Grades answers with Anthropic's API. Requires ``ANTHROPIC_API_KEY``."""

    def __init__(self, model: str = "claude-sonnet-4-6") -> None:
        self.model = model

    def score(self, question: str, actual_answer: str, ideal_answer: str) -> JudgeResult:
        response = _client().messages.create(
            model=self.model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=question,
                        actual_answer=actual_answer,
                        ideal_answer=ideal_answer,
                    ),
                }
            ],
        )
        blocks = getattr(response, "content", None) or []
        text = "".join(
            getattr(b, "text", "") for b in blocks if getattr(b, "type", "text") == "text"
        )
        return parse_judge_response(text)


class OllamaJudge:
    """Grades answers with a local Ollama model. No API key required."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def score(self, question: str, actual_answer: str, ideal_answer: str) -> JudgeResult:
        text = self._llm.generate(
            JUDGE_PROMPT.format(
                question=question,
                actual_answer=actual_answer,
                ideal_answer=ideal_answer,
            )
        )
        return parse_judge_response(text)
