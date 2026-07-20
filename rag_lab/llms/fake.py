from collections.abc import Iterator

from rag_lab.types import GenerationStats


def _fake_stats(prompt: str, reply: str) -> GenerationStats:
    return GenerationStats(
        prompt_tokens=len(prompt.split()),
        prompt_eval_ms=100.0,
        output_tokens=len(reply.split()),
        generation_ms=200.0,
    )


class FakeLLM:
    """Deterministic LLM for tests. Returns a fixed reply, streamed word by word."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self._last_stats: GenerationStats | None = None

    def generate(self, prompt: str) -> str:
        return "".join(self.stream(prompt))

    def stream(self, prompt: str) -> Iterator[str]:
        self._last_stats = _fake_stats(prompt, self.reply)
        words = self.reply.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else f"{word} "

    def last_stats(self) -> GenerationStats | None:
        return self._last_stats


class ScriptedLLM:
    """Deterministic LLM that returns a queued sequence of replies, one per call.

    Used to drive the agent loop in tests: supply one reply per expected turn,
    plus a final reply for the synthesis step.
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self._index = 0
        self._last_stats: GenerationStats | None = None

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self.replies[self._index]
        self._index += 1
        self._last_stats = _fake_stats(prompt, reply)
        return reply

    def stream(self, prompt: str) -> Iterator[str]:
        yield self.generate(prompt)

    def last_stats(self) -> GenerationStats | None:
        return self._last_stats
