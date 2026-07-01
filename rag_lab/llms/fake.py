from collections.abc import Iterator


class FakeLLM:
    """Deterministic LLM for tests. Returns a fixed reply, streamed word by word."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def generate(self, prompt: str) -> str:
        return "".join(self.stream(prompt))

    def stream(self, prompt: str) -> Iterator[str]:
        words = self.reply.split(" ")
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else f"{word} "


class ScriptedLLM:
    """Deterministic LLM that returns a queued sequence of replies, one per call.

    Used to drive the agent loop in tests: supply one reply per expected turn,
    plus a final reply for the synthesis step.
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self._index = 0

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self.replies[self._index]
        self._index += 1
        return reply

    def stream(self, prompt: str) -> Iterator[str]:
        yield self.generate(prompt)
