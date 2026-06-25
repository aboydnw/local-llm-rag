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
