from rag_lab.chunkers import splitting


def _enc():
    return splitting.get_encoder()


def test_token_windows_yields_original_text_when_under_cap():
    windows = list(splitting.token_windows("short text", _enc(), max_tokens=100, overlap=10))
    assert windows == ["short text"]


def test_token_windows_splits_with_token_overlap():
    enc = _enc()
    text = ("word " * 300).strip()
    windows = list(splitting.token_windows(text, enc, max_tokens=50, overlap=10))
    assert len(windows) >= 2
    first_tail = enc.encode(windows[0])[-10:]
    second_head = enc.encode(windows[1])[:10]
    assert first_tail == second_head


def test_split_sentences_breaks_on_terminal_punctuation():
    result = splitting.split_sentences("First one. Second one! Third one?")
    assert result == ["First one.", "Second one!", "Third one?"]


def test_split_sentences_returns_whole_text_when_no_boundaries():
    assert splitting.split_sentences("no terminal punctuation here") == [
        "no terminal punctuation here"
    ]


def test_cascade_split_keeps_small_paragraphs_together():
    text = "Para one has words.\n\nPara two has words."
    chunks = list(splitting.cascade_split(text, _enc(), max_tokens=100, overlap=0))
    assert chunks == [text]


def test_cascade_split_falls_back_to_sentences_for_oversized_paragraph():
    body = " ".join(f"Sentence number {i} has several words." for i in range(60))
    chunks = list(splitting.cascade_split(body, _enc(), max_tokens=40, overlap=0))
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.rstrip().endswith(".")


def test_cascade_split_windows_a_single_oversized_sentence():
    body = "word " * 400
    chunks = list(splitting.cascade_split(body.strip(), _enc(), max_tokens=50, overlap=0))
    assert len(chunks) >= 2


def test_cascade_split_never_exceeds_max_tokens_with_overlap():
    enc = _enc()
    body = " ".join(
        f"Alpha beta gamma delta epsilon zeta eta theta number {i} done."
        for i in range(40)
    )
    max_tokens = 20
    chunks = list(splitting.cascade_split(body, enc, max_tokens=max_tokens, overlap=15))
    assert chunks
    assert all(len(enc.encode(c)) <= max_tokens for c in chunks)
