import pytest

from rag_lab.types import GenerationStats, combine_stats


def test_derived_throughput_properties():
    stats = GenerationStats(
        prompt_tokens=2000,
        prompt_eval_ms=4000.0,
        output_tokens=100,
        generation_ms=10000.0,
    )
    assert stats.prompt_eval_tps == pytest.approx(500.0)
    assert stats.generation_tps == pytest.approx(10.0)
    assert stats.total_ms == pytest.approx(14000.0)


def test_zero_duration_yields_zero_throughput():
    stats = GenerationStats(
        prompt_tokens=10, prompt_eval_ms=0.0, output_tokens=5, generation_ms=0.0
    )
    assert stats.prompt_eval_tps == 0.0
    assert stats.generation_tps == 0.0


def test_combine_stats_sums_tokens_and_durations():
    a = GenerationStats(
        prompt_tokens=100, prompt_eval_ms=50.0, output_tokens=10, generation_ms=20.0
    )
    b = GenerationStats(
        prompt_tokens=200, prompt_eval_ms=150.0, output_tokens=30, generation_ms=80.0
    )
    combined = combine_stats([a, b])
    assert combined == GenerationStats(
        prompt_tokens=300, prompt_eval_ms=200.0, output_tokens=40, generation_ms=100.0
    )


def test_combine_stats_empty_returns_none():
    assert combine_stats([]) is None
