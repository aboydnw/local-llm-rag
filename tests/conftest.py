from pathlib import Path

import pytest


@pytest.fixture
def fixture_corpus() -> Path:
    return Path(__file__).parent / "fixtures" / "corpus"
