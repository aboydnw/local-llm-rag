from types import SimpleNamespace

from rag_lab.studio import models
from rag_lab.studio.models import PullEvent


def test_is_installed_matches_tagless_against_latest():
    assert models.is_installed("nomic-embed-text", ["nomic-embed-text:latest"])


def test_is_installed_matches_exact_tag():
    assert models.is_installed("qwen2.5:3b", ["qwen2.5:3b"])


def test_is_installed_false_when_absent():
    assert not models.is_installed("llama3.2:3b", ["qwen2.5:3b"])


def test_is_installed_false_for_empty_list():
    assert not models.is_installed("qwen2.5:3b", [])


def test_pull_event_is_frozen():
    ev = PullEvent(status="downloading", fraction=0.5)
    assert ev.status == "downloading"
    assert ev.fraction == 0.5


def test_installed_models_reads_names_from_client():
    client = SimpleNamespace(
        list=lambda: SimpleNamespace(
            models=[
                SimpleNamespace(model="qwen2.5:3b"),
                SimpleNamespace(model="nomic-embed-text:latest"),
            ]
        )
    )
    assert models.installed_models(client) == ["qwen2.5:3b", "nomic-embed-text:latest"]


def test_pull_progress_normalizes_fractions():
    chunks = [
        SimpleNamespace(status="pulling manifest", completed=None, total=None),
        SimpleNamespace(status="downloading", completed=50, total=100),
        SimpleNamespace(status="downloading", completed=100, total=100),
    ]
    client = SimpleNamespace(pull=lambda model, stream: iter(chunks))
    events = list(models.pull_progress("qwen2.5:3b", client))
    assert events[0] == PullEvent("pulling manifest", None)
    assert events[1] == PullEvent("downloading", 0.5)
    assert events[2] == PullEvent("downloading", 1.0)


def test_pull_progress_handles_zero_total():
    chunks = [SimpleNamespace(status="verifying", completed=0, total=0)]
    client = SimpleNamespace(pull=lambda model, stream: iter(chunks))
    events = list(models.pull_progress("x", client))
    assert events[0] == PullEvent("verifying", None)
