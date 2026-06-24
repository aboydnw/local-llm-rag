from pathlib import Path

import pytest

from rag_lab.loaders.github_issues import GitHubIssuesLoader, _load_paginated_array


def _fake_fetch(repo, number):
    issue = {
        "title": "What is titiler?",
        "body": "A dynamic tile server.",
        "html_url": f"https://github.com/{repo}/issues/{number}",
        "state": "open",
        "labels": [{"name": "question"}, {"name": "docs"}],
    }
    comments = [
        {"user": {"login": "alice"}, "body": "It wraps rio-tiler."},
        {"user": {"login": "bob"}, "body": "See the README."},
    ]
    return issue, comments


def test_issue_document_includes_title_body_and_comments():
    loader = GitHubIssuesLoader("developmentseed/titiler", [42], fetch_fn=_fake_fetch)
    (doc,) = list(loader.load())
    assert "What is titiler?" in doc.text
    assert "A dynamic tile server." in doc.text
    assert "It wraps rio-tiler." in doc.text
    assert "bob" in doc.text


def test_issue_document_metadata():
    loader = GitHubIssuesLoader("developmentseed/titiler", [42], fetch_fn=_fake_fetch)
    (doc,) = list(loader.load())
    assert doc.path == Path("developmentseed/titiler#42")
    assert doc.metadata["source"] == "developmentseed/titiler#42"
    assert doc.metadata["number"] == "42"
    assert doc.metadata["url"] == "https://github.com/developmentseed/titiler/issues/42"
    assert doc.metadata["state"] == "open"
    assert doc.metadata["labels"] == "question, docs"


def test_loader_yields_one_document_per_number():
    loader = GitHubIssuesLoader("o/r", [1, 2, 3], fetch_fn=_fake_fetch)
    assert len(list(loader.load())) == 3


def test_load_paginated_array_single_merged():
    assert _load_paginated_array('[{"id": 1}, {"id": 2}]') == [{"id": 1}, {"id": 2}]


def test_load_paginated_array_concatenated_pages():
    assert _load_paginated_array('[{"id": 1}][{"id": 2}]') == [{"id": 1}, {"id": 2}]


def test_load_paginated_array_handles_empty():
    assert _load_paginated_array("[]") == []
    assert _load_paginated_array("") == []


def test_pull_request_is_rejected():
    def fetch(repo, number):
        issue = {
            "title": "A PR",
            "body": "body",
            "html_url": "url",
            "state": "open",
            "labels": [],
            "pull_request": {"url": "https://api.github.com/.../pulls/5"},
        }
        return issue, []

    loader = GitHubIssuesLoader("o/r", [5], fetch_fn=fetch)
    with pytest.raises(ValueError):
        list(loader.load())
