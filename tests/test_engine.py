"""Checks on the trained engine (skipped until scripts 03 and 04 have run).

Only the fast models are used here, so no encoder is downloaded or loaded.
"""

import pytest

from tagger.engine import DEFAULT_MODELS_DIR, TaggingEngine

if not (DEFAULT_MODELS_DIR / "settings.json").exists():
    pytest.skip("models/settings.json not built yet", allow_module_level=True)

TITLE = "Shrinking a Docker image with multi-stage builds"
BODY = ("A Dockerfile with a multi-stage build copies only the compiled binary into a small base image. "
        "We pushed the image to our registry and deployed the containers to Kubernetes with Helm. ") * 3


@pytest.fixture(scope="module")
def engine():
    return TaggingEngine.load()


@pytest.mark.parametrize("model", ["keyword", "tfidf"])
def test_only_approved_tags(engine, model):
    if model not in engine.available_models:
        pytest.skip(f"{model} not trained")
    suggestion = engine.suggest(TITLE, BODY, model=model)
    approved = set(engine.taxonomy.tags)
    assert set(suggestion.tag_names()) <= approved
    assert all(c.tag in approved for c in suggestion.candidates)
    assert len(suggestion.candidates) <= 5
    assert len(suggestion.tags) <= engine.settings["models"][model]["max_tags"] <= 5


def test_empty_draft_needs_review(engine):
    suggestion = engine.suggest("", "   ", model="keyword")
    assert suggestion.needs_review and suggestion.review_reason == "empty"
    assert suggestion.tags == [] and suggestion.candidates == []


def test_snippet_comes_from_text(engine):
    suggestion = engine.suggest(TITLE, BODY, model="keyword")
    docker = [c for c in suggestion.candidates if c.tag == "Docker & Kubernetes"][0]
    # This sentence has the most Docker & Kubernetes keywords (containers, Kubernetes, Helm).
    assert docker.snippet == "We pushed the image to our registry and deployed the containers to Kubernetes with Helm."
