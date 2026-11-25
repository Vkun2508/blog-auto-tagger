"""Smoke test: the Tag a draft page runs and shows tags / the review warning."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (ROOT / "models" / "settings.json").exists(), reason="models/settings.json is missing"
)


def run_example(example_key):
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180).run()
    assert not app.exception
    app.button(key=example_key).click().run()
    app.button(key="suggest").click().run()
    assert not app.exception
    return app


def test_normal_example_shows_tags():
    app = run_example("example_normal")
    assert any(box.value for box in app.checkbox)  # at least one tag is pre-checked
    assert not app.warning


def test_short_example_asks_for_review():
    app = run_example("example_short")
    assert app.warning  # the review banner
