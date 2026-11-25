import pytest

from tagger.models.keyword import KeywordModel


@pytest.fixture(scope="module")
def model():
    return KeywordModel()


def score(model, tag, title, body=""):
    column = model.taxonomy.tags.index(tag)
    return float(model.predict_scores([title], [body])[0, column])


def test_capital_keyword_is_case_sensitive(model):
    assert score(model, "AI & LLMs", "How AI changes code review") > 0
    assert score(model, "AI & LLMs", "He said it was fine") == 0       # "ai" inside "said"
    assert score(model, "AI & LLMs", "ai is everywhere") == 0          # lowercase "ai" is not "AI"


def test_in_go_does_not_fire_on_go_to(model):
    assert score(model, "Go", "I want to go to the shop") == 0
    assert score(model, "Go", "A web server written in Go") > 0


def test_java_does_not_match_javascript(model):
    assert score(model, "Java", "JavaScript closures explained") == 0
    assert score(model, "Java", "Java streams explained") > 0


def test_score_formula(model):
    # One distinct keyword in the title and none in the body: 1 - exp(-3/4)
    assert score(model, "Rust", "Learning Rust") == pytest.approx(1 - 2.718281828 ** -0.75, abs=1e-4)
