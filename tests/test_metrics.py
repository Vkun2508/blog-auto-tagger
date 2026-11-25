"""A 3-post, 4-tag example worked out by hand.

Tags:      A  B  C  D
true p0:   1  1  0  0      pred p0: 1 0 0 0
true p1:   0  1  0  0      pred p1: 0 1 1 0
true p2:   1  0  1  1      pred p2: 1 0 0 1

Per tag: A TP2 -> P1 R1 F1 1 | B TP1 FN1 -> P1 R.5 F1 2/3 | C FP1 FN1 -> 0 | D TP1 -> 1
Micro: TP=4 FP=1 FN=2 -> P=4/5, R=4/6, F1=2*4/(2*4+1+2)=8/11
Macro: P=(1+1+0+1)/4=0.75, R=(1+.5+0+1)/4=0.625, F1=(1+2/3+0+1)/4=2/3
Top 3 by score: p0 A,B,D (2 hits) | p1 B,C,A (1 hit) | p2 A,D,B (2 hits)
P@3 = (2/3 + 1/3 + 2/3)/3 = 5/9      R@3 = (2/2 + 1/1 + 2/3)/3 = 8/9
"""

import pytest

from tagger.metrics import evaluate, per_tag_report

Y_TRUE = [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 1, 1]]
Y_PRED = [[1, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]]
SCORES = [[0.9, 0.4, 0.1, 0.3], [0.2, 0.8, 0.6, 0.1], [0.7, 0.5, 0.2, 0.6]]


def test_toy_example_exact_values():
    result = evaluate(Y_TRUE, Y_PRED, SCORES)
    assert result["micro_precision"] == pytest.approx(4 / 5)
    assert result["micro_recall"] == pytest.approx(4 / 6)
    assert result["micro_f1"] == pytest.approx(8 / 11)
    assert result["macro_precision"] == pytest.approx(0.75)
    assert result["macro_recall"] == pytest.approx(0.625)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["p_at_3"] == pytest.approx(5 / 9)
    assert result["r_at_3"] == pytest.approx(8 / 9)


def test_per_tag_report():
    report = per_tag_report(Y_TRUE, Y_PRED, ["A", "B", "C", "D"])
    assert list(report.columns) == ["tag", "support", "precision", "recall", "f1"]
    assert report["support"].tolist() == [2, 2, 1, 1]
    assert report["f1"].tolist() == pytest.approx([1, 2 / 3, 0, 1])
