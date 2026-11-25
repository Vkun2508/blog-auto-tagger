"""Session 4: choose each model's score threshold and tag limit on the validation split,
then pick the default model for the app.

Run (about 2 minutes; needs 03_train.py first):   python scripts/04_tune_thresholds.py

Steps for every model:
  1. Try every global threshold 0.05..0.95 (step 0.05) with max_tags 1..5; keep the best val micro-F1.
  2. For tags with >= 20 validation posts, try a threshold of their own (best F1 for that tag).
     Keep these per-tag thresholds only if they raise overall val micro-F1.
  3. Record the precision / recall / review-rate trade-off at every global threshold.

Writes models/settings.json, reports/threshold_tradeoffs.csv and charts in reports/figures/.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))          # lets this script import the tagger package

from tagger.charts import COLORS, MODEL_COLORS, new_figure, save_figure  # noqa: E402
from tagger.dataset import label_matrix, load_posts, predict_split  # noqa: E402
from tagger.engine import encoder_library_works  # noqa: E402
from tagger.models import MODEL_FILES, MODEL_LABELS, MODEL_NAMES, load_model  # noqa: E402
from tagger.suggest import MAX_CANDIDATES, apply_settings  # noqa: E402
from tagger.taxonomy import load_taxonomy  # noqa: E402
from tagger.tuning import THRESHOLDS, micro_f1, tune_global  # noqa: E402

MODELS_DIR = PROJECT / "models"
REPORTS = PROJECT / "reports"
FIGURES = REPORTS / "figures"

MIN_WORDS = 30
MIN_VAL_POSITIVES = 20          # fewer positives than this and a per-tag threshold would just fit noise
SPEED_TOLERANCE = 0.01          # a faster model within this much micro-F1 of the best is preferred
SPEED_SAMPLE = 50


def tune_per_tag(scores, n_words, Y, tags, settings):
    """For each frequent tag, the threshold that maximises that tag's own F1 (others stay global)."""
    per_tag = {}
    for j, tag in enumerate(tags):
        if Y[:, j].sum() < MIN_VAL_POSITIVES:
            continue
        best_f1, best_threshold = -1, settings["threshold"]
        for threshold in THRESHOLDS:
            trial = {**settings, "per_tag_thresholds": {tag: threshold}}
            Y_pred, _ = apply_settings(scores, tags, trial, n_words, MIN_WORDS)
            f1 = f1_score(Y[:, j], Y_pred[:, j], zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold
        per_tag[tag] = best_threshold
    return per_tag


def tradeoff_rows(name, scores, n_words, Y, tags, max_tags):
    """Quality and workload at every global threshold (with the model's best max_tags)."""
    rows = []
    for threshold in THRESHOLDS:
        Y_pred, reasons = apply_settings(scores, tags, {"threshold": threshold, "max_tags": max_tags}, n_words, MIN_WORDS)
        p, r, f, _ = precision_recall_fscore_support(Y, Y_pred, average="micro", zero_division=0)
        rows.append({"model": name, "threshold": threshold, "max_tags": max_tags,
                     "precision": round(p, 4), "recall": round(r, 4), "micro_f1": round(f, 4),
                     "wrong_per_post": round(((Y_pred == 1) & (Y == 0)).sum() / len(Y), 3),
                     "missed_per_post": round(((Y_pred == 0) & (Y == 1)).sum() / len(Y), 3),
                     "review_pct": round(100 * np.mean([reason is not None for reason in reasons]), 1)})
    return rows


def ms_per_post(model, posts):
    """Median time to score one post on its own (after one warm-up call)."""
    model.predict_scores([posts[0]["title"]], [posts[0]["body"]])
    times = []
    for post in posts[:SPEED_SAMPLE]:
        start = time.perf_counter()
        model.predict_scores([post["title"]], [post["body"]])
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def draw_tradeoff(name, table):
    fig, ax = new_figure()
    for column, color in zip(["precision", "recall", "micro_f1"], COLORS):
        ax.plot(table["threshold"], table[column], color=color, linewidth=2, label=column.replace("_", "-"))
    ax.set_xlabel("Global threshold")
    ax.set_ylabel("Validation score")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    save_figure(fig, FIGURES / f"threshold_tradeoff_{name}.png", f"{MODEL_LABELS[name]}: threshold trade-off")


def draw_review_rate(table):
    fig, ax = new_figure()
    for name, rows in table.groupby("model", sort=False):
        ax.plot(rows["threshold"], rows["review_pct"], color=MODEL_COLORS[name], linewidth=2, label=MODEL_LABELS[name])
    ax.set_xlabel("Global threshold")
    ax.set_ylabel("Posts routed to an editor for review (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    save_figure(fig, FIGURES / "review_rate_vs_threshold.png", "Review workload vs threshold")


def main():
    total_start = time.time()
    tags = load_taxonomy().tags
    val = load_posts("val")
    Y = label_matrix(val, tags)
    n_words = [p["n_words"] for p in val]

    names = [n for n in MODEL_NAMES if all((MODELS_DIR / f).exists() for f in MODEL_FILES[n])]
    if not encoder_library_works():
        names = [n for n in names if n not in ("semantic", "hybrid")]
        print("sentence-transformers is not installed: skipping semantic and hybrid")

    settings = {"default_model": None, "min_words": MIN_WORDS, "max_candidates": MAX_CANDIDATES, "models": {}}
    all_tradeoffs, results = [], {}
    for name in names:
        start = time.time()
        model = load_model(name, MODELS_DIR)
        scores = predict_split(model, val, "val")

        f1_global, threshold, max_tags = tune_global(scores, n_words, Y, tags, MIN_WORDS)
        model_settings = {"threshold": threshold, "max_tags": max_tags, "per_tag_thresholds": {}}
        per_tag = tune_per_tag(scores, n_words, Y, tags, model_settings)
        Y_pred, _ = apply_settings(scores, tags, {**model_settings, "per_tag_thresholds": per_tag}, n_words, MIN_WORDS)
        f1_per_tag = micro_f1(Y, Y_pred)
        if f1_per_tag > f1_global:            # keep per-tag thresholds only if they actually help
            model_settings["per_tag_thresholds"] = per_tag
        model_settings["val_micro_f1_global"] = round(f1_global, 4)
        model_settings["val_micro_f1_per_tag"] = round(f1_per_tag, 4)
        settings["models"][name] = model_settings

        speed = ms_per_post(model, val)
        results[name] = {"f1": max(f1_global, f1_per_tag), "ms": speed}
        rows = tradeoff_rows(name, scores, n_words, Y, tags, max_tags)
        all_tradeoffs += rows
        draw_tradeoff(name, pd.DataFrame(rows))
        kept = "kept" if model_settings["per_tag_thresholds"] else "not kept"
        print(f"{name}: threshold {threshold}, max_tags {max_tags}, micro-F1 {f1_global:.3f}; "
              f"per-tag {f1_per_tag:.3f} ({kept}); {speed:.1f} ms/post ({time.time() - start:.0f}s)")

    # Default model: the best val micro-F1, unless a faster model is within 0.01 of it.
    best_name = max(results, key=lambda n: results[n]["f1"])
    close_enough = [n for n in results if results[n]["f1"] >= results[best_name]["f1"] - SPEED_TOLERANCE]
    default = min(close_enough, key=lambda n: results[n]["ms"])
    numbers = "; ".join(f"{n}: micro-F1 {results[n]['f1']:.3f}, {results[n]['ms']:.1f} ms/post" for n in results)
    settings["default_model"] = default
    settings["selection_note"] = (
        f"Rule: pick the model with the best validation micro-F1 ({best_name}); if another model is within "
        f"{SPEED_TOLERANCE} micro-F1 of it and faster per post, pick the fastest such model. "
        f"Chosen: {default}. Validation numbers: {numbers}.")

    (MODELS_DIR / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    tradeoffs = pd.DataFrame(all_tradeoffs)
    tradeoffs.to_csv(REPORTS / "threshold_tradeoffs.csv", index=False)
    draw_review_rate(tradeoffs)
    print(f"\nDefault model: {default}\n{settings['selection_note']}")
    print(f"Wrote models/settings.json and reports/threshold_tradeoffs.csv ({time.time() - total_start:.0f}s)")


if __name__ == "__main__":
    main()
