"""Session 3: train the four models on the training split and compare settings on validation.

Run (about 4 minutes on a laptop CPU when data/cache exists; the first run also downloads the
90 MB MiniLM encoder and encodes every train/val post once, which adds about 10 minutes):
    python scripts/03_train.py

How configs are compared: each config's scores are judged at their own best global threshold
(0.05..0.95) and tag limit (1..5) on validation, the same search 04_tune_thresholds.py uses.
A single fixed threshold would be unfair, because models put scores on different scales.
The micro-F1 at a fixed 0.5 threshold is kept as an extra column. Test is never used here.

Writes:
    models/keyword.json, tfidf.joblib, semantic.joblib, hybrid.json   the chosen models
    reports/validation_tfidf_grid.csv, validation_semantic_pooling.csv, validation_hybrid.csv
    reports/validation_experiments.md and figures/title_weight.png, figures/pooling.png
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))          # lets this script import the tagger package

from tagger.charts import COLORS, new_figure, save_figure  # noqa: E402
from tagger.dataset import label_matrix, load_posts  # noqa: E402
from tagger.metrics import evaluate  # noqa: E402
from tagger.models.hybrid import HybridModel  # noqa: E402
from tagger.models.keyword import KeywordModel  # noqa: E402
from tagger.models.semantic import POOLINGS, SemanticModel, cached_embeddings  # noqa: E402
from tagger.models.tfidf import TfidfModel  # noqa: E402
from tagger.reporting import markdown_table  # noqa: E402
from tagger.suggest import apply_settings  # noqa: E402
from tagger.taxonomy import load_taxonomy  # noqa: E402
from tagger.tuning import tune_global  # noqa: E402

MODELS_DIR = PROJECT / "models"
REPORTS = PROJECT / "reports"
FIGURES = REPORTS / "figures"

FIXED_SETTINGS = {"threshold": 0.5, "max_tags": 5}     # only for the reference column
MIN_WORDS = 30
TITLE_WEIGHTS = [0, 1, 2, 3]
C_VALUES = [4, 16, 64]
HYBRID_WEIGHTS = [0.3, 0.5, 0.7]
Y_LABEL = "validation micro-F1 at best threshold"
np.random.seed(42)                         # nothing here is random, but fix the seed anyway


def score_row(scores, posts, Y, tags):
    """Validation metrics for a score matrix, judged at its own best threshold and tag limit."""
    n_words = [p["n_words"] for p in posts]
    tuned_f1, threshold, max_tags = tune_global(scores, n_words, Y, tags, MIN_WORDS)
    Y_best, _ = apply_settings(scores, tags, {"threshold": threshold, "max_tags": max_tags}, n_words, MIN_WORDS)
    best = evaluate(Y, Y_best, scores)
    Y_fixed, _ = apply_settings(scores, tags, FIXED_SETTINGS, n_words, MIN_WORDS)
    fixed = evaluate(Y, Y_fixed, scores)
    return {"tuned_micro_f1": round(tuned_f1, 4), "best_threshold": threshold, "best_max_tags": max_tags,
            "macro_f1": round(best["macro_f1"], 4), "micro_precision": round(best["micro_precision"], 4),
            "micro_recall": round(best["micro_recall"], 4), "p_at_3": round(best["p_at_3"], 4),
            "micro_f1_at_0.5": round(fixed["micro_f1"], 4)}


def draw_title_weight(grid):
    fig, ax = new_figure()
    for C, color in zip(C_VALUES, COLORS):
        rows = grid[grid["C"] == C]
        ax.plot(rows["title_weight"], rows["tuned_micro_f1"], color=color, linewidth=2, marker="o", markersize=7,
                label=f"C = {C}")
    ax.set_xlabel("Title weight (how much louder title words are than body words)")
    ax.set_ylabel(Y_LABEL)
    ax.set_xticks(TITLE_WEIGHTS)
    ax.legend(frameon=False)
    save_figure(fig, FIGURES / "title_weight.png", "TF-IDF: effect of the title weight")


def draw_pooling(table):
    fig, ax = new_figure(9, 5)
    width = 0.26
    x = np.arange(len(POOLINGS))
    for k, (C, color) in enumerate(zip(C_VALUES, COLORS)):
        values = [table[(table["pooling"] == p) & (table["C"] == C)]["tuned_micro_f1"].iloc[0] for p in POOLINGS]
        ax.bar(x + (k - 1) * width, values, width, color=color, label=f"C = {C}", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(POOLINGS)
    ax.set_ylabel(Y_LABEL)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, FIGURES / "pooling.png", "Semantic model: how title and body vectors are combined")


def main():
    total_start = time.time()
    taxonomy = load_taxonomy()
    tags = taxonomy.tags
    train, val = load_posts("train"), load_posts("val")
    Y_train, Y_val = label_matrix(train, tags), label_matrix(val, tags)
    train_titles, train_bodies = [p["title"] for p in train], [p["body"] for p in train]
    val_titles, val_bodies = [p["title"] for p in val], [p["body"] for p in val]
    print(f"Train {len(train)} posts, validation {len(val)} posts, {len(tags)} tags\n")
    REPORTS.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # (a) Keyword baseline: nothing to train.
    start = time.time()
    keyword_model = KeywordModel(taxonomy)
    keyword_result = score_row(keyword_model.predict_scores(val_titles, val_bodies), val, Y_val, tags)
    print(f"(a) keyword baseline: tuned val micro-F1 {keyword_result['tuned_micro_f1']:.3f} "
          f"({time.time() - start:.0f}s)")

    # (b) TF-IDF grid: title weight x C.
    print("(b) TF-IDF grid")
    start = time.time()
    grid_rows, best_tfidf, best_tfidf_scores, best_f1 = [], None, None, -1
    for title_weight in TITLE_WEIGHTS:
        for C in C_VALUES:
            fit_start = time.time()
            model = TfidfModel(title_weight, C).fit(train_titles, train_bodies, Y_train)
            scores = model.predict_scores(val_titles, val_bodies)
            row = {"title_weight": title_weight, "C": C, **score_row(scores, val, Y_val, tags),
                   "seconds": round(time.time() - fit_start, 1)}
            grid_rows.append(row)
            print(f"   title_weight={title_weight} C={C}: tuned micro-F1 {row['tuned_micro_f1']:.3f} "
                  f"(threshold {row['best_threshold']}, {row['seconds']}s)")
            if row["tuned_micro_f1"] > best_f1:          # keep the best model seen so far
                best_tfidf, best_tfidf_scores, best_f1 = model, scores, row["tuned_micro_f1"]
    tfidf_grid = pd.DataFrame(grid_rows)
    tfidf_grid.to_csv(REPORTS / "validation_tfidf_grid.csv", index=False)
    draw_title_weight(tfidf_grid)
    # to_dict("records") keeps each column's type, so max_tags stays a whole number in the report.
    best_tfidf_row = tfidf_grid.loc[[tfidf_grid["tuned_micro_f1"].idxmax()]].to_dict("records")[0]
    print(f"   best: title_weight={best_tfidf.title_weight} C={best_tfidf.C} ({time.time() - start:.0f}s total)")

    # (c) Semantic: encode once (cached), then compare poolings x C on the same vectors.
    print("(c) Semantic model")
    start = time.time()
    train_embeddings = cached_embeddings("train", [p["id"] for p in train], train_titles, train_bodies)
    val_embeddings = cached_embeddings("val", [p["id"] for p in val], val_titles, val_bodies)
    print(f"   embeddings ready ({time.time() - start:.0f}s)")
    pooling_rows, best_semantic, best_semantic_scores, best_f1 = [], None, None, -1
    for pooling in POOLINGS:
        for C in C_VALUES:
            fit_start = time.time()
            model = SemanticModel(pooling, C).fit(train_titles, train_bodies, Y_train, embeddings=train_embeddings)
            scores = model.predict_scores(val_titles, val_bodies, embeddings=val_embeddings)
            row = {"pooling": pooling, "C": C, **score_row(scores, val, Y_val, tags),
                   "seconds": round(time.time() - fit_start, 1)}
            pooling_rows.append(row)
            print(f"   pooling={pooling} C={C}: tuned micro-F1 {row['tuned_micro_f1']:.3f} "
                  f"(threshold {row['best_threshold']}, {row['seconds']}s)")
            if row["tuned_micro_f1"] > best_f1:
                best_semantic, best_semantic_scores, best_f1 = model, scores, row["tuned_micro_f1"]
    pooling_table = pd.DataFrame(pooling_rows)
    pooling_table.to_csv(REPORTS / "validation_semantic_pooling.csv", index=False)
    draw_pooling(pooling_table)
    best_semantic_row = pooling_table.loc[[pooling_table["tuned_micro_f1"].idxmax()]].to_dict("records")[0]
    print(f"   best: pooling={best_semantic.pooling} C={best_semantic.C} ({time.time() - start:.0f}s total)")

    # (d) Hybrid: mix the two best models' scores.
    print("(d) Hybrid")
    hybrid_rows = []
    for weight in HYBRID_WEIGHTS:
        scores = weight * best_tfidf_scores + (1 - weight) * best_semantic_scores
        hybrid_rows.append({"weight_tfidf": weight, **score_row(scores, val, Y_val, tags)})
        print(f"   weight_tfidf={weight}: tuned micro-F1 {hybrid_rows[-1]['tuned_micro_f1']:.3f}")
    hybrid_table = pd.DataFrame(hybrid_rows)
    hybrid_table.to_csv(REPORTS / "validation_hybrid.csv", index=False)
    best_hybrid_row = hybrid_table.loc[[hybrid_table["tuned_micro_f1"].idxmax()]].to_dict("records")[0]
    best_weight = float(best_hybrid_row["weight_tfidf"])

    # Save the chosen configs. They were fitted on train only; val stays free for threshold tuning.
    keyword_model.save(MODELS_DIR)
    best_tfidf.save(MODELS_DIR)
    best_semantic.save(MODELS_DIR)
    HybridModel(best_weight, best_tfidf, best_semantic).save(MODELS_DIR)
    print(f"\nSaved models to {MODELS_DIR}")

    summary = pd.DataFrame([
        {"model": "keyword", "chosen config": "keywords from taxonomy (no training)", **keyword_result},
        {"model": "tfidf", "chosen config": f"title_weight={best_tfidf.title_weight}, C={best_tfidf.C}",
         **{k: best_tfidf_row[k] for k in keyword_result}},
        {"model": "semantic", "chosen config": f"pooling={best_semantic.pooling}, C={best_semantic.C}",
         **{k: best_semantic_row[k] for k in keyword_result}},
        {"model": "hybrid", "chosen config": f"weight_tfidf={best_weight}",
         **{k: best_hybrid_row[k] for k in keyword_result}},
    ])
    report = [
        "# Validation experiments", "",
        "Written by `scripts/03_train.py`. Models are trained on the train split and compared on the "
        "validation split; the test split is not used.", "",
        "**How configs are compared.** Different models (and different C values) put their scores on "
        "different scales: a strongly regularised model rarely scores above 0.5 even when it ranks the "
        "right tags first. Judging everything at one fixed threshold would therefore measure score scale, "
        "not quality. So each config is judged at its own best global threshold (0.05 to 0.95) and tag "
        "limit (1 to 5) on validation - the same search `04_tune_thresholds.py` uses - and the config "
        "with the highest `tuned_micro_f1` is chosen. `best_threshold` and `best_max_tags` show where "
        "that best was found; macro-F1, precision and recall are measured there too. P@3 ignores "
        "thresholds. `micro_f1_at_0.5` (fixed threshold 0.5, up to 5 tags) is kept only for reference.", "",
        "## Chosen configs", "", markdown_table(summary), "",
        "## TF-IDF grid (title weight x C)", "",
        "Title weight 0 means the title is ignored; higher values make title words count more. "
        "See `figures/title_weight.png`.", "", markdown_table(tfidf_grid), "",
        "## Semantic model (pooling x C)", "",
        "title_only uses only the title vector; body_mean averages the body chunk vectors; the title+ "
        "options join the title vector with the averaged (mean) or strongest (max) body values. "
        "See `figures/pooling.png`.", "", markdown_table(pooling_table), "",
        "## Hybrid (weight given to TF-IDF, using the best TF-IDF and semantic configs)", "",
        markdown_table(hybrid_table), "",
        f"Total time: {time.time() - total_start:.0f} seconds.", ""]
    (REPORTS / "validation_experiments.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote reports/validation_experiments.md. Total time {time.time() - total_start:.0f}s")


if __name__ == "__main__":
    main()
