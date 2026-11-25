"""Session 5: the final, one-time evaluation on the test split (newest posts, never used before).

Run (about 3 minutes; needs 03 and 04 first):   python scripts/05_evaluate.py

Every model uses its tuned settings from models/settings.json and the same suggest_from_scores
logic the app uses. Reports contain titles and URLs only, never post body text.

Writes in reports/: test_results.csv/.md, test_by_frequency.csv, test_per_tag.csv, short_drafts.csv,
similar_tag_confusions.csv, test_predictions.csv, error_examples.md and figures/model_comparison.png,
figures/test_by_frequency.png.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))          # lets this script import the tagger package

from tagger.charts import MODEL_COLORS, new_figure, save_figure  # noqa: E402
from tagger.cleaning import word_count  # noqa: E402
from tagger.dataset import label_matrix, load_posts, predict_split  # noqa: E402
from tagger.engine import TaggingEngine  # noqa: E402
from tagger.metrics import evaluate, per_tag_report  # noqa: E402
from tagger.reporting import markdown_table  # noqa: E402
from tagger.suggest import apply_settings, suggest_from_scores  # noqa: E402

RAW_DIR = PROJECT / "data" / "raw"
REPORTS = PROJECT / "reports"
FIGURES = REPORTS / "figures"

TIMING_CALLS = 200
SHORT_DRAFTS = {"full": None, "first_40_words": 40, "first_10_words": 10}
BUCKETS = ["common", "medium", "rare"]
SIMILAR_PAIRS = [("AI & LLMs", "Machine Learning"), ("Machine Learning", "Data Science"),
                 ("Docker & Kubernetes", "DevOps"), ("Cloud", "DevOps"), ("JavaScript", "TypeScript"),
                 ("React", "JavaScript"), ("React", "Frontend"), ("API Development", "Software Architecture"),
                 ("Security", "DevOps")]
EXAMPLES_PER_CATEGORY = 3


def tag_set(row, tags):
    return {tags[j] for j in np.flatnonzero(row)}


def time_engine(engine, name, raw_posts):
    """Median milliseconds for one engine.suggest call (cleaning included), after a warm-up call."""
    engine.suggest(raw_posts[0]["title"], raw_posts[0]["body_markdown"], model=name)
    times = []
    for k in range(TIMING_CALLS):
        post = raw_posts[k % len(raw_posts)]
        start = time.perf_counter()
        engine.suggest(post["title"], post["body_markdown"], model=name)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def frequency_buckets(train, tags):
    """Split tags into thirds by how often they appear in training: common, medium, rare."""
    counts = label_matrix(train, tags).sum(axis=0)
    by_count = [tags[j] for j in np.argsort(-counts, kind="stable")]
    return {tag: bucket for bucket, part in zip(BUCKETS, np.array_split(by_count, 3)) for tag in part}, \
        dict(zip(tags, counts.tolist()))


def describe(post, true_tags, suggested, scores_row, tags):
    """A few Markdown lines about one post's result (title and URL only)."""
    score_of = dict(zip(tags, scores_row))
    shown = ", ".join(f"{t} ({score_of[t]:.2f})" for t in sorted(suggested, key=lambda t: -score_of[t]))
    return [f"- **{post['title']}** ({post['url']})",
            f"  - true tags: {', '.join(sorted(true_tags))}",
            f"  - suggested: {shown or '(none)'}",
            f"  - false positives: {', '.join(sorted(suggested - true_tags)) or 'none'}; "
            f"missed: {', '.join(sorted(true_tags - suggested)) or 'none'}"]


def draw_comparison(results):
    metrics = ["micro_f1", "macro_f1", "p_at_3", "r_at_3"]
    fig, ax = new_figure(9, 5)
    width = 0.8 / len(results)
    x = np.arange(len(metrics))
    for k, row in enumerate(results.itertuples()):
        ax.bar(x + (k - (len(results) - 1) / 2) * width, [getattr(row, m) for m in metrics], width,
               color=MODEL_COLORS[row.model], label=row.label, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["micro-F1", "macro-F1", "P@3", "R@3"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test score")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    save_figure(fig, FIGURES / "model_comparison.png", "Test results by model")


def draw_frequency(table, names, labels):
    fig, ax = new_figure()
    width = 0.8 / len(names)
    x = np.arange(len(BUCKETS))
    for k, name in enumerate(names):
        values = [table[(table["model"] == name) & (table["bucket"] == b)]["macro_f1"].iloc[0] for b in BUCKETS]
        ax.bar(x + (k - (len(names) - 1) / 2) * width, values, width, color=MODEL_COLORS[name],
               label=labels[name], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["Common tags", "Medium tags", "Rare tags"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test macro-F1")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, FIGURES / "test_by_frequency.png", "Test macro-F1 by how common a tag is in training")


def main():
    total_start = time.time()
    engine = TaggingEngine.load()
    tags = engine.taxonomy.tags
    min_words = engine.settings["min_words"]
    names = engine.available_models
    labels = {n: engine.model_label(n) for n in names}
    test, train = load_posts("test"), load_posts("train")
    Y = label_matrix(test, tags)
    n_words = [p["n_words"] for p in test]
    raw_posts = [json.loads((RAW_DIR / f"{p['id']}.json").read_text(encoding="utf-8")) for p in test]
    bucket_of, train_counts = frequency_buckets(train, tags)
    print(f"Test split: {len(test)} posts. Models: {', '.join(names)}\n")

    result_rows, bucket_rows, short_rows, outputs = [], [], [], {}
    for name in names:
        start = time.time()
        model = engine.get_model(name)
        model_settings = engine.settings["models"][name]
        scores = predict_split(model, test, "test")
        Y_pred, reasons = apply_settings(scores, tags, model_settings, n_words, min_words)
        outputs[name] = {"scores": scores, "Y_pred": Y_pred, "reasons": reasons}

        result = evaluate(Y, Y_pred, scores)
        auto = np.array([r is None for r in reasons])
        auto_f1 = f1_score(Y[auto], Y_pred[auto], average="micro", zero_division=0) if auto.any() else 0.0
        ms = time_engine(engine, name, raw_posts)
        result_rows.append({"model": name, "label": labels[name], **{k: round(v, 4) for k, v in result.items()},
                            "review_coverage_pct": round(100 * auto.mean(), 1),
                            "auto_micro_f1": round(auto_f1, 4), "ms_per_post": round(ms, 1)})

        report = per_tag_report(Y, Y_pred, tags)
        report["bucket"] = report["tag"].map(bucket_of)
        outputs[name]["per_tag"] = report
        for bucket in BUCKETS:
            bucket_rows.append({"model": name, "bucket": bucket,
                                "tags": "; ".join(report[report["bucket"] == bucket]["tag"]),
                                "macro_f1": round(report[report["bucket"] == bucket]["f1"].mean(), 4)})

        # Short drafts: what if the editor has written only the title and the first few words?
        for draft, limit in SHORT_DRAFTS.items():
            if limit is None:
                draft_scores, draft_words = scores, n_words
            else:
                bodies = [" ".join(p["body"].split()[:limit]) for p in test]
                draft_scores = model.predict_scores([p["title"] for p in test], bodies)
                draft_words = [word_count(p["title"]) + word_count(b) for p, b in zip(test, bodies)]
            draft_pred, draft_reasons = apply_settings(draft_scores, tags, model_settings, draft_words, min_words)
            draft_result = evaluate(Y, draft_pred, draft_scores)
            short_rows.append({"model": name, "draft": draft, "micro_f1": round(draft_result["micro_f1"], 4),
                               "macro_f1": round(draft_result["macro_f1"], 4),
                               "p_at_3": round(draft_result["p_at_3"], 4),
                               "review_pct": round(100 * np.mean([r is not None for r in draft_reasons]), 1)})
            if draft == "first_40_words":
                outputs[name]["Y_pred_40"] = draft_pred
                outputs[name]["scores_40"] = draft_scores
        print(f"{name}: micro-F1 {result['micro_f1']:.3f}, macro-F1 {result['macro_f1']:.3f}, "
              f"P@3 {result['p_at_3']:.3f}, coverage {100 * auto.mean():.0f}%, {ms:.1f} ms/post "
              f"({time.time() - start:.0f}s)")

    results = pd.DataFrame(result_rows)
    results.to_csv(REPORTS / "test_results.csv", index=False)
    draw_comparison(results)
    buckets = pd.DataFrame(bucket_rows)
    buckets.to_csv(REPORTS / "test_by_frequency.csv", index=False)
    draw_frequency(buckets, names, labels)
    pd.DataFrame(short_rows).to_csv(REPORTS / "short_drafts.csv", index=False)

    # Everything below is about the default model.
    default = engine.default_model
    out = outputs[default]
    per_tag = out["per_tag"]
    per_tag["train_count"] = per_tag["tag"].map(train_counts)
    per_tag.round(4).to_csv(REPORTS / "test_per_tag.csv", index=False)

    true_sets = [tag_set(row, tags) for row in Y]
    pred_sets = [tag_set(row, tags) for row in out["Y_pred"]]
    confusion_rows = []
    for a, b in SIMILAR_PAIRS:
        for suggested, true in [(a, b), (b, a)]:
            count = sum(1 for t, s in zip(true_sets, pred_sets) if suggested in s and suggested not in t and true in t)
            confusion_rows.append({"suggested_tag": suggested, "true_tag": true, "posts": count})
    pd.DataFrame(confusion_rows).to_csv(REPORTS / "similar_tag_confusions.csv", index=False)

    prediction_rows = []
    model_settings = engine.settings["models"][default]
    for i, post in enumerate(test):
        _, candidates, needs_review, reason = suggest_from_scores(out["scores"][i], tags, model_settings,
                                                                  n_words[i], min_words)
        prediction_rows.append({"id": post["id"], "title": post["title"], "url": post["url"], "n_words": n_words[i],
                                "true_tags": "; ".join(post["tags"]),
                                "suggested_tags": "; ".join(t for t in tags if t in pred_sets[i]),
                                "candidate_scores": "; ".join(f"{c.tag}: {c.score:.3f}" for c in candidates),
                                "needs_review": needs_review, "reason": reason or ""})
    pd.DataFrame(prediction_rows).to_csv(REPORTS / "test_predictions.csv", index=False)

    write_error_examples(test, tags, true_sets, pred_sets, out, bucket_of, default, labels)
    write_results_md(results, buckets, pd.DataFrame(short_rows), engine)
    print(f"\nWrote test reports to {REPORTS} ({time.time() - total_start:.0f}s)")


def write_error_examples(test, tags, true_sets, pred_sets, out, bucket_of, default, labels):
    scores = out["scores"]
    pred_40 = [tag_set(row, tags) for row in out["Y_pred_40"]]
    n_words = np.array([p["n_words"] for p in test])
    long_cutoff = np.percentile(n_words, 90)
    pairs = SIMILAR_PAIRS + [(b, a) for a, b in SIMILAR_PAIRS]      # both directions

    def is_similar_mix(i):
        return any(a in pred_sets[i] and a not in true_sets[i] and b in true_sets[i] for a, b in pairs)

    categories = [
        ("Similar tags confused", "The model suggested one tag of a similar pair while the post's true tag "
         "was the other one.", [i for i in range(len(test)) if is_similar_mix(i)]),
        ("Short drafts", "Exactly right on the full post, but wrong when given only the title and the first "
         "40 words.", [i for i in range(len(test)) if true_sets[i] == pred_sets[i] != pred_40[i]]),
        ("Niche / rare tags", "A tag from the rare third (fewest training posts) was missed.",
         [i for i in range(len(test)) if any(bucket_of[t] == "rare" for t in true_sets[i] - pred_sets[i])]),
        ("Long multi-topic posts", f"Among the longest 10% of posts (>= {long_cutoff:.0f} words) with 3 or more "
         "true tags.", [i for i in range(len(test)) if n_words[i] >= long_cutoff and len(true_sets[i]) >= 3]),
    ]
    lines = ["# Error examples", "",
             f"Model: {labels[default]} (`{default}`), test split, tuned settings. Titles and URLs only.", ""]
    for heading, explanation, found in categories:
        lines += [f"## {heading}", "", f"{explanation} Found in {len(found)} test posts. First examples:", ""]
        for i in found[:EXAMPLES_PER_CATEGORY]:
            lines += describe(test[i], true_sets[i], pred_sets[i], scores[i], tags)
            if heading == "Short drafts":
                shown = ", ".join(sorted(pred_40[i])) or "(none)"
                lines.append(f"  - suggested from title + first 40 words: {shown}")
        lines.append("")

    false_positives = [(float(scores[i, j]), i, tags[j]) for i in range(len(test)) for j in range(len(tags))
                       if out["Y_pred"][i, j] == 1 and tags[j] not in true_sets[i]]
    lines += ["## 5 highest-scoring false positives", "", "| Score | Suggested tag | Post | True tags |",
              "|---|---|---|---|"]
    for score, i, tag in sorted(false_positives, reverse=True)[:5]:
        lines.append(f"| {score:.3f} | {tag} | [{test[i]['title']}]({test[i]['url']}) | {', '.join(sorted(true_sets[i]))} |")

    missed = pd.DataFrame([{"tag": t, "missed": sum(1 for i in range(len(test)) if t in true_sets[i] - pred_sets[i]),
                            "test_posts": sum(1 for s in true_sets if t in s)} for t in tags])
    missed["miss_rate"] = (missed["missed"] / missed["test_posts"].clip(lower=1)).round(3)
    lines += ["", "## 5 most-missed true tags", "",
              markdown_table(missed.sort_values(["missed", "tag"], ascending=[False, True]).head(5)), ""]
    (REPORTS / "error_examples.md").write_text("\n".join(lines), encoding="utf-8")


def write_results_md(results, buckets, short, engine):
    main_columns = ["model", "label", "micro_precision", "micro_recall", "micro_f1", "macro_f1",
                    "p_at_3", "r_at_3", "review_coverage_pct", "auto_micro_f1", "ms_per_post"]
    bucket_table = buckets.pivot(index="model", columns="bucket", values="macro_f1")[BUCKETS].reset_index()
    lines = ["# Test results", "",
             "Written by `scripts/05_evaluate.py`: the first and only use of the test split (the newest "
             f"{len(load_posts('test'))} posts). Each model uses its tuned threshold and tag limit. ", "",
             "- review_coverage_pct: posts NOT sent to an editor for review (higher = less manual work)",
             "- auto_micro_f1: micro-F1 on the posts that were auto-suggested only",
             f"- ms_per_post: median time of one engine.suggest call (cleaning included), {TIMING_CALLS} calls", "",
             markdown_table(results[main_columns]), "",
             f"Default model: `{engine.default_model}`. {engine.settings.get('selection_note', '')}", "",
             "## Macro-F1 by tag frequency (thirds by training count)", "", markdown_table(bucket_table), "",
             "## Short drafts (title + first N words of the body)", "", markdown_table(short), "",
             "See also: test_per_tag.csv, similar_tag_confusions.csv, test_predictions.csv, error_examples.md, "
             "figures/model_comparison.png, figures/test_by_frequency.png.", ""]
    (REPORTS / "test_results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
