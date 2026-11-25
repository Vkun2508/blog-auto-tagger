"""Session 6: a human audit of the labels. Authors' own tags are not perfect, so a person checks
40 random test posts and we see how that changes the model's measured quality.

Step 1 (seconds):  python scripts/06_audit.py --make
    writes reports/audit/audit_sample.csv. Open it in Excel and, for each post, fill in
    missing_tags (approved tags the post should have) and wrong_tags (original tags that do not fit),
    separated by "; ", plus any reviewer_notes.
    Also writes reports/audit/audit_context.csv with the first 220 words of each cleaned post,
    so the reviewer can judge a post without opening it (not uploaded to GitHub: authors' text).
Step 2 (seconds):  python scripts/06_audit.py --score
    reads the filled file and writes reports/audit/audit_results.md and audit_results.csv.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))          # lets this script import the tagger package

from tagger.dataset import load_posts  # noqa: E402
from tagger.engine import TaggingEngine  # noqa: E402
from tagger.taxonomy import load_taxonomy  # noqa: E402

RAW_DIR = PROJECT / "data" / "raw"
AUDIT_DIR = PROJECT / "reports" / "audit"
SAMPLE_FILE = AUDIT_DIR / "audit_sample.csv"
CONTEXT_FILE = AUDIT_DIR / "audit_context.csv"
CONTEXT_WORDS = 220
SAMPLE_SIZE = 40
SEED = 42
COLUMNS = ["post_id", "title", "url", "original_tags", "model_suggested_tags", "model_scores",
           "missing_tags", "wrong_tags", "reviewer_notes"]


def split_tags(cell):
    return [t.strip() for t in str(cell or "").split(";") if t.strip()]


def make_sample():
    engine = TaggingEngine.load()
    test = load_posts("test")
    sample = sorted(random.Random(SEED).sample(test, SAMPLE_SIZE), key=lambda p: p["id"])
    # The engine gets the raw post text, exactly as the app would.
    items = []
    for post in sample:
        raw = json.loads((RAW_DIR / f"{post['id']}.json").read_text(encoding="utf-8"))
        items.append((post["id"], raw["title"], raw["body_markdown"]))
    suggestions = dict(engine.suggest_many(items))
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_FILE, "w", newline="", encoding="utf-8-sig") as f:   # BOM so Excel reads UTF-8
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for post in sample:
            s = suggestions[post["id"]]
            writer.writerow([post["id"], post["title"], post["url"], "; ".join(post["tags"]),
                             "; ".join(s.tag_names()), "; ".join(f"{c.tag}: {c.score:.2f}" for c in s.candidates),
                             "", "", ""])
    with open(CONTEXT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["post_id", "title", "url", "original_tags", "model_suggested_tags",
                         f"body_first_{CONTEXT_WORDS}_words"])
        for post in sample:
            writer.writerow([post["id"], post["title"], post["url"], "; ".join(post["tags"]),
                             "; ".join(suggestions[post["id"]].tag_names()),
                             " ".join(post["body"].split()[:CONTEXT_WORDS])])
    print(f"Wrote {SAMPLE_SIZE} posts (model: {engine.default_model}) to {SAMPLE_FILE}")
    print(f"Wrote reviewer context (first {CONTEXT_WORDS} words of each post) to {CONTEXT_FILE}")
    print("Fill in missing_tags / wrong_tags / reviewer_notes, then run: python scripts/06_audit.py --score")


def precision_recall_f1(pairs):
    """Micro precision/recall/F1 over (suggested set, true set) pairs."""
    tp = sum(len(s & t) for s, t in pairs)
    fp = sum(len(s - t) for s, t in pairs)
    fn = sum(len(t - s) for s, t in pairs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score_sample(path):
    approved = set(load_taxonomy().tags)
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    unknown, original_pairs, corrected_pairs = [], [], []
    posts_with_missing, model_fp, fp_judged_correct = 0, 0, 0
    for row in rows:
        missing, wrong = split_tags(row["missing_tags"]), split_tags(row["wrong_tags"])
        # Reviewers type tag names by hand, so check every one against the taxonomy.
        for tag in missing + wrong:
            if tag not in approved:
                unknown.append(f"post {row['post_id']}: '{tag}'")
        missing = {t for t in missing if t in approved}
        wrong = {t for t in wrong if t in approved}
        original = set(split_tags(row["original_tags"]))
        corrected = (original - wrong) | missing
        suggested = set(split_tags(row["model_suggested_tags"]))
        original_pairs.append((suggested, original))
        corrected_pairs.append((suggested, corrected))
        posts_with_missing += bool(missing)
        model_fp += len(suggested - original)
        fp_judged_correct += len((suggested - original) & missing)

    table = pd.DataFrame([
        {"labels": "original (author tags)", **dict(zip(["precision", "recall", "f1"], precision_recall_f1(original_pairs)))},
        {"labels": "corrected (after audit)", **dict(zip(["precision", "recall", "f1"], precision_recall_f1(corrected_pairs)))},
    ]).round(4)
    table.to_csv(AUDIT_DIR / "audit_results.csv", index=False)
    share = fp_judged_correct / model_fp if model_fp else 0.0
    lines = ["# Label audit results", "", f"Posts audited: {len(rows)} (random test posts, seed {SEED}).", "",
             "| Labels | Precision | Recall | F1 |", "|---|---|---|---|"]
    lines += [f"| {r.labels} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} |" for r in table.itertuples()]
    lines += ["", f"- Posts with at least one missing tag: {posts_with_missing} of {len(rows)}",
              f"- Model 'false positives' (suggested but not in the author's tags): {model_fp}; "
              f"judged correct by the reviewer: {fp_judged_correct} ({share:.0%})",
              f"- Unknown tag names typed by the reviewer (ignored): {', '.join(unknown) or 'none'}", ""]
    (AUDIT_DIR / "audit_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Make or score the label audit sample.")
    parser.add_argument("--make", action="store_true", help="write a new audit sample")
    parser.add_argument("--score", action="store_true", help="score the filled audit sample")
    parser.add_argument("--file", default=str(SAMPLE_FILE), help="filled audit CSV (default: audit_sample.csv)")
    args = parser.parse_args()
    if args.make:
        make_sample()
    elif args.score:
        score_sample(Path(args.file))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
