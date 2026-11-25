# Blog Auto-Tagger

Suggests up to five approved tags for a blog post, with a score for each, and sends uncertain posts to
an editor for review. Editors accept, remove or add tags, and the decisions are saved and exported.

Internship project: **Automated Content Tagging for Blogs** (Persevex, Artificial Intelligence).
Author: Vinochan R.

- **Live demo:** https://blog-auto-tagger.streamlit.app (free hosting: if it is asleep, click "Yes, get this app back up" and wait about a minute)
- **Report:** `reports/report.md` (PDF in the submission folder)
- **Study notes:** `docs/LEARNING_NOTES.md` · **Viva prep:** `docs/VIVA_QA.md`

## What it does

```
DEV.to posts ─► clean text ─► map author tags to 22 approved tags ─► group duplicates ─► split by date
                                                                     (oldest 70% train · 15% validation · newest 15% test)
      ┌──────────────────────────────────────────────────────────────────────┘
      ▼
 3 models on the same split, plus a hybrid:
   1. Keyword baseline   taxonomy keywords, title matches count 3x
   2. TF-IDF + LogReg    word weights, one yes/no classifier per tag, title weight tested
   3. Semantic           MiniLM sentence embeddings of title + body chunks, then LogReg
   4. Hybrid             average of 2 and 3
      ▼
 tune threshold + max tags on validation ─► evaluate once on the newest posts ─► pick the default model
      ▼
 Streamlit editor app: tag a draft · batch review · CSV/JSON export · how it works
```

## Setup (Windows, PowerShell)

```powershell
py -3.13 -m venv C:\Users\vinoc\.venvs\blog-tagger
& C:\Users\vinoc\.venvs\blog-tagger\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` installs the CPU-only build of PyTorch (smaller, and the same as the free host uses).
On macOS or Linux, activate with `source <venv>/bin/activate`; everything else is the same.

## Commands, in order

| Step | Command | What it does | Time |
|------|---------|--------------|------|
| 0 | `python scripts/00_python_refresher.py` | Python warm-up | seconds |
| 1 | `python scripts/01_download_devto.py` | Download the posts listed by tag from the DEV.to API into `data/raw/` (resumable; `--test` for a 20-second trial) | ~1.5 h |
| 2 | `python scripts/02_prepare_data.py` | Clean text, map tags with the alias map, group near-duplicates, split by date, write tag counts | ~1 min |
| 3 | `python scripts/03_train.py` | Train keyword, TF-IDF (title-weight grid) and semantic (pooling comparison) models; save to `models/` | ~10 min |
| 4 | `python scripts/04_tune_thresholds.py` | Choose threshold and max tags per model on validation; pick the default model | ~1 min |
| 5 | `python scripts/05_evaluate.py` | Final test-set evaluation, tag-frequency breakdown, short drafts, error examples | ~3 min |
| 6 | `python scripts/06_audit.py --make` then `--score` | Reviewer audit of 40 held-out posts and the effect of incomplete labels | ~1 min |
| 7 | `python -m pytest -q` | Run the tests | ~30 s |
| 8 | `streamlit run streamlit_app.py` | Start the editor app at http://localhost:8501 | - |

Tag one draft from the command line:

```powershell
python -m tagger --title "Securing a FastAPI service with OAuth2" --body "Your draft text..."
python -m tagger --file my_draft.md
```

Everything is seeded (`random_state=42`), so re-running steps 2 to 6 gives the same numbers.

## Project layout

```
taxonomy/taxonomy.yaml   22 approved tags: definition, examples, aliases (alias map), keywords
tagger/                  the reusable engine
  cleaning.py            HTML / Markdown / plain text -> clean text (removes nav, footers, tag widgets, boilerplate)
  taxonomy.py            loads the taxonomy; maps raw author tags to approved tags
  models/                keyword.py · tfidf.py · semantic.py · hybrid.py (same predict_scores() shape)
  suggest.py             thresholds, max tags, review rules (empty / too short / low confidence)
  engine.py              loads models + settings; suggest() for one draft, suggest_many() for a batch
  metrics.py             micro/macro precision, recall, F1, P@3, R@3
  feedback.py            editor decisions -> feedback.jsonl, CSV/JSON export
scripts/                 01-06, run in order (see the table above)
models/                  trained models + settings.json (thresholds, default model)
reports/                 tag counts, experiments, test results, charts, error examples, audit, report.md
samples/demo_batch.csv   10 original demo drafts for the batch page
streamlit_app.py         the editor app
tests/                   pytest checks for the brief's success criteria
docs/                    study notes and viva questions
```

## Data and copyright

Article text belongs to its authors, so `data/raw/` and `data/processed/` are not in this repository.
`data/article_ids.csv` lists every post used; step 1 rebuilds the dataset from the public DEV.to API.

## How the brief is met

| Brief | Where |
|-------|-------|
| Taxonomy with definitions, examples and an alias map; variants merged | `taxonomy/taxonomy.yaml` |
| Tag counts and tags with too few examples | `reports/tag_counts.csv`, `reports/dataset_summary.md` |
| Template text removed; near-duplicates grouped; recent posts held out | `tagger/cleaning.py`, `scripts/02_prepare_data.py` |
| Reviewer check of held-out labels | `reports/audit/` |
| Keyword baseline vs TF-IDF + one-vs-rest logistic regression, title weight tested | `tagger/models/`, `reports/validation_tfidf_grid.csv` |
| Semantic model with chunking and documented pooling | `tagger/models/semantic.py`, `reports/validation_semantic_pooling.csv` |
| Threshold, max suggestions, per-tag thresholds, trade-offs | `scripts/04_tune_thresholds.py`, `reports/threshold_tradeoffs.csv` |
| Editor prototype with scores, accept/remove/add, saved feedback | `streamlit_app.py`, `tagger/feedback.py` |
| micro/macro-F1, P@3, R@3, by tag frequency, review coverage, speed | `reports/test_results.md`, `reports/test_by_frequency.csv` |
| Error examples and incomplete labels | `reports/error_examples.md`, `reports/audit/audit_results.md` |
| Only approved tags; empty and very short drafts go to review | `tagger/suggest.py`, `tests/` |
| Batch export with post IDs, suggested tags, scores, review status, final tags | Batch review page |
