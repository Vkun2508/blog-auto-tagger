# Learning notes

Plain-language notes on everything we built. Read these before your presentation. Each section says
**what** we did, **why**, and **where** it lives in the code. Real numbers from our runs are in
`reports/` (the report quotes them); these notes explain the ideas behind them.

---

## Session 0: setup

**The project in one sentence:** a program reads a blog post and suggests up to 5 tags from an approved list; when it isn't sure, a human editor decides.

**Multi-label:** one post can have several correct tags at once (Python *and* API Development *and* Security). A "multi-class" problem would allow only one.

**Terminal:** a window where you type commands instead of clicking. `cd folder` moves into a folder.

**Virtual environment (venv):** a private box of Python libraries for one project. It stops this project's libraries from clashing with other projects. Ours lives at `C:\Users\vinoc\.venvs\blog-tagger`, outside OneDrive, so OneDrive doesn't sync thousands of library files. You "activate" it each time you open a new terminal; the prompt then starts with `(blog-tagger)`.

**pip:** Python's app store. `python -m pip install -r requirements.txt` installs every library listed in `requirements.txt`.

**git:** a save-history for code. `git add` picks what to save, `git commit` saves a snapshot with a message, `git push` uploads it to GitHub. `.gitignore` lists files git should never save.

**Python pieces we used** (see `scripts/00_python_refresher.py`):
- *variable*: a name holding a value
- *list* `[...]`: several values in order
- *dictionary* `{key: value}`: look something up by its key (our alias map is one)
- *function* `def name(...)`: a reusable recipe
- *loop* `for x in items`: do something for each item
- *if*: do something only when a condition is true

---

## Session 1: getting the data (`scripts/01_download_devto.py`)

**API:** a door a website leaves open for programs. Instead of a web page, it answers with data.
DEV.to's API is free and needs no key: `https://dev.to/api/articles?tag=python` returns posts.

**JSON:** the format that data comes back in. It looks like Python dictionaries and lists, so
`response.json()` turns it straight into Python data.

**Rate limit:** websites stop you if you ask too fast (HTTP code 429 = "too many requests"). We wait
1 second between requests and back off longer if we get a 429. Being polite keeps us from being blocked.

**Caching and resuming:** every post is saved as its own file (`data/raw/<id>.json`). If the script
stops, running it again skips posts already on disk. That's why a 1.5-hour download was safe.

**Copyright:** the article text belongs to its authors, so `data/raw/` is never uploaded to GitHub.
We upload the *script* and the *list of post IDs* (`data/article_ids.csv`) instead, so anyone can
rebuild exactly our dataset. The brief allows a "reproducible data preparation script".

**What we got:** 4,705 posts from Nov 2025 to Nov 2026, found through 29 seed tags. Each post has a
title, the body in Markdown, the author's own tags (up to 4 on DEV.to), a date and a URL.

---

## Session 2: the taxonomy (`taxonomy/taxonomy.yaml`, `tagger/taxonomy.py`)

**Taxonomy:** the official list of tags editors are allowed to use. Ours has **22 tags**. Each one has:
- a **definition** (when to use it), so two editors make the same choice,
- **examples** (titles that clearly deserve it),
- **aliases**: the messy tags authors actually typed that mean the same thing,
- **keywords**: words the keyword baseline looks for in the text.

**Alias map:** a dictionary from raw tag to approved tag: `ml`, `machinelearning`, `deeplearning` →
**Machine Learning**; `llm`, `agents`, `rag`, `mcp` → **AI & LLMs**; `security`, `cybersecurity` →
**Security**. This is the "merge variants such as ML and Machine Learning" step from the brief.
A test checks that no alias points to two different tags.

**Dropped tags:** some author tags say nothing about the topic: `programming`, `beginners`,
`tutorial`, `discuss`, `webdev` (used on 828 posts of every kind). They're listed under
`dropped_generic` with the reason. Keeping them would teach the model to predict noise.

**Case rule for keywords:** "AI", "Rust" and "in Go" are written with capitals in the YAML, so they
only match with capitals. That stops "said" matching "ai", "rust on my bike" matching Rust, or
"go to settings" matching Go.

**Tag counts:** we count how many posts carry each tag in each split and flag tags with too few
examples to evaluate reliably (`reports/tag_counts.csv`). Frontend is the smallest tag (145 posts); only
Data Science is flagged (just 10 test posts).

**An alias we got wrong:** we first mapped `statemanagement` to React. Error analysis showed Flutter posts
using it too, so it moved to `dropped_generic`. The audit also showed `oauth` should mean Security. Fixing
the taxonomy when the data proves it wrong is normal, and a good story for the viva.

---

## Session 3: cleaning (`tagger/cleaning.py`)

**Why clean?** A model learns from every word you give it. If every post from one blog ends with
"Follow me on Twitter", the model may learn that sentence instead of the topic. The brief asks us
to remove navigation, footers, tag widgets and template text.

**HTML vs Markdown:** HTML uses tags like `<p>`, `<nav>`, `<footer>`. Markdown uses symbols like
`# Heading` and `[link](url)`. Our cleaner detects which one it got and handles both (plus plain text).

**What gets removed:** `<nav>`, `<footer>`, `<aside>`, `<script>` and friends; DEV.to embeds like
`{% embed ... %}`; images and bare URLs; lines like "follow me", "subscribe", "originally published at";
and **tag-widget lines** such as `#python #ai`, because those would literally hand the answer to the
model (this is called *label leakage*).

**Code blocks:** we keep the language name and the first 30 lines. Code holds useful clues
(`import pandas`, `kubectl apply`) but a 500-line dump would drown out the text.

**Regex (regular expression):** a mini-language for finding text patterns, e.g. `^#\w+` = "a line
starting with a hashtag". We use it for the boilerplate and widget lines.

**Tests:** `tests/test_cleaning.py` feeds the cleaner small examples and checks the result. Tests
prove the code does what we claim and catch mistakes when we change something later.

---

## Session 4: duplicates and the split (`scripts/02_prepare_data.py`)

**Train / validation / test:** like studying from a textbook (train), doing practice papers to decide
how to revise (validation), and sitting the final exam you've never seen (test). We only look at the
test set once, at the very end.

**Why split by date?** The system will tag *new* drafts. Training on older posts and testing on the
**newest** 15% imitates that: oldest 70% train, next 15% validation, newest 15% test.

**Why group duplicates first?** Authors republish posts (801 of ours say they were first published on
another blog) and sometimes post the same article twice. If one copy lands in train and another in test,
the model has "seen the exam" and the score is fake. We find duplicates
by same title, same original URL, or 90%+ similar text, join them into groups (union-find), and keep
each group in one split.

**Cosine similarity:** a score from 0 to 1 saying how alike two texts' word lists are. 1 = the same words
in the same proportions.

---

## Session 5: measuring quality (`tagger/metrics.py`)

Toy example: the true tags are {Python, API}. The model suggests {Python, Docker}.
- **True positive (TP):** suggested and correct → Python.
- **False positive (FP):** suggested but wrong → Docker (a "wrong suggestion").
- **False negative (FN):** correct but not suggested → API (a "missed tag").
- **Precision** = TP / (TP + FP) = 1/2: of what we suggested, how much was right?
- **Recall** = TP / (TP + FN) = 1/2: of what was right, how much did we find?
- **F1** = 2·P·R / (P + R): one number that is high only when both are high.

**Micro vs macro:** *micro* adds up TP/FP/FN over all tags, so common tags dominate. *Macro* computes
F1 per tag and averages, so a rare tag like Frontend counts as much as AI & LLMs. Reporting both
shows if we're only good at the popular tags.

**P@3 and R@3:** look only at the model's top 3 tags per post. P@3 = how many of those 3 are right;
R@3 = how many of the true tags appear in the top 3. Useful because the editor sees a ranked list.

**Review coverage:** the share of posts the system tags on its own, without sending to an editor.

---

## Session 5–6: the keyword baseline and TF-IDF (`tagger/models/keyword.py`, `tfidf.py`)

**Baseline:** the simplest reasonable method. If a clever model can't beat it, the clever model isn't
worth it. Ours counts taxonomy keywords: a match in the title counts 3, each match in the body counts 1,
and the total becomes a 0–1 score.

**TF-IDF:** turns text into numbers. *TF* (term frequency): how often a word appears in this post.
*IDF* (inverse document frequency): how rare the word is across all posts. "the" is everywhere so it
scores low; "kubectl" is rare so it scores high. Each post becomes a long list of word weights.
We also use **bigrams** (two-word phrases like "machine learning").

**Logistic regression:** a simple model that learns a weight for every word and outputs a probability
from 0 to 1 ("how likely is this post about Docker?").

**One-vs-rest:** we train **one yes/no model per tag** (22 of them). Each answers its own question,
so a post can get several yes answers. That's how multi-label works here.

**Title weight experiment:** titles are short but packed with topic words. We built separate TF-IDF
features for the title and body and tried multiplying the title part by 0, 1, 2 and 3, keeping
whichever scored best on validation (the brief asks for this test). See `reports/figures/title_weight.png`.
**Result:** weight 0 and 1 tied (0.688 vs 0.682) and 2-3 were worse. DEV.to titles are often catchy
("I got rejected 2 minutes after applying") rather than descriptive, so the body carries the topic.

**Fair comparison:** each setting was judged at its *own best threshold*, not at a fixed 0.5. A strongly
regularised model rarely scores above 0.5 even when its ranking is right, so a fixed cut-off would
measure the score scale instead of the quality.

**C (regularisation):** how strongly the model is allowed to trust individual words. Small C = cautious,
large C = trusts more (risk of memorising). We tried 1, 4 and 16 on validation.

---

## Session 7: the semantic model (`tagger/models/semantic.py`)

**Embedding:** a pretrained neural network (`all-MiniLM-L6-v2`, free, 90 MB) turns a piece of text into
384 numbers that capture its *meaning*. Texts about similar things get similar numbers, even with
different words ("container orchestration" vs "Kubernetes").

**Chunking:** the model reads at most about 256 word-pieces at a time, but posts are often 1,000+ words.
So we cut the body into ~150-word chunks (at most 8, spread across the post so long articles are
covered start to end) and embed each one.

**Pooling = combining chunks:** we tried four ways on validation: title only; the **mean** (average) of
the chunk vectors; title + mean; title + **max** (the strongest value in each position). The winner is
documented in `reports/validation_experiments.md` and the report.
**Result:** the mean of the body chunks won (0.667); the title alone was weakest (0.522).

**Then logistic regression again:** the 384 or 768 numbers per post go into the same one-vs-rest
logistic regression as before. Embeddings are cached to disk so they're computed only once.

**Hybrid:** the average of the TF-IDF and semantic scores. Two models that make *different* mistakes
often do better together. **Result:** 70% TF-IDF + 30% semantic was best, and the hybrid became the
app's default model.

---

## Session 8: thresholds (`scripts/04_tune_thresholds.py`)

Every model gives each tag a score from 0 to 1. The **threshold** decides how high a score must be before
we suggest the tag.
- **Low threshold:** more suggestions → fewer missed tags, but more wrong ones.
- **High threshold:** fewer, safer suggestions → more missed tags, and more posts with *no* tag, which
  go to an editor (higher review rate).

We tried thresholds from 0.05 to 0.95 and a maximum of 1 to 5 tags per post on the **validation** set
and kept the best micro-F1. Tags with at least 20 validation examples can get their **own** threshold
(some tags are easy, some hard); we keep per-tag thresholds only if they actually help. Charts:
`reports/figures/threshold_tradeoff_*.png` and `review_rate_vs_threshold.png`.

**Review rule** (`tagger/suggest.py`): a post goes to the editor when it has fewer than 30 words
(too little text to judge), is empty, or no tag passes its threshold.

---

## Session 9: final evaluation (`scripts/05_evaluate.py`, `scripts/06_audit.py`)

We run every model, with its tuned settings, on the **test set** (newest posts, never seen before) and
report micro-F1, macro-F1, P@3, R@3, review coverage and speed (milliseconds per post). Results are in
`reports/test_results.md`.

**The numbers to remember (test set, 705 newest posts):** keyword baseline micro-F1 0.600; TF-IDF 0.714
(11 ms per post); semantic 0.675; **hybrid 0.719** (173 ms per post, 97% of posts tagged without review).
The hybrid is used in the app; TF-IDF is nearly as good and 16x faster, so it suits bulk tagging.

**By tag frequency:** we split tags into common, medium and rare thirds and report each, because a
model can look good overall while failing the rare tags.

**Error analysis** (`reports/error_examples.md`): real mistakes in four groups: similar tags
(AI & LLMs vs Machine Learning, Docker & Kubernetes vs DevOps), short drafts, niche topics, and long posts
that cover several subjects.

**Incomplete labels:** authors could only pick 4 tags and often spent slots on generic ones like
`beginners`. So a post about deploying Python with Docker may be labelled only "Python". If the model
correctly suggests Docker & Kubernetes, the scoring counts it as *wrong*. The **audit**
(`reports/audit/`) checks 40 test posts by hand: which tags are really missing or wrong. Re-scoring with
the corrected labels shows how much the original labels underrate the model. **Result:** precision rose from
0.704 to 0.765, 7 of 40 posts were missing a correct tag, and 21% of the model's "wrong" suggestions were
actually right.

---

## Sessions 10–11: the editor app (`streamlit_app.py`)

**Streamlit:** a Python library that turns a script into a web app. Every click re-runs the script from
top to bottom; **session state** (`st.session_state`) is the app's memory between re-runs.

**Tag a draft:** paste a title and body (text, Markdown or HTML), get ranked tags with score bars. The
thin tick on each bar is that tag's threshold: bars past the tick are suggested. The editor ticks or
unticks tags, adds missing ones (only from the approved list), and saves the decision to
`feedback/feedback.jsonl`, one JSON record per line, for future evaluation.

**Supporting sentence ("Why:")**: for each tag we show the sentence from the post with the most keyword
matches for that tag. It's a simple, explainable rule, documented here and in the report.

**Batch review:** tag 10 demo drafts, an uploaded CSV, or the latest DEV.to posts; edit the final tags
in a table; export CSV or JSON with post_id, suggested_tags, scores, review_status and final_tags (the
brief's export).

---

## Session 12: the live link

**Hosting:** Streamlit Community Cloud runs our app for free from the GitHub repo. When we push new code,
the app updates. It sleeps after a while without visitors; clicking "Wake up" starts it again (~1 minute).

**CPU-only PyTorch:** the semantic model needs PyTorch. The normal version includes GPU code (gigabytes);
the CPU version is much smaller and fits the free server.

**Feedback on the host is temporary:** the free server's disk resets on restart, so the app offers
download buttons. A real deployment would save to a database.

---

## Session 13–14: report, slides, submission

The report tells the story in the order the brief asks for: problem → data → taxonomy → cleaning →
split → models → thresholds → results → errors → limitations → how to reproduce. Every number comes
from a file in `reports/`. The viva questions with answers are in `docs/VIVA_QA.md`.
