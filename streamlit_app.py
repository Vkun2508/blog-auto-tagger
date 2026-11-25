"""Blog Auto-Tagger: an editor app to tag a draft, review a batch and export final tags.

The model proposes tags (pencil grey), the editor decides (ink), red marks what needs attention.
All the tagging logic lives in the `tagger` package; this file only draws screens.
"""

import html
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from tagger.engine import TaggingEngine
from tagger.feedback import (
    EXPORT_COLUMNS,
    append_feedback,
    export_csv,
    export_json,
    load_feedback,
    make_record,
)
from tagger.suggest import REVIEW_REASON_TEXT

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
DEMO_BATCH = ROOT / "samples" / "demo_batch.csv"
LOADING_MESSAGE = "Loading the semantic model (first run only, about 30 seconds)…"
DEVTO_TAGS = ["python", "javascript", "webdev", "beginners", "tutorial", "react", "devops", "docker", "ai", "programming"]
# Older Streamlit versions have no MultiselectColumn, so we keep a text fallback.
HAS_MULTISELECT = hasattr(st.column_config, "MultiselectColumn")

st.set_page_config(page_title="Blog Auto-Tagger", page_icon=":material/sell:", layout="wide")

# ---------------------------------------------------------------- example drafts

EXAMPLE_NORMAL = (
    "Securing a FastAPI REST API with OAuth2 and JWT",
    """A REST API that anyone can call is a REST API that anyone can break. In this post we secure a
Python FastAPI service with OAuth2 and JSON Web Tokens (JWT), step by step.

First we add a /token endpoint that checks a username and a hashed password, then returns a signed
JWT access token with a short expiry. Passwords are hashed with bcrypt and never stored in plain text.

Next we write a dependency that reads the Authorization header, verifies the token signature and
loads the current user. Any route that depends on it returns a 401 error when the token is missing
or expired. We also add scopes so that read-only clients cannot call the endpoints that change data.

Finally we cover refresh tokens, keeping the secret key in an environment variable, rate limiting
login attempts, and testing the protected endpoints with pytest and the FastAPI TestClient.""",
)

EXAMPLE_SHORT = ("Docker tips", "Some quick tips.")

EXAMPLE_HTML = (
    "",
    """<html><body>
<nav><a href="/">Home</a> <a href="/archive">Archive</a> <a href="/about">About</a> <a href="/login">Log in</a></nav>
<article>
<h1>Autoscaling workloads in Kubernetes</h1>
<p>Kubernetes can add and remove pods for you. The Horizontal Pod Autoscaler (HPA) watches CPU,
memory or custom metrics and changes the replica count of a Deployment to match the load.</p>
<p>To use it, set resource requests on your containers, install the metrics server, and create an HPA
with a target such as 70 percent CPU. When traffic grows, the HPA adds pods; when it drops, it
removes them after a cool-down period so the cluster does not flap.</p>
<p>For node capacity, pair the HPA with the Cluster Autoscaler, which adds worker nodes when pods
cannot be scheduled. Tools like KEDA go further and scale on queue length or events. Always load test
your scaling rules, and keep the minimum replicas above one for services that must stay available.</p>
</article>
<div class="tags">#webdev #python</div>
<footer>Copyright 2025. Subscribe to the newsletter. Follow us on social media.</footer>
</body></html>""",
)

# ---------------------------------------------------------------- styling

CSS = """
<style>
.tag-row { margin: 0 0 14px 0; }
.tag-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.tag-name { font-weight: 600; overflow-wrap: anywhere; }
.tag-score, .score-num { font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; }
.bar { position: relative; height: 8px; margin: 5px 0 3px; background: #E3DDD2; border-radius: 4px; corner-shape: squircle; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-fill.on { background: #1F2328; }
.bar-fill.off { background: #A8A29A; }
.bar-tick { position: absolute; top: -3px; width: 2px; height: 14px; background: #B3261E; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
h1, h2, h3 { text-wrap: balance; }
p, li { text-wrap: pretty; }
.why { font-size: 0.85rem; color: #6B6760; }
.legend { font-size: 0.85rem; color: #6B6760; margin-bottom: 12px; }
.pills { display: flex; flex-wrap: wrap; gap: 6px; margin: 4px 0 12px; }
.pill { padding: 2px 10px; border: 1px solid #1F2328; border-radius: 999px; corner-shape: squircle; font-size: 0.85rem; }
.pill-none { color: #6B6760; font-size: 0.9rem; }
.steps { display: flex; flex-wrap: wrap; gap: 8px; list-style: none; padding: 0; margin: 8px 0 16px; counter-reset: step; }
.steps li { counter-increment: step; padding: 6px 12px; background: #F2EEE6; border: 1px solid #E3DDD2; border-radius: 6px; }
.steps li::before { content: counter(step) ". "; font-family: "IBM Plex Mono", monospace; color: #B3261E; }
</style>
"""

# ---------------------------------------------------------------- engine

@st.cache_resource(show_spinner="Starting the tagger…")
def get_engine():
    # Cached so the models are loaded once per server, not once per click.
    return TaggingEngine.load()


def fallback_model(engine):
    """The default model, or TF-IDF if the default can't load here (e.g. torch missing on a host)."""
    models = engine.available_models
    for name in (engine.default_model, "tfidf"):
        if name in models:
            return name
    return models[0]


def current_model(engine):
    """The model chosen in the sidebar (falls back to the default)."""
    return st.session_state.get("model_choice", fallback_model(engine))


def run_suggest(engine, title, body, model):
    """Run one suggestion; show a friendly error and return None if the model fails."""
    message = LOADING_MESSAGE if model in ("semantic", "hybrid") else "Tagging…"
    try:
        with st.spinner(message):
            return engine.suggest(title, body, model=model)
    except Exception as error:  # a model can fail to load (e.g. torch missing); the others still work
        st.error(
            f"The {engine.model_label(model)} model could not run ({error}). "
            "Pick another model in the sidebar and try again."
        )
        return None


def save_records(records, run_key):
    """Append decisions to the feedback file once per run, and count them for this session."""
    saved_runs = st.session_state.setdefault("saved_runs", set())
    if run_key in saved_runs:  # idempotent: a second click must not write the same decisions again
        return
    for record in records:
        append_feedback(record)
    saved_runs.add(run_key)
    st.session_state["saved_count"] = st.session_state.get("saved_count", 0) + len(records)
    st.toast(f"Saved {len(records)} decision(s) to feedback", icon=":material/check:")


def download_buttons(records, key, csv_name, json_name):
    """A pair of CSV / JSON download buttons for a list of records."""
    left, right = st.columns(2)
    off = not records
    left.download_button("Download CSV", export_csv(records) if records else b"", csv_name,
                         mime="text/csv", icon=":material/download:", disabled=off, key=f"{key}_csv")
    right.download_button("Download JSON", export_json(records) if records else b"", json_name,
                          mime="application/json", icon=":material/download:", disabled=off, key=f"{key}_json")


# ---------------------------------------------------------------- small HTML pieces

def score_rows(suggestion):
    """Suggested tags plus the top candidates, best score first, no duplicates."""
    rows = list(suggestion.tags)
    seen = {row.tag for row in rows}
    rows += [c for c in suggestion.candidates if c.tag not in seen]
    return sorted(rows, key=lambda row: row.score, reverse=True)


def score_list_html(suggestion):
    """The signature element: one score bar per tag, with a tick at its threshold."""
    suggested = {t.tag for t in suggestion.tags}
    parts = ['<div class="legend">Bar = model score · tick = the threshold a tag must pass</div>']
    for row in score_rows(suggestion):
        css = "on" if row.tag in suggested else "off"
        why = f'<div class="why">Why: {html.escape(row.snippet)}</div>' if row.snippet else ""
        parts.append(
            f'<div class="tag-row"><div class="tag-head"><span class="tag-name">{html.escape(row.tag)}</span>'
            f'<span class="tag-score">{row.score:.2f}</span></div>'
            f'<div class="bar"><div class="bar-fill {css}" style="width:{min(max(row.score, 0), 1) * 100:.1f}%"></div>'
            f'<div class="bar-tick" style="left:{min(max(row.threshold, 0), 1) * 100:.1f}%"></div></div>{why}</div>'
        )
    return "".join(parts)


def pills_html(tags):
    if not tags:
        return '<div class="pill-none">No tags selected yet.</div>'
    return '<div class="pills">' + "".join(f'<span class="pill">{html.escape(t)}</span>' for t in tags) + "</div>"


def in_taxonomy_order(engine, tags):
    wanted = set(tags)
    return [t for t in engine.taxonomy.tags if t in wanted]


# ---------------------------------------------------------------- page 1: tag a draft

def first_heading(text):
    """Title guess for an uploaded file: first Markdown '# ' line or HTML <h1>/<title>."""
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE) or re.search(
        r"<(?:h1|title)[^>]*>(.*?)</(?:h1|title)>", text, re.IGNORECASE | re.DOTALL
    )
    return re.sub(r"<[^>]+>", "", match.group(1)).strip() if match else ""


def fill_from_upload():
    # Runs as a callback, i.e. before the widgets are drawn, so it may set their values.
    uploaded = st.session_state.get("draft_file")
    if uploaded is None:
        return
    text = uploaded.getvalue().decode("utf-8", errors="replace")
    st.session_state["draft_body"] = text
    if not st.session_state.get("draft_title", "").strip():
        st.session_state["draft_title"] = first_heading(text)


def fill_example(example):
    st.session_state["draft_title"], st.session_state["draft_body"] = example


def tag_page():
    engine = get_engine()
    model = current_model(engine)
    st.title("Tag a draft")
    st.caption("Paste a draft, get suggested tags with the evidence, then confirm the final tags.")

    st.caption("Try an example")
    a, b, c = st.columns(3)
    a.button("Normal draft (FastAPI security)", on_click=fill_example, args=(EXAMPLE_NORMAL,),
             key="example_normal", icon=":material/article:", width="stretch")
    b.button("Very short draft", on_click=fill_example, args=(EXAMPLE_SHORT,),
             key="example_short", icon=":material/short_text:", width="stretch")
    c.button("HTML page with menus", on_click=fill_example, args=(EXAMPLE_HTML,),
             key="example_html", icon=":material/code:", width="stretch")

    st.text_input("Title", key="draft_title")
    st.text_area(
        "Body", key="draft_body", height=300,
        help="Paste plain text, Markdown or HTML. Navigation, footers and tag widgets are removed automatically.",
    )
    st.file_uploader(
        "Or upload a file to fill the body", type=["md", "markdown", "html", "htm", "txt"],
        key="draft_file", on_change=fill_from_upload,
    )

    if st.button("Suggest tags", type="primary", key="suggest", icon=":material/auto_awesome:"):
        suggestion = run_suggest(engine, st.session_state.get("draft_title", ""), st.session_state.get("draft_body", ""), model)
        if suggestion is not None:
            # A new run id gives the checkboxes fresh keys, so old ticks never leak into new results.
            st.session_state["result"] = {
                "run": st.session_state.get("run_counter", 0) + 1,
                "title": st.session_state.get("draft_title", ""),
                "suggestion": suggestion,
            }
            st.session_state["run_counter"] = st.session_state["result"]["run"]

    result = st.session_state.get("result")
    if result is None:
        st.info("Nothing tagged yet. Write or paste a draft above, or pick one of the examples, then choose Suggest tags.",
                icon=":material/edit_note:")
    else:
        show_result(engine, result)
    show_saved_decisions()


def show_result(engine, result):
    suggestion = result["suggestion"]
    st.divider()
    st.caption(f"Model: {engine.model_label(suggestion.model)}")
    left, right = st.columns(2, gap="large")

    with left:
        if suggestion.needs_review:
            st.warning(REVIEW_REASON_TEXT.get(suggestion.review_reason, "This draft needs a human look."),
                       icon=":material/flag:")
        else:
            st.success(f"Confident enough to suggest {len(suggestion.tags)} tags", icon=":material/check_circle:")
        st.html(score_list_html(suggestion))
        with st.expander("Cleaned text the model saw"):
            st.caption(f"{suggestion.n_words} words after cleaning")
            st.text(f"Title: {suggestion.title_clean}\n\n{suggestion.body_clean[:1500]}")

    with right:
        st.subheader("Editor decision")
        suggested = {t.tag for t in suggestion.tags}
        picked = [
            row.tag for row in score_rows(suggestion)
            if st.checkbox(f"{row.tag} ({row.score:.2f})", value=row.tag in suggested, key=f"pick_{result['run']}_{row.tag}")
        ]
        extra_options = [t for t in engine.taxonomy.tags if t not in picked]
        extras = st.multiselect("Add a missing tag", extra_options, key=f"extra_{result['run']}",
                                help="Only approved tags can be added.")
        final_tags = in_taxonomy_order(engine, picked + extras)
        st.caption("Final tags")
        st.html(pills_html(final_tags))
        run_key = f"draft-{result['run']}"
        already_saved = run_key in st.session_state.get("saved_runs", set())
        record = make_record(f"draft-{datetime.now():%Y%m%d-%H%M%S}", result["title"], suggestion, final_tags)
        # on_click runs before the rerun, so the button is already disabled when the page redraws.
        st.button("Decision saved" if already_saved else "Save decision", key="save_decision",
                  icon=":material/save:", disabled=already_saved, on_click=save_records, args=([record], run_key))
        if already_saved:
            st.caption("Saved.")


def show_saved_decisions():
    st.divider()
    st.subheader("Saved decisions this session")
    st.metric("Decisions saved", st.session_state.get("saved_count", 0))
    try:
        records = load_feedback()
    except Exception as error:
        records = []
        st.error(f"Could not read the saved feedback ({error}). Reload the page and try again.")
    st.caption("Download the feedback file: on the hosted demo it is temporary and can disappear.")
    download_buttons(records, "feedback", "feedback.csv", "feedback.json")


# ---------------------------------------------------------------- page 2: batch review

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_devto_posts(tag, count):
    """Latest DEV.to posts for a tag. Cached one hour so reruns do not hit the API again."""
    headers = {"User-Agent": "blog-auto-tagger-demo"}
    listing = requests.get("https://dev.to/api/articles", params={"tag": tag, "per_page": count},
                           headers=headers, timeout=15)
    listing.raise_for_status()
    posts = []
    for item in listing.json():
        # The list has no body text, so each article is fetched on its own.
        detail = requests.get(f"https://dev.to/api/articles/{item['id']}", headers=headers, timeout=15)
        detail.raise_for_status()
        posts.append({"post_id": f"devto-{item['id']}", "title": item.get("title", ""),
                      "body": detail.json().get("body_markdown") or ""})
    return posts


def read_batch_source(source):
    """Return a list of {post_id,title,body} dicts, or None (after showing why)."""
    if source.startswith("Demo"):
        if not DEMO_BATCH.exists():
            st.error("The demo batch file (samples/demo_batch.csv) is missing. Choose another source for now.")
            return None
        return frame_to_posts(pd.read_csv(DEMO_BATCH))
    if source.startswith("Upload"):
        uploaded = st.file_uploader("CSV file", type=["csv"], key="batch_csv",
                                    help="Required columns: post_id, title, body.")
        if uploaded is None:
            return None
        try:
            frame = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f"We could not read that file as a CSV ({error}). Check that it is a comma-separated file and upload it again.")
            return None
        missing = [c for c in ("post_id", "title", "body") if c not in frame.columns]
        if missing:
            st.error(f"The CSV is missing these columns: {', '.join(missing)}. Add them to the header row and upload it again.")
            return None
        return frame_to_posts(frame)
    left, right = st.columns(2)
    tag = left.selectbox("DEV.to tag", DEVTO_TAGS, key="devto_tag")
    count = right.slider("Number of posts", 5, 20, 10, key="devto_count")
    try:
        with st.spinner("Fetching posts from DEV.to…"):
            return fetch_devto_posts(tag, count)
    except Exception as error:
        st.error(f"DEV.to did not answer ({error}). Wait a minute and press the rerun option, or choose another source.")
        return None


def frame_to_posts(frame):
    frame = frame[["post_id", "title", "body"]].fillna("").astype(str)
    return frame.to_dict("records")


def parse_final_tags(value, valid):
    """Turn a table cell (list, or '; '-joined text) into (valid tags, unknown tags)."""
    items = str(value or "").split(";") if isinstance(value, str) else list(value or [])
    items = [str(i).strip() for i in items if str(i).strip()]
    return [i for i in items if i in valid], [i for i in items if i not in valid]


def batch_table_rows(results):
    rows = []
    for post_id, title, suggestion in results:
        rows.append({
            "post_id": str(post_id),
            "title": title,
            "status": "Needs review" if suggestion.needs_review else "Auto",
            "reason": REVIEW_REASON_TEXT.get(suggestion.review_reason, "") if suggestion.review_reason else "",
            "suggested": "; ".join(suggestion.tag_names()),
            "top scores": " · ".join(f"{c.tag} {c.score:.2f}" for c in suggestion.candidates[:3]),
        })
    return rows


def batch_page():
    engine = get_engine()
    model = current_model(engine)
    st.title("Batch review")
    st.caption("Tag many posts at once, fix the final tags in the table, then export or save them.")

    source = st.radio("Where do the posts come from?",
                      ["Demo batch (10 sample drafts)", "Upload a CSV", "Latest DEV.to posts"], key="batch_source")
    posts = read_batch_source(source)

    if posts is not None:
        st.caption(f"{len(posts)} posts ready. Model: {engine.model_label(model)}")
        if st.button("Tag this batch", type="primary", key="tag_batch", icon=":material/auto_awesome:"):
            try:
                with st.spinner(LOADING_MESSAGE if model in ("semantic", "hybrid") else "Tagging the batch…"):
                    pairs = engine.suggest_many([(p["post_id"], p["title"], p["body"]) for p in posts], model=model)
            except Exception as error:
                st.error(f"The {engine.model_label(model)} model could not run ({error}). Pick another model in the sidebar and try again.")
                pairs = None
            if pairs is not None:
                titles = {str(p["post_id"]): p["title"] for p in posts}
                results = [(pid, titles.get(str(pid), ""), s) for pid, s in pairs]
                st.session_state["batch"] = {
                    "run": st.session_state.get("batch_counter", 0) + 1,
                    "results": results,
                    "final": {str(pid): s.tag_names() for pid, _, s in results},  # the editor starts from the suggestions
                }
                st.session_state["batch_counter"] = st.session_state["batch"]["run"]

    batch = st.session_state.get("batch")
    if batch is None:
        st.info("Choose a source and press Tag this batch. You will get a table with the suggested tags, "
                "where you can correct the final tags and then export them as CSV or JSON.",
                icon=":material/table:")
        return
    show_batch(engine, batch)


def show_batch(engine, batch):
    results = batch["results"]
    reviews = sum(1 for _, _, s in results if s.needs_review)
    c1, c2, c3 = st.columns(3)
    c1.metric("Posts", len(results))
    c2.metric("Auto-suggested", len(results) - reviews)
    c3.metric("Needs review", reviews)

    only_review = st.toggle("Show only posts that need review", key="only_review")
    rows = batch_table_rows(results)
    # Keep the table's starting data stable while the editor is used; rebuild it only when the filter flips.
    base_key = (batch["run"], only_review)
    if st.session_state.get("batch_base_key") != base_key:
        shown = [r for r in rows if r["status"] == "Needs review"] if only_review else rows
        for row in shown:
            row["final tags"] = (batch["final"][row["post_id"]] if HAS_MULTISELECT
                                 else "; ".join(batch["final"][row["post_id"]]))
        frame = pd.DataFrame(shown)
        if not frame.empty:
            frame = frame[["post_id", "title", "final tags", "status", "suggested", "top scores", "reason"]]
        st.session_state["batch_base"] = frame
        st.session_state["batch_base_key"] = base_key
    base = st.session_state["batch_base"]

    if base.empty:
        st.info("No posts need review in this batch.", icon=":material/check_circle:")
        edited = base
    else:
        if HAS_MULTISELECT:
            final_config = st.column_config.MultiselectColumn("final tags", options=engine.taxonomy.tags,
                                                              help="Click to add or remove approved tags.")
        else:
            final_config = st.column_config.TextColumn("final tags", help="Approved tags separated by '; '.")
        edited = st.data_editor(
            base, key=f"batch_editor_{batch['run']}_{only_review}", hide_index=True, width="stretch",
            disabled=[c for c in base.columns if c != "final tags"],
            column_config={
                "final tags": final_config,
                "title": st.column_config.TextColumn("title", width="medium"),
                "reason": st.column_config.TextColumn("reason", width="small"),
            },
        )

    # Remember edits so they survive filter changes and feed the export.
    valid, unknown = set(engine.taxonomy.tags), set()
    for _, row in edited.iterrows():
        good, bad = parse_final_tags(row["final tags"], valid)
        batch["final"][row["post_id"]] = in_taxonomy_order(engine, good)
        unknown.update(bad)
    if unknown:
        st.warning(f"These tags are not in the approved list and were dropped: {', '.join(sorted(unknown))}.")

    records = [make_record(pid, title, s, batch["final"][str(pid)]) for pid, title, s in results]
    st.subheader("Export")
    st.caption("Exports contain: " + ", ".join(EXPORT_COLUMNS))
    download_buttons(records, "batch", "tag_export.csv", "tag_export.json")
    run_key = f"batch-{batch['run']}"
    already_saved = run_key in st.session_state.get("saved_runs", set())
    st.button("All saved to feedback" if already_saved else "Save all to feedback", key="save_all",
              icon=":material/save:", disabled=already_saved, on_click=save_records, args=(records, run_key))
    if already_saved:
        st.caption("Saved.")


# ---------------------------------------------------------------- page 3: how it works

def show_image(name, caption):
    path = REPORTS / "figures" / name
    if path.exists():
        st.image(str(path), caption=caption)


def show_test_results(engine):
    path = REPORTS / "test_results.csv"
    if not path.exists():
        st.info("The test results are not available yet. Run the evaluation scripts to create reports/test_results.csv.")
        return
    table = pd.read_csv(path)
    # The CSV has a model key and a readable label: show the label, hide the key.
    is_default = (table.iloc[:, 0].astype(str) == engine.default_model).tolist()
    if "label" in table.columns:
        table = table.drop(columns=table.columns[0]).rename(columns={"label": "model"})

    def highlight_default(row):
        mark = "background-color: #F2D9D6; font-weight: 600" if is_default[row.name] else ""
        return [mark] * len(row)

    table = table.rename(columns={
        "label": "Model", "model": "Model", "micro_precision": "Micro precision", "micro_recall": "Micro recall",
        "micro_f1": "Micro-F1", "macro_precision": "Macro precision", "macro_recall": "Macro recall",
        "macro_f1": "Macro-F1", "p_at_3": "P@3", "r_at_3": "R@3", "review_coverage_pct": "Auto-tagged %",
        "auto_micro_f1": "Micro-F1 (auto-tagged)", "ms_per_post": "ms / post",
    })
    number_format = {c: "{:.1f}" if c in ("ms / post", "Auto-tagged %") else "{:.3f}"
                     for c in table.select_dtypes("number").columns}
    st.dataframe(table.style.apply(highlight_default, axis=1).format(number_format),
                 hide_index=True, width="stretch")
    st.caption("The highlighted row is the default model used by the app.")


def how_page():
    engine = get_engine()
    st.title("How it works")
    st.write(
        "The tagger reads a blog post and proposes tags from a fixed list of 22 approved tags. "
        "It was trained on DEV.to posts, using the authors' own tags (mapped to the approved list) as the answers. "
        "Every suggestion shows its score and the sentence that supports it, and unsure cases are sent to a human editor."
    )
    st.html('<ol class="steps"><li>Download</li><li>Clean</li><li>Map tags</li><li>Train 3 models</li><li>Tune thresholds</li></ol>')

    st.header("How well does it work?")
    show_test_results(engine)
    show_image("model_comparison.png", "Models compared on the held-out test posts.")
    show_image("review_rate_vs_threshold.png", "A higher threshold means fewer wrong tags but more posts sent to review.")
    show_image(f"threshold_tradeoff_{engine.default_model}.png", f"Threshold trade-off for the {engine.model_label(engine.default_model)} model.")
    note = engine.settings.get("selection_note")
    if note:
        st.info(note, icon=":material/info:")

    st.header("Approved tags")
    text = st.text_input("Filter tags", key="tag_filter", help="Type part of a tag name or definition.").strip().lower()
    taxonomy = engine.taxonomy
    shown = [t for t in taxonomy.tags if text in t.lower() or text in taxonomy.definitions.get(t, "").lower()]
    if not shown:
        st.caption("No tag matches that filter. Clear the box to see all tags.")
    for tag in shown:
        with st.expander(tag):
            st.write(taxonomy.definitions.get(tag, ""))
            for title, values in (("Examples", taxonomy.examples), ("Aliases (raw tags merged into this one)", taxonomy.aliases),
                                  ("Keywords", taxonomy.keywords)):
                if values.get(tag):
                    st.markdown(f"**{title}:** " + ", ".join(str(v) for v in values[tag]))

    st.header("Limitations")
    st.markdown(
        "- The labels come from the authors' own tags. They are often incomplete (DEV.to allows at most 4 per post), "
        "so a correct suggestion can still count as a miss.\n"
        "- Feedback saved on the hosted demo is temporary. Download it before you leave.\n"
        "- The semantic model needs about 30 seconds to load the first time.\n"
        "- The app can fall asleep after a period of inactivity on the free host and will need a moment to wake up."
    )


# ---------------------------------------------------------------- app shell

def sidebar(engine):
    with st.sidebar:
        st.markdown("### Blog Auto-Tagger")
        st.caption("Suggests tags for blog drafts and lets an editor confirm them.")
        models = engine.available_models
        start = fallback_model(engine)
        st.selectbox(
            "Model", models, key="model_choice",
            index=models.index(start),
            format_func=lambda m: engine.model_label(m),
        )
        chosen = st.session_state.get("model_choice", start)
        config = engine.settings.get("models", {}).get(chosen, {})
        st.caption(f"{'Recommended model · ' if chosen == engine.default_model else ''}Threshold {config.get('threshold', '?')} · up to {config.get('max_tags', '?')} tags")


def main():
    st.html(CSS)
    try:
        engine = get_engine()
    except Exception as error:
        st.error(f"The tagger could not start ({error}). Reload the page; if it keeps failing, the model files may be missing.")
        st.stop()
    sidebar(engine)
    pages = [
        st.Page(tag_page, title="Tag a draft", icon=":material/edit_note:", url_path="tag", default=True),
        st.Page(batch_page, title="Batch review", icon=":material/table_rows:", url_path="batch"),
        st.Page(how_page, title="How it works", icon=":material/help:", url_path="how"),
    ]
    st.navigation(pages).run()


main()
