"""Record what the editor finally chose, so the suggestions can be measured and improved later."""

import csv
import io
import json
from datetime import datetime, timezone

from tagger import PROJECT_DIR
from tagger.taxonomy import load_taxonomy

DEFAULT_FEEDBACK_PATH = PROJECT_DIR / "feedback" / "feedback.jsonl"

EXPORT_COLUMNS = ["post_id", "title", "model", "suggested_tags", "scores", "review_status",
                  "review_reason", "final_tags", "editor_changed", "reviewed_at"]


def make_record(post_id, title, suggestion, final_tags):
    """One feedback row: what the model suggested and what the editor kept."""
    approved = set(load_taxonomy().tags)
    final = []
    for tag in final_tags or []:
        if tag in approved and tag not in final:     # only approved tags, each once
            final.append(tag)
    suggested = suggestion.tag_names()
    return {
        "post_id": post_id,
        "title": title,
        "model": suggestion.model,
        "suggested_tags": suggested,
        "scores": {c.tag: round(c.score, 3) for c in suggestion.candidates},
        "review_status": "needs_review" if suggestion.needs_review else "auto_suggested",
        "review_reason": suggestion.review_reason,
        "final_tags": final,
        "editor_changed": set(final) != set(suggested),
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_feedback(record, path=None):
    """Add one record as a line of JSON (the file only ever grows, nothing is overwritten)."""
    path = path or DEFAULT_FEEDBACK_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_feedback(path=None):
    path = path or DEFAULT_FEEDBACK_PATH
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def to_cell(value):
    """Lists become "a; b", score dicts become "Tag: 0.91; Tag2: 0.55" for spreadsheets."""
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{tag}: {score}" for tag, score in value.items())
    if value is None:
        return ""
    return value


def export_csv(records):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(EXPORT_COLUMNS)
    for record in records:
        writer.writerow([to_cell(record.get(column)) for column in EXPORT_COLUMNS])
    return buffer.getvalue().encode("utf-8-sig")     # the BOM makes Excel read UTF-8 correctly


def export_json(records):
    rows = [{column: record.get(column) for column in EXPORT_COLUMNS} for record in records]
    return json.dumps(rows, indent=2, ensure_ascii=False).encode("utf-8")
