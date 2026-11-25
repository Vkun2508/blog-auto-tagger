import csv
import io
import json

from tagger.feedback import EXPORT_COLUMNS, append_feedback, export_csv, export_json, load_feedback, make_record
from tagger.suggest import Suggestion, TagScore


def make_suggestion():
    python = TagScore("Python", 0.91, 0.5)
    api = TagScore("API Development", 0.55, 0.5)
    return Suggestion([python, api], [python, api, TagScore("Security", 0.3, 0.5)],
                      False, None, "tfidf", 120, "Title", "Body")


def test_record_filters_final_tags_and_flags_change():
    record = make_record(7, "Title", make_suggestion(), ["Python", "Not A Tag", "Security"])
    assert record["final_tags"] == ["Python", "Security"]
    assert record["editor_changed"] is True
    assert record["review_status"] == "auto_suggested"
    assert record["scores"]["Python"] == 0.91

    unchanged = make_record(7, "Title", make_suggestion(), ["API Development", "Python"])
    assert unchanged["editor_changed"] is False


def test_exports_have_exact_columns(tmp_path):
    record = make_record(7, "Title", make_suggestion(), ["Python"])
    path = tmp_path / "feedback.jsonl"
    append_feedback(record, path)
    records = load_feedback(path)

    csv_bytes = export_csv(records)
    assert csv_bytes.startswith(b"\xef\xbb\xbf")          # UTF-8 BOM for Excel
    rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
    assert rows[0] == EXPORT_COLUMNS
    row = dict(zip(rows[0], rows[1]))
    assert row["suggested_tags"] == "Python; API Development"
    assert row["scores"] == "Python: 0.91; API Development: 0.55; Security: 0.3"

    data = json.loads(export_json(records))
    assert list(data[0].keys()) == EXPORT_COLUMNS
    assert data[0]["final_tags"] == ["Python"]
