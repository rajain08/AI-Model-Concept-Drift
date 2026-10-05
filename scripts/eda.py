"""Create compact comparison metrics for the cleaned SMS datasets.

Run after clean_and_split.py:
    python scripts/eda.py
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median

from clean_and_split import PHONE_RE  # single source of truth for phone pattern

URL_RE = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|co\.uk|net|org|info|biz)\b", re.I)
MASK_RE = re.compile(r"XXX")


def length_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"mean": 0, "median": 0, "min": 0, "max": 0}
    return {"mean": mean(lengths), "median": median(lengths), "min": min(lengths), "max": max(lengths)}


def summarize(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    n = len(rows) or 1
    labels = Counter(row["label"] for row in rows)
    return {
        "rows": len(rows),
        "spam_ratio": labels["spam"] / n,
        "label_counts": dict(labels),
        "message_length": length_stats([len(r["text"]) for r in rows]),
        "message_length_by_label": {
            label: length_stats([len(r["text"]) for r in rows if r["label"] == label]) for label in ("spam", "ham")
        },
        "url_frequency": sum(bool(URL_RE.search(r["text"])) for r in rows) / n,
        "phone_frequency": sum(bool(PHONE_RE.search(r["text"])) for r in rows) / n,
        "xxx_mask_frequency": sum(bool(MASK_RE.search(r["text"])) for r in rows) / n,
    }


def main() -> int:
    output_dir = Path("data/processed")
    report = {
        "uci_sms_spam": summarize(output_dir / "uci_messages.csv"),
        # Population after cleaning/dedup but BEFORE the forced 50/50 sample: use this for spam ratio
        "super_sms_population": summarize(output_dir / "super_cleaned_full.csv"),
        "super_sms_sample": summarize(output_dir / "super_messages.csv"),
        "notes": [
            "Spam ratio for Super SMS must be read from super_sms_population; the sample is forced to 0.5.",
            "Super SMS masks phone numbers/amounts as XXX. phone_frequency is therefore not comparable "
            "across datasets unless UCI is masked too (see xxx_mask_frequency).",
            "url_frequency also matches bare domains such as 'txt.com'.",
        ],
    }
    # Text saved to JSON, table format view screenshot saved under /data/eda/eda_table_output.png 
    destination = output_dir / "eda_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())