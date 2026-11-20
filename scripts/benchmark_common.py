"""Shared scoring and integrity checks for the Step 7 benchmark."""

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

LABELS = ("ham", "spam")
OUTCOMES = (*LABELS, "invalid")
DATASETS = {"uci_sms_spam": "uci_messages.csv", "super_sms": "super_messages.csv"}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, fields=None):
    fields = fields or (list(rows[0]) if rows else ["status"])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite_number(value, name, minimum=None, maximum=None):
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum) or (
        maximum is not None and number > maximum
    ):
        raise ValueError(f"Invalid {name}: {value!r}")
    return number


def spam_score(row):
    """Accept the team's LoRA probability column without inventing confidence."""
    values = [finite_number(row[key], key, 0, 1) for key in ("spam_score", "spam_probability")
              if row.get(key) not in (None, "")]
    if len(values) == 2 and values[0] != values[1]:
        raise ValueError("spam_score and spam_probability disagree")
    return values[0] if values else None


def validate_predictions(records, expected, *, stream=False, allow_prefix=False):
    """Require exact coverage, labels, unique IDs and (for streams) causal order."""
    expected_map = {row["message_id"]: row for row in expected}
    if len(expected_map) != len(expected):
        raise ValueError("Duplicate expected message IDs")
    seen = set()
    for index, row in enumerate(records):
        mid = row.get("message_id")
        if mid in seen or mid not in expected_map:
            raise ValueError(f"Duplicate or unexpected prediction ID: {mid}")
        seen.add(mid)
        source = expected_map[mid]
        if row.get("label") != source["label"] or row.get("prediction") not in OUTCOMES:
            raise ValueError(f"Invalid prediction or mismatched label: {mid}")
        if stream:
            if index >= len(expected) or mid != expected[index]["message_id"]:
                raise ValueError(f"Prediction order differs from stream at {mid}")
            for field in ("stream_position", "window", "phase", "dataset"):
                if str(row.get(field)) != str(source[field]):
                    raise ValueError(f"Prediction {field} differs from stream at {mid}")
            if finite_number(row.get("post_drift_fraction"), "post_drift_fraction", 0, 1) != float(source["post_drift_fraction"]):
                raise ValueError(f"Prediction source fraction differs at {mid}")
        for field in ("latency_seconds", "ollama_eval_seconds"):
            if row.get(field) not in (None, ""):
                finite_number(row[field], field, 0)
        spam_score(row)
    if not allow_prefix and seen != set(expected_map):
        raise ValueError(f"Incomplete predictions: {len(records)}/{len(expected)}")
    return records


def score(records):
    """Invalid outputs are incorrect; spam abstentions count as false negatives."""
    if not records:
        raise ValueError("Cannot score an empty prediction set")
    counts = Counter((row["label"], row["prediction"]) for row in records)
    if any(actual not in LABELS or predicted not in OUTCOMES for actual, predicted in counts):
        raise ValueError("Unexpected label or prediction")
    tp, tn = counts["spam", "spam"], counts["ham", "ham"]
    fp = counts["ham", "spam"]
    fn = counts["spam", "ham"] + counts["spam", "invalid"]
    invalid = counts["spam", "invalid"] + counts["ham", "invalid"]
    actual_ham = sum(counts["ham", prediction] for prediction in OUTCOMES)
    actual_spam = len(records) - actual_ham
    result = {
        "messages": len(records), "accuracy": (tp + tn) / len(records),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "invalid": invalid, "invalid_rate": invalid / len(records),
        "actual_ham": actual_ham, "actual_spam": actual_spam,
        "false_positive_rate": fp / actual_ham if actual_ham else None,
        "false_positive_rate_unavailable_reason": None if actual_ham else "No actual ham messages",
        "confusion_matrix": [[counts[actual, pred] for pred in OUTCOMES] for actual in LABELS],
        "roc_auc": None, "pr_auc": None, "pr_auc_unavailable_reason": None,
    }
    scores = [spam_score(row) for row in records]
    if all(value is None for value in scores):
        result["pr_auc_unavailable_reason"] = "Label-only run: no continuous spam scores"
    elif any(value is None for value in scores):
        result["pr_auc_unavailable_reason"] = "Continuous spam scores missing for some messages"
    elif not actual_spam:
        result["pr_auc_unavailable_reason"] = "No actual spam messages"
    else:
        from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
        actual = [r["label"] == "spam" for r in records]
        precision_curve, recall_curve, _ = precision_recall_curve(actual, scores)
        # Trapezoidal area, explicitly distinct from average precision (AP).
        result["pr_auc"] = float(auc(recall_curve, precision_curve))
        if actual_ham:
            result["roc_auc"] = float(roc_auc_score(actual, scores))
    for source, target in (("latency_seconds", "inference_wall_seconds"), ("ollama_eval_seconds", "model_generation_seconds")):
        values = [row.get(source) for row in records]
        available = all(value not in (None, "") for value in values)
        times = [finite_number(value, source, 0) for value in values] if available else []
        result[target] = sum(times) if available else None
        if source == "latency_seconds":
            result["mean_latency_seconds"] = statistics.mean(times) if available else None
            result["p95_latency_seconds"] = sorted(times)[math.ceil(0.95 * len(times)) - 1] if available else None
    return result


def aggregate(values):
    values = [value for value in values if value is not None]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None}


def retention(before, after):
    result = {}
    for metric in ("accuracy", "f1"):
        result[f"{metric}_retention_pct"] = 100 * after[metric] / before[metric] if before[metric] else None
        result[f"{metric}_change_pp"] = 100 * (after[metric] - before[metric])
    return result


def load_inputs(processed, stream_dir):
    processed, stream_dir = Path(processed), Path(stream_dir)
    report = read_json(processed / "cleaning_report.json")
    manifest = read_json(stream_dir / "stream_manifest.json")
    stream = read_csv(stream_dir / "stream.csv")
    if report["leakage"]["passed"] is not True or not manifest["checks"] or not all(v is True for v in manifest["checks"].values()):
        raise ValueError("Input integrity checks failed")
    if report["mask_uci"] != manifest["mask_uci"]:
        raise ValueError("Stream and cleaning report use different UCI masking settings")
    if manifest["stream_csv_sha256"] != sha256(stream_dir / "stream.csv") or manifest["messages"] != len(stream):
        raise ValueError("Stream hash or row count differs from manifest")
    sources, hashes, texts = {}, {}, set()
    for dataset, filename in DATASETS.items():
        rows = read_csv(processed / filename)
        hashes[dataset] = sha256(processed / filename)
        for split in ("train", "val", "test"):
            path = processed / "splits" / dataset / f"{split}.txt"
            expected_hash = report["split_file_sha256"][dataset][split]
            if sha256(path) != expected_hash:
                raise ValueError(f"Split hash mismatch: {path}")
            ids = path.read_text(encoding="utf-8").split()
            if len(ids) != len(set(ids)) or set(ids) != {r["message_id"] for r in rows if r["split"] == split}:
                raise ValueError(f"Split membership mismatch: {path}")
            if split == "train" and manifest["source_split_sha256"][dataset] != expected_hash:
                raise ValueError(f"Stream uses different {dataset} split")
        for row in rows:
            mid = row["message_id"]
            normalized = " ".join(row["text"].split()).casefold()
            if mid in sources or normalized in texts:
                raise ValueError(f"Duplicate source ID or normalized text: {mid}")
            if row["dataset"] != dataset or row["label"] not in LABELS or row["split"] not in ("train", "val", "test"):
                raise ValueError(f"Invalid source row: {mid}")
            sources[mid] = row
            texts.add(normalized)
    seen = set()
    windows = {}
    for pos, row in enumerate(stream, 1):
        mid = row["message_id"]
        source = sources.get(mid)
        if mid in seen or not source or source["split"] != "train" or int(row["stream_position"]) != pos:
            raise ValueError(f"Invalid stream membership/order: {mid}")
        if int(row["window"]) != (pos - 1) // manifest["settings"]["window_size"] + 1:
            raise ValueError(f"Noncontiguous stream window at {mid}")
        for field in ("text", "label", "dataset"):
            if row[field] != source[field]:
                raise ValueError(f"Stream {field} differs from source: {mid}")
        seen.add(mid)
        windows.setdefault(int(row["window"]), []).append(row)
    size = manifest["settings"]["window_size"]
    schedule = [step["post_drift_fraction"] for step in manifest["settings"]["schedule"] for _ in range(step["windows"])]
    if list(windows) != list(range(1, len(schedule) + 1)) or len(windows) != manifest["windows"]:
        raise ValueError("Stream windows differ from schedule")
    for window, rows in windows.items():
        fraction = schedule[window - 1]
        phase = "pre_drift" if fraction == 0 else "post_drift" if fraction == 1 else "transition"
        if len(rows) != size or any(float(r["post_drift_fraction"]) != fraction or r["phase"] != phase for r in rows):
            raise ValueError(f"Invalid window {window}")
        if sum(r["dataset"] == "super_sms" for r in rows) != round(size * fraction) or sum(r["label"] == "spam" for r in rows) != round(size * manifest["settings"]["spam_ratio"]):
            raise ValueError(f"Window composition mismatch: {window}")
    onset = next(r for r in stream if float(r["post_drift_fraction"]) > 0)
    if int(onset["stream_position"]) != manifest["drift_onset_position"] or int(onset["window"]) != manifest["drift_onset_window"]:
        raise ValueError("Drift onset differs from manifest")
    return stream, list(sources.values()), manifest, hashes
