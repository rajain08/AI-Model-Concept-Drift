import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from evaluate_baseline import MODEL, build_prompt


STREAM = Path("data/processed/stream/stream.csv")
MANIFEST = Path("data/processed/stream/stream_manifest.json")
UCI = Path("data/processed/uci_messages.csv")
SUPER = Path("data/processed/super_messages.csv")

PRED_FIELDS = [
    "stream_position", "window", "phase",
    "post_drift_fraction", "message_id", "dataset",
    "label", "prediction", "raw_response",
    "latency_seconds", "ollama_eval_seconds"
]


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the data preparation scripts first."
        )
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def validate_inputs():
    stream = read_csv(STREAM)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_rows = read_csv(UCI) + read_csv(SUPER)

    if sha256(STREAM) != manifest["stream_csv_sha256"]:
        raise ValueError("Stream hash does not match its manifest.")

    if len(stream) != manifest["messages"]:
        raise ValueError("Stream row count does not match manifest.")

    if not all(manifest["checks"].values()):
        raise ValueError("Stream manifest contains failed integrity checks.")

    ids = [r["message_id"] for r in stream]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate stream message IDs detected.")

    split_map = {r["message_id"]: r["split"] for r in source_rows}

    for pos, row in enumerate(stream, start=1):
        if int(row["stream_position"]) != pos:
            raise ValueError(f"Stream position mismatch at row {pos}.")
        if split_map.get(row["message_id"]) != "train":
            raise ValueError(
                f"Non-training message in stream: {row['message_id']}"
            )
        if row["label"] not in ("spam", "ham"):
            raise ValueError(f"Invalid label: {row['label']}")

    onset = next(
        int(r["stream_position"]) for r in stream
        if float(r["post_drift_fraction"]) > 0
    )
    if onset != manifest["drift_onset_position"]:
        raise ValueError("Drift onset does not match manifest.")

    print(
        f"INPUT PASS: {len(stream)} messages, "
        f"{len(set(r['window'] for r in stream))} windows, "
        f"drift onset {onset}"
    )
    return stream, source_rows, manifest


def choose_examples(source_rows, stream_ids, seed):
    """Use training examples NOT present in the evaluation stream."""
    rng = random.Random(seed)
    examples = []

    for label, count in [("spam", 9), ("ham", 6)]:
        pool = [
            r for r in source_rows
            if r["dataset"] == "uci_sms_spam"
            and r["split"] == "train"
            and r["label"] == label
            and r["message_id"] not in stream_ids
        ]

        if len(pool) < count:
            raise ValueError(f"Not enough non-stream {label} examples.")

        examples.extend(rng.sample(pool, count))

    rng.shuffle(examples)
    return examples


def predict(text, examples, seed):
    payload = {
        "model": MODEL,
        "prompt": build_prompt(text, examples),
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": seed,
            "num_predict": 128
        }
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    for attempt in range(3):
        try:
            start = time.perf_counter()
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.load(response)

            elapsed = time.perf_counter() - start
            raw = result.get("response", "").strip()
            match = re.match(r"(?i)^\s*(spam|ham)\b", raw)
            prediction = match.group(1).lower() if match else "invalid"

            eval_seconds = result.get("eval_duration", 0) / 1e9
            return prediction, raw, elapsed, eval_seconds

        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def calculate_metrics(rows):
    tp = tn = fp = fn = invalid = correct = 0

    for row in rows:
        actual = row["label"]
        pred = row["prediction"]

        if pred == actual:
            correct += 1
        if pred == "invalid":
            invalid += 1
        if actual == "spam" and pred == "spam":
            tp += 1
        if actual == "ham" and pred == "ham":
            tn += 1
        if actual == "ham" and pred == "spam":
            fp += 1
        if actual == "spam" and pred != "spam":
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "messages": len(rows),
        "accuracy": correct / len(rows) if rows else 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "invalid": invalid,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def write_window_metrics(rows, path):
    grouped = defaultdict(list)
    for r in rows:
        grouped[int(r["window"])].append(r)

    output = []
    for window in sorted(grouped):
        group = grouped[window]
        output.append({
            "window": window,
            "phase": group[0]["phase"],
            "post_drift_fraction": group[0]["post_drift_fraction"],
            **calculate_metrics(group)
        })

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def gpu_name():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() or "Not detected"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "Not detected"


def run_seed(seed, args, stream, source_rows, manifest):
    limited = args.limit > 0
    selected = stream[:args.limit] if limited else stream

    folder_name = f"seed_{seed}"
    if limited:
        folder_name += f"_smoke{args.limit}"

    output = args.output_dir / folder_name
    output.mkdir(parents=True, exist_ok=True)

    example_rows = choose_examples(
        source_rows,
        {r["message_id"] for r in stream},
        seed
    )

    config = {
        "seed": seed,
        "model": MODEL,
        "prompt_examples": [r["message_id"] for r in example_rows],
        "stream_sha256": sha256(STREAM),
        "stream_manifest": manifest,
        "limit": args.limit,
        "git_commit": git_commit(),
        "gpu_detected": gpu_name(),
        "note": (
            "Training-stream diagnostic, not held-out test accuracy. "
            "Few-shot examples excluded from the entire stream."
        )
    }

    config_path = output / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        keys = ["seed", "model", "prompt_examples", "stream_sha256", "limit"]
        if any(old.get(k) != config[k] for k in keys):
            raise ValueError("Existing run configuration differs.")
    else:
        write_json(config_path, config)

    pred_path = output / "predictions.csv"

    # Resume from already completed predictions after interruptions.
    completed = read_csv(pred_path) if pred_path.exists() else []

    if len(completed) > len(selected):
        raise ValueError("More predictions than selected messages.")

    for index, record in enumerate(completed):
        source = selected[index]
        if (
            record["message_id"] != source["message_id"]
            or record["label"] != source["label"]
            or int(record["stream_position"]) != int(source["stream_position"])
        ):
            raise ValueError("Prediction order or labels do not match stream.")

    print(f"Seed {seed}: {len(completed)}/{len(selected)} already completed")

    with pred_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PRED_FIELDS)

        if not completed:
            writer.writeheader()

        for index in range(len(completed), len(selected)):
            row = selected[index]
            pred, raw, elapsed, eval_seconds = predict(
                row["text"], example_rows, seed
            )

            record = {
                "stream_position": row["stream_position"],
                "window": row["window"],
                "phase": row["phase"],
                "post_drift_fraction": row["post_drift_fraction"],
                "message_id": row["message_id"],
                "dataset": row["dataset"],
                "label": row["label"],
                "prediction": pred,
                "raw_response": raw,
                "latency_seconds": round(elapsed, 4),
                "ollama_eval_seconds": round(eval_seconds, 4)
            }

            writer.writerow(record)
            f.flush()

            if (index + 1) % 50 == 0 or index + 1 == len(selected):
                print(f"Seed {seed}: {index + 1}/{len(selected)}")

    predictions = read_csv(pred_path)
    if len(predictions) != len(selected):
        raise ValueError("Incomplete prediction coverage.")

    metrics = calculate_metrics(predictions)
    metrics["seed"] = seed
    metrics["model"] = MODEL
    metrics["training_stream_only"] = True
    metrics["inference_wall_seconds"] = round(
        sum(float(r["latency_seconds"]) for r in predictions), 2
    )
    metrics["inference_wall_hours"] = round(
        metrics["inference_wall_seconds"] / 3600, 4
    )
    metrics["ollama_eval_seconds"] = round(
        sum(float(r["ollama_eval_seconds"]) for r in predictions), 2
    )
    metrics["gpu_hours_measured"] = None

    write_window_metrics(predictions, output / "window_metrics.csv")

    if not args.skip_detectors:
        if limited:
            raise ValueError(
                "Drift detection requires the full stream. "
                "Use --skip-detectors for smoke tests."
            )

        print("Running ADWIN, KS, PSI and MMD...")
        subprocess.run([
            sys.executable, "scripts/detect_drift.py",
            "--stream-path", str(STREAM),
            "--predictions-path", str(pred_path),
            "--output-dir", str(output / "drift_detection"),
            "--seed", str(seed)
        ], check=True)

        metrics["drift_detection_completed"] = True
    else:
        metrics["drift_detection_completed"] = False

    write_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[29822]
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-detectors", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/harness")
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be zero or positive")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("Seeds must be unique")
    if args.limit and not args.skip_detectors:
        parser.error("Use --skip-detectors when --limit is set")

    stream, source_rows, manifest = validate_inputs()
    summaries = [
        run_seed(seed, args, stream, source_rows, manifest)
        for seed in args.seeds
    ]

    aggregate = {
        "seeds": args.seeds,
        "runs": len(summaries),
        "metrics": {}
    }

    for field in ("accuracy", "precision", "recall", "f1"):
        values = [r[field] for r in summaries]
        aggregate["metrics"][field] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None
        }

    aggregate_name = (
        f"aggregate_smoke{args.limit}.json"
        if args.limit else "aggregate.json"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / aggregate_name, aggregate)
    print("\nHARNESS COMPLETED")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
