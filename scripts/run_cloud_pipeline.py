"""Run the repository stream through the live cloud prototype.

The API service remains responsible for prediction, online scoring, and durable
event storage. This command supplies the fixed stream, exports the service
results in the repository's standard formats, and runs the existing detector
and optional fine-tuning stages.

Examples:
    python scripts/run_cloud_pipeline.py --limit 100 --skip-detectors
    python scripts/run_cloud_pipeline.py --output-dir results/cloud_pipeline
    python scripts/run_cloud_pipeline.py --run-adaptation --output-dir results/cloud_pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STREAM = ROOT / "data" / "processed" / "stream" / "stream.csv"
DEFAULT_DB = ROOT / "results" / "cloud_prototype.db"
PREDICTION_FIELDS = [
    "stream_position", "window", "phase", "post_drift_fraction",
    "message_id", "dataset", "label", "prediction", "raw_response",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

# JSON request to URL 
def request_json(url: str, payload: Any | None = None) -> dict[str, Any]:
    data = None
    headers = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot reach cloud service at {url}: {error}") from error

# Post a batch of events to the cloud service, checking health and processing results
def post_stream(service_url: str, rows: list[dict[str, str]], batch_size: int) -> None:
    health = request_json(f"{service_url}/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"Cloud service is not healthy: {health}")
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        events = [
            {
                "id": row["message_id"],
                "text": row["text"],
                "label": row["label"],
                "stream_position": row["stream_position"],
                "window": row["window"],
                "phase": row["phase"],
                "post_drift_fraction": row["post_drift_fraction"],
                "dataset": row["dataset"],
            }
            for row in batch
        ]
        result = request_json(f"{service_url}/events", {"events": events})
        if result.get("processed") != len(batch):
            raise RuntimeError(f"Cloud service processed an incomplete batch: {result}")
        print(f"Submitted {min(start + len(batch), len(rows))}/{len(rows)} events", flush=True)

# Export the predictions and window metrics from the cloud database to CSV files
def export_results(
    db_path: Path,
    stream: list[dict[str, str]],
    output_dir: Path,
) -> tuple[Path, Path]:
    if not db_path.exists():
        raise FileNotFoundError(f"Cloud database not found: {db_path}")
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    stored = {
        row["id"]: dict(row)
        for row in db.execute(
            "SELECT id, prediction, raw_response FROM events"
        )
    }
    missing = [row["message_id"] for row in stream if row["message_id"] not in stored]
    if missing:
        raise RuntimeError(f"Cloud database is missing {len(missing)} stream predictions")

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for row in stream:
            saved = stored[row["message_id"]]
            writer.writerow({
                **{field: row[field] for field in PREDICTION_FIELDS[:7]},
                "prediction": saved["prediction"],
                "raw_response": saved["raw_response"],
            })

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in stream:
        saved = stored[row["message_id"]]
        grouped[int(row["window"])].append({**row, "prediction": saved["prediction"]})

    metrics_path = output_dir / "window_metrics.csv"
    metric_fields = [
        "window", "phase", "post_drift_fraction", "messages",
        "accuracy", "precision", "recall", "f1", "invalid",
        "tn", "fp", "fn", "tp",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        for window in sorted(grouped):
            rows = grouped[window]
            tp = tn = fp = fn = invalid = 0
            for row in rows:
                prediction = row["prediction"].lower()
                if prediction not in {"ham", "spam"}:
                    invalid += 1
                elif row["label"] == "spam" and prediction == "spam":
                    tp += 1
                elif row["label"] == "ham" and prediction == "ham":
                    tn += 1
                elif row["label"] == "ham" and prediction == "spam":
                    fp += 1
                else:
                    fn += 1
            total = len(rows)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            writer.writerow({
                "window": window,
                "phase": rows[0]["phase"],
                "post_drift_fraction": rows[0]["post_drift_fraction"],
                "messages": total,
                "accuracy": (tp + tn) / total if total else 0.0,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
                if precision + recall else 0.0,
                "invalid": invalid, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            })
    return prediction_path, metrics_path

# Run the drift detector script on the predictions and stream, checking for output metrics
def run_detector(
    prediction_path: Path,
    stream_path: Path,
    metrics_path: Path,
    output_dir: Path,
) -> None:
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "detect_drift.py"),
            "--stream-path", str(stream_path),
            "--predictions-path", str(prediction_path),
            "--output-dir", str(output_dir),
        ],
        check=True,
    )
    if not metrics_path.exists():
        raise RuntimeError("Drift detector completed without window metrics")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stream-path", type=Path, default=DEFAULT_STREAM)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    parser.add_argument("--service-url", default="http://localhost:8080")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "cloud_pipeline")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-detectors", action="store_true")
    parser.add_argument("--run-adaptation", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("--batch-size must be positive and --limit cannot be negative")

    stream = read_csv(args.stream_path)
    selected = stream[:args.limit] if args.limit else stream
    if not selected:
        raise ValueError("The stream is empty")
    started = time.perf_counter()
    post_stream(args.service_url.rstrip("/"), selected, args.batch_size)
    prediction_path, metrics_path = export_results(args.db_path, selected, args.output_dir)

    detector_dir = args.output_dir / "drift_detection"
    if args.skip_detectors:
        print("Skipped existing ADWIN/KS/PSI/MMD detector stage")
    elif args.limit:
        raise ValueError("--limit requires --skip-detectors because detectors need the full stream")
    else:
        run_detector(prediction_path, args.stream_path, metrics_path, detector_dir)

    if args.run_adaptation:
        if args.limit or args.skip_detectors:
            raise ValueError("--run-adaptation requires the full stream and detector stage")
        subprocess.run(
            [
                sys.executable, str(ROOT / "scripts" / "train_adapted.py"),
                "--predictions-path", str(prediction_path),
                "--window-metrics-path", str(metrics_path),
                "--alerts-path", str(detector_dir / "alerts.csv"),
                "--output-dir", str(args.output_dir / "fine_tuning"),
            ],
            check=True,
        )

    summary = request_json(f"{args.service_url.rstrip('/')}/metrics")
    (args.output_dir / "cloud_metrics.json").write_text(
        json.dumps({
            "stream_messages": len(selected),
            "service_url": args.service_url,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "service_metrics": summary,
            "predictions": str(prediction_path),
            "window_metrics": str(metrics_path),
            "detectors_run": not args.skip_detectors,
            "adaptation_run": args.run_adaptation,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
