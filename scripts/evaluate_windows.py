
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from evaluate_baseline import (
    MODEL,
    SEED,
    read_csv,
    get_examples,
    predict,
    calculate_metrics,
)

STREAM_PATH = Path("data/processed/stream/stream.csv")
UCI_PATH = Path("data/processed/uci_messages.csv")
SUPER_PATH = Path("data/processed/super_messages.csv")

FIELDS = [
    "stream_position",
    "window",
    "phase",
    "post_drift_fraction",
    "message_id",
    "dataset",
    "label",
    "prediction",
    "raw_response",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of stream messages; 0 means all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/window_baseline"),
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit cannot be negative")

    if not STREAM_PATH.exists():
        raise FileNotFoundError(
            "Run python scripts/build_stream.py first."
        )

    uci = read_csv(UCI_PATH)
    super_sms = read_csv(SUPER_PATH)
    stream = read_csv(STREAM_PATH)

    examples = get_examples(uci)
    example_ids = {r["message_id"] for r in examples}

    # Verify that the stream contains only training data.
    all_rows = uci + super_sms
    split_lookup = {
        r["message_id"]: r["split"] for r in all_rows
    }

    for row in stream:
        message_id = row["message_id"]
        if split_lookup.get(message_id) != "train":
            raise ValueError(
                f"Stream message is not in training: {message_id}"
            )

    # Do not score messages already shown to Llama as examples.
    overlaps = [
        r for r in stream
        if r["message_id"] in example_ids
    ]

    stream = [
        r for r in stream
        if r["message_id"] not in example_ids
    ]

    if args.limit:
        stream = stream[:args.limit]

    print(f"Prompt examples: {len(examples)}")
    print(f"Excluded prompt-example overlaps: {len(overlaps)}")
    print(f"Messages to evaluate: {len(stream)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_path = args.output_dir / "predictions.csv"

    if prediction_path.exists():
        raise FileExistsError(
            f"{prediction_path} exists. Choose another output folder."
        )

    window_records = defaultdict(list)
    window_info = {}
    start = time.time()

    with prediction_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for index, row in enumerate(stream, start=1):
            prediction, raw = predict(
                row["text"], examples
            )

            result = {
                "stream_position": row["stream_position"],
                "window": row["window"],
                "phase": row["phase"],
                "post_drift_fraction": row["post_drift_fraction"],
                "message_id": row["message_id"],
                "dataset": row["dataset"],
                "label": row["label"],
                "prediction": prediction,
                "raw_response": raw,
            }

            writer.writerow(result)
            f.flush()

            window = int(row["window"])
            window_records[window].append(result)

            window_info[window] = {
                "phase": row["phase"],
                "post_drift_fraction": float(
                    row["post_drift_fraction"]
                ),
            }

            if index % 50 == 0 or index == len(stream):
                print(
                    f"Processed {index}/{len(stream)}",
                    flush=True,
                )

    summaries = []

    for window in sorted(window_records):
        metrics = calculate_metrics(
            window_records[window]
        )

        summaries.append({
            "window": window,
            "phase": window_info[window]["phase"],
            "post_drift_fraction": window_info[window][
                "post_drift_fraction"
            ],
            "messages_scored": metrics["total"],
            "invalid": metrics["invalid"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tn": metrics["tn"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tp": metrics["tp"],
        })

    metrics_path = args.output_dir / "window_metrics.csv"

    with metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summaries[0])
        )
        writer.writeheader()
        writer.writerows(summaries)

    # Plot accuracy and spam F1 by window.
    windows = [r["window"] for r in summaries]
    accuracies = [100 * r["accuracy"] for r in summaries]
    f1_values = [100 * r["f1"] for r in summaries]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        windows, accuracies,
        marker="o", label="Accuracy",
    )
    ax.plot(
        windows, f1_values,
        marker="s", label="Spam F1",
    )

    ax.axvline(
        5.5, linestyle="--",
        label="Simulated drift begins",
    )

    ax.set_xlabel("Stream window (200 messages planned)")
    ax.set_ylabel("Performance (%)")
    ax.set_title(
        "Unadapted Llama 3.2 3B Across Simulated Drift"
    )
    ax.set_ylim(0, 100)
    ax.set_xticks(sorted(windows))
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        args.output_dir / "window_performance.png",
        dpi=200,
    )
    plt.close(fig)

    metadata = {
        "model": MODEL,
        "seed": SEED,
        "runtime": "Ollama",
        "prompt_examples": len(examples),
        "prompt_example_ids": sorted(example_ids),
        "overlap_excluded": len(overlaps),
        "overlap_ids": [
            r["message_id"] for r in overlaps
        ],
        "messages_evaluated": len(stream),
        "test_limit": args.limit,
        "windows_evaluated": len(summaries),
        "runtime_seconds": round(
            time.time() - start, 2
        ),
        "notes": [
            "Training-stream diagnostic, not held-out evaluation.",
            "No adaptation or drift detection performed.",
            "Prompt-example overlaps excluded from scoring.",
            "Message order represents simulated drift, not real time.",
            "Invalid responses count as incorrect in accuracy.",
        ],
    }

    with (
        args.output_dir / "run_info.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nWINDOW RESULTS")
    for row in summaries:
        print(
            f"Window {row['window']:2d} "
            f"({row['phase']}): "
            f"n={row['messages_scored']}, "
            f"accuracy={row['accuracy']:.2%}, "
            f"F1={row['f1']:.2%}"
        )

    print("\nSaved results to:", args.output_dir)
    print("Per-window evaluation completed.")


if __name__ == "__main__":
    main()
