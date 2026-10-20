
import argparse
import csv
import json
import random
import re
import time
import urllib.request
from pathlib import Path

MODEL = "llama3.2:3b"
SEED = 29822
DATA_DIR = Path("data/processed")
FIELDS = ["message_id", "label", "prediction", "raw_response"]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def get_examples(uci):
    """Select 15 examples from UCI training data only."""
    rng = random.Random(SEED)
    train = [r for r in uci if r["split"] == "train"]

    spam = [r for r in train if r["label"] == "spam"]
    ham = [r for r in train if r["label"] == "ham"]

    examples = rng.sample(spam, 9) + rng.sample(ham, 6)
    rng.shuffle(examples)

    return examples


def build_prompt(message, examples):
    prompt = (
        "You are an SMS spam classifier.\n"
        "Spam is unsolicited or unwanted messaging, "
        "including fraudulent offers and phishing.\n"
        "Ham is legitimate non-spam messaging.\n"
        "Classify the message as spam or ham.\n"
        "Answer with exactly one word: spam or ham.\n\n"
    )

    for example in examples:
        prompt += (
            f"SMS: {json.dumps(example['text'])}\n"
            f"Label: {example['label']}\n\n"
        )

    prompt += (
        f"SMS: {json.dumps(message)}\n"
        "Label:"
    )

    return prompt


def predict(message, examples):
    payload = {
        "model": MODEL,
        "prompt": build_prompt(message, examples),
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": SEED,
            "num_predict": 128
        }
    }

    request = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(
        request, timeout=300
    ) as response:
        result = json.load(response)

    raw = result["response"].strip()
    match = re.match(r"(?i)^\s*(spam|ham)\b", raw)

    prediction = match.group(1).lower() if match else "invalid"

    return prediction, raw


def calculate_metrics(records):
    tp = tn = fp = fn = invalid = 0

    for row in records:
        actual = row["label"]
        predicted = row["prediction"]

        if predicted == "invalid":
            invalid += 1
        elif actual == "spam" and predicted == "spam":
            tp += 1
        elif actual == "ham" and predicted == "ham":
            tn += 1
        elif actual == "ham" and predicted == "spam":
            fp += 1
        elif actual == "spam" and predicted == "ham":
            fn += 1

    total = len(records)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "total": total,
        "invalid": invalid,
        "accuracy": (tp + tn) / total if total else 0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


def evaluate(name, rows, examples, output_dir):
    output_file = output_dir / f"{name}_predictions.csv"

    if output_file.exists():
        raise FileExistsError(
            f"{output_file} already exists. "
            "Use a new output directory for another run."
        )

    records = []
    start = time.time()

    with output_file.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for index, row in enumerate(rows, start=1):
            prediction, raw = predict(row["text"], examples)

            record = {
                "message_id": row["message_id"],
                "label": row["label"],
                "prediction": prediction,
                "raw_response": raw
            }

            writer.writerow(record)
            f.flush()
            records.append(record)

            print(
                f"{name}: {index}/{len(rows)} | "
                f"actual={row['label']} | "
                f"predicted={prediction}",
                flush=True
            )

    metrics = calculate_metrics(records)
    metrics["runtime_seconds"] = round(
        time.time() - start, 2
    )

    print(f"\n{name} METRICS")
    print(json.dumps(metrics, indent=2))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Number of test messages per dataset; 0 means all"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/baseline")
    )
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit cannot be negative")

    uci = read_csv(DATA_DIR / "uci_messages.csv")
    super_sms = read_csv(DATA_DIR / "super_messages.csv")

    examples = get_examples(uci)

    uci_test = [r for r in uci if r["split"] == "test"]
    super_test = [
        r for r in super_sms if r["split"] == "test"
    ]

    if not uci_test or not super_test:
        raise ValueError("One or both test sets are empty.")

    if args.limit:
        uci_test = uci_test[:args.limit]
        super_test = super_test[:args.limit]

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    summary = {
        "model": MODEL,
        "runtime": "Ollama",
        "seed": SEED,
        "shots": 15,
        "shot_message_ids": [
            r["message_id"] for r in examples
        ],
        "temperature": 0,
        "num_predict": 128,
        "test_limit": args.limit,
        "note": (
            "Local paper-inspired baseline. "
            "Not an exact replication until settings "
            "and datasets are fully matched."
        ),
        "results": {}
    }

    summary["results"]["uci"] = evaluate(
        "uci", uci_test, examples, args.output_dir
    )

    summary["results"]["super_sms"] = evaluate(
        "super_sms", super_test,
        examples, args.output_dir
    )

    with (
        args.output_dir / "metrics.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nFinished!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
