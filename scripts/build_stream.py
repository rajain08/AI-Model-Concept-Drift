"""Build the simulated drift stream from the cleaned training splits.

Run from the repository root, after clean_and_split.py:
    python scripts/build_stream.py

Messages are replayed in fixed-size windows. Early windows are all UCI
(pre-drift); after that, the share of Super SMS (post-drift) messages grows
on the schedule in configs/drift_stream.yaml.

The datasets have no timestamps, so message order comes from a seeded
shuffle. The drift is real (the datasets are from different years); only
when it happens is simulated.

Guarantees:
  * only training messages are streamed; the run fails if any validation
    or test message gets in
  * every window has the same spam count, so drift shows up in content,
    not in the spam ratio
  * same config + seed = same stream

Writes to data/processed/stream/:
  stream.csv            messages in stream order
  windows.csv           per-window size, post-drift share, spam count
  stream_manifest.json  settings, drift onset, checks, file hashes
"""

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import yaml

DATASETS = {"pre_drift": "uci_sms_spam", "post_drift": "super_sms"}
STREAM_FIELDS = [
    "stream_position", "window", "phase", "post_drift_fraction",
    "message_id", "dataset", "is_post_drift", "label", "text",
]
WINDOW_FIELDS = [
    "window", "phase", "post_drift_fraction", "messages",
    "pre_drift_messages", "post_drift_messages", "spam", "ham",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_ids(path: Path) -> set[str]:
    return set(path.read_text(encoding="utf-8").split())


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    stream = config.get("stream") or {}
    required = ("window_size", "spam_ratio", "source_split", "schedule")
    missing = [key for key in required if stream.get(key) is None]
    if missing:
        raise ValueError(f"{path}: stream settings missing or null: {missing}")
    if stream["source_split"] != "train":
        raise ValueError("stream.source_split must be 'train'; val and test are held out")
    if not 0 < stream["spam_ratio"] < 1:
        raise ValueError("stream.spam_ratio must be between 0 and 1")
    if stream["window_size"] < 10:
        raise ValueError("stream.window_size is too small")
    for step in stream["schedule"]:
        if not 0 <= step["post_drift_fraction"] <= 1 or step["windows"] < 1:
            raise ValueError(f"invalid schedule step: {step}")
    if stream["schedule"][0]["post_drift_fraction"] != 0:
        raise ValueError("the schedule must start with pre-drift-only windows (post_drift_fraction: 0)")
    return config


def check_cleaning_report(processed: Path, config: dict) -> dict:
    """Refuse to build a stream on top of a cleaning run that failed its checks."""
    report = json.loads((processed / "cleaning_report.json").read_text(encoding="utf-8"))
    if not report["leakage"]["passed"]:
        raise RuntimeError("cleaning_report.json says the leakage check failed; rerun clean_and_split.py")
    expected_mask = config["preprocessing"]["mask_uci"]
    if report["mask_uci"] != expected_mask:
        raise RuntimeError(f"mask_uci is {report['mask_uci']} in cleaning_report.json "
                           f"but {expected_mask} in the config; rerun clean_and_split.py to match")
    for dataset in DATASETS.values():
        for split, expected_hash in report["split_file_sha256"][dataset].items():
            path = processed / "splits" / dataset / f"{split}.txt"
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"{path} does not match cleaning_report.json; rerun clean_and_split.py")
    return report


def load_pools(processed: Path, config: dict) -> tuple[dict, dict]:
    """Return {role: {label: [rows]}} from the training split, plus held-out IDs."""
    pools, held_out = {}, {}
    for role, dataset in DATASETS.items():
        csv_path = Path(config["datasets"][role]["path"])
        if not csv_path.is_absolute():
            csv_path = processed / csv_path.name
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = {row["message_id"]: row for row in csv.DictReader(handle)}
        split_dir = processed / "splits" / dataset
        train_ids = read_ids(split_dir / "train.txt")
        held_out[dataset] = read_ids(split_dir / "val.txt") | read_ids(split_dir / "test.txt")
        missing = train_ids - rows.keys()
        if missing:
            raise RuntimeError(f"{len(missing)} {dataset} train IDs are not in {csv_path}")
        pools[role] = {"spam": [], "ham": []}
        for message_id in sorted(train_ids):  # sorted so the shuffle below is reproducible
            pools[role][rows[message_id]["label"]].append(rows[message_id])
    return pools, held_out


def plan_windows(stream_cfg: dict) -> list[dict]:
    """Expand the schedule into per-window message counts."""
    size, ratio = stream_cfg["window_size"], stream_cfg["spam_ratio"]
    spam_total = round(size * ratio)
    plan = []
    for step in stream_cfg["schedule"]:
        fraction = step["post_drift_fraction"]
        post = round(size * fraction)
        post_spam = round(post * ratio)
        pre_spam = spam_total - post_spam
        counts = {
            ("post_drift", "spam"): post_spam,
            ("post_drift", "ham"): post - post_spam,
            ("pre_drift", "spam"): pre_spam,
            ("pre_drift", "ham"): (size - post) - pre_spam,
        }
        if fraction == 0:
            phase = "pre_drift"
        elif fraction == 1:
            phase = "post_drift"
        else:
            phase = "transition"
        for _ in range(step["windows"]):
            plan.append({"phase": phase, "post_drift_fraction": fraction, "counts": counts})
    return plan


def check_budget(plan: list[dict], pools: dict) -> dict:
    """Fail early, with a clear message, if a pool would run out of messages."""
    needed = {}
    for window in plan:
        for key, count in window["counts"].items():
            if count < 0:
                raise ValueError("spam_ratio and schedule give a negative count; lower spam_ratio")
            needed[key] = needed.get(key, 0) + count
    budget = {}
    for (role, label), count in needed.items():
        available = len(pools[role][label])
        budget[f"{role}_{label}"] = {"needed": count, "available": available}
        if count > available:
            raise ValueError(
                f"Not enough {role} {label} messages in the training split: "
                f"the schedule needs {count}, only {available} exist. "
                "Use fewer windows, a smaller window_size, or a lower spam_ratio."
            )
    return budget


def build(plan: list[dict], pools: dict, seed: int) -> list[dict]:
    rng = random.Random(seed)
    queues = {}
    for role, by_label in pools.items():
        for label, rows in by_label.items():
            shuffled = rows[:]
            rng.shuffle(shuffled)
            queues[(role, label)] = shuffled
    stream, position = [], 0
    for number, window in enumerate(plan, start=1):
        batch = []
        for (role, label), count in window["counts"].items():
            batch.extend((role, queues[(role, label)].pop()) for _ in range(count))
        rng.shuffle(batch)  # mix sources and labels within the window
        for role, row in batch:
            position += 1
            stream.append({
                "stream_position": position,
                "window": number,
                "phase": window["phase"],
                "post_drift_fraction": window["post_drift_fraction"],
                "message_id": row["message_id"],
                "dataset": row["dataset"],
                "is_post_drift": int(role == "post_drift"),
                "label": row["label"],
                "text": row["text"],
            })
    return stream


def summarize_windows(stream: list[dict]) -> list[dict]:
    windows = {}
    for row in stream:
        w = windows.setdefault(row["window"], {
            "window": row["window"], "phase": row["phase"],
            "post_drift_fraction": row["post_drift_fraction"], "messages": 0,
            "pre_drift_messages": 0, "post_drift_messages": 0, "spam": 0, "ham": 0,
        })
        w["messages"] += 1
        w["post_drift_messages" if row["is_post_drift"] else "pre_drift_messages"] += 1
        w[row["label"]] += 1
    return [windows[k] for k in sorted(windows)]


def verify(stream: list[dict], windows: list[dict], held_out: dict) -> dict:
    ids = [row["message_id"] for row in stream]
    all_held_out = set().union(*held_out.values())
    checks = {
        "no_duplicate_messages": len(ids) == len(set(ids)),
        "no_validation_or_test_messages": not (set(ids) & all_held_out),
        "equal_spam_per_window": len({w["spam"] for w in windows}) == 1,
        "equal_window_size": len({w["messages"] for w in windows}) == 1,
        "starts_pre_drift_only": windows[0]["post_drift_messages"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Stream checks failed (nothing written): {checks}")
    return checks


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/drift_stream.yaml"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    config = load_config(args.config)
    stream_cfg, seed = config["stream"], config["seed"]
    report = check_cleaning_report(args.processed_dir, config)
    pools, held_out = load_pools(args.processed_dir, config)
    plan = plan_windows(stream_cfg)
    budget = check_budget(plan, pools)

    stream = build(plan, pools, seed)
    windows = summarize_windows(stream)
    checks = verify(stream, windows, held_out)

    out_dir = args.processed_dir / "stream"
    out_dir.mkdir(exist_ok=True)
    write_csv(out_dir / "stream.csv", stream, STREAM_FIELDS)
    write_csv(out_dir / "windows.csv", windows, WINDOW_FIELDS)

    onset = next(w for w in windows if w["post_drift_messages"] > 0)
    manifest = {
        "seed": seed,
        "config": str(args.config),
        "config_sha256": sha256_file(args.config),
        "settings": {k: stream_cfg[k] for k in ("window_size", "spam_ratio", "source_split", "schedule")},
        "mask_uci": report["mask_uci"],
        "ordering": "seeded shuffle within each pool and window; no real timestamps",
        "windows": len(windows),
        "messages": len(stream),
        "drift_onset_window": onset["window"],
        "drift_onset_position": (onset["window"] - 1) * stream_cfg["window_size"] + 1,
        "fully_post_drift_from_window": next(w["window"] for w in windows if w["pre_drift_messages"] == 0),
        "budget": budget,
        "checks": checks,
        "source_split_sha256": {d: report["split_file_sha256"][d]["train"] for d in DATASETS.values()},
        "stream_csv_sha256": sha256_file(out_dir / "stream.csv"),
    }
    (out_dir / "stream_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Built {len(windows)} windows ({len(stream)} messages); "
          f"drift starts at window {manifest['drift_onset_window']}.")
    print(f"{'window':>6} {'phase':<11} {'post%':>5} {'pre':>4} {'post':>4} {'spam':>4} {'ham':>4}")
    for w in windows:
        print(f"{w['window']:>6} {w['phase']:<11} {w['post_drift_fraction']*100:>5.0f} "
              f"{w['pre_drift_messages']:>4} {w['post_drift_messages']:>4} {w['spam']:>4} {w['ham']:>4}")
    print(f"Outputs written to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())