"""Drift-Triggered Full Fine-Tuning Pipeline (Phase 5).

Implements drift-triggered full fine-tuning of Llama 3.2 3B (or a configurable
Hugging Face sequence classification model), triggered by the ADWIN alert on the
error stream (with automatic fallback to calibrated statistical/embedding alerts
or a user-specified trigger window when ADWIN does not fire at default delta).

Supports memory-efficient 100% full-parameter fine-tuning (LOMO / fused layer-wise
backward updates) so all 3.21B parameters can be fine-tuned on a 16GB Apple Silicon
Mac (MPS) as well as standard AdamW on high-VRAM CUDA GPUs.

After fine-tuning on the post-trigger adaptation buffer, evaluates the adapted
model on:
  1. New test split (Post-drift: Super SMS held-out test split, 2,000 messages)
  2. Old test split (Pre-drift: UCI SMS Spam held-out test split, 1,031 messages)

Computes and records:
  - Adapted model weights & tokenizer (`adapted_model/`)
  - Step/epoch training logs (`training_logs.csv`, `training_history.json`)
  - Post-drift F1, accuracy, precision, recall, confusion matrix
  - Pre-drift F1 & Pre-drift F1 retention vs. Phase 2 baseline
  - GPU / accelerator training & inference time (`metrics.json`)
  - Diagnostic plots & Markdown summary (`FINETUNE_REPORT.md`)

Usage:
    # Smoke test on a small sample to verify pipeline end-to-end:
    .venv/bin/python scripts/train_adapted.py --limit 32 --epochs 1 --output-dir results/fine_tuning_smoke

    # Full run with Llama 3.2 3B triggered by drift detection:
    .venv/bin/python scripts/train_adapted.py --output-dir results/fine_tuning
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

MPL_CACHE = Path("/tmp/matplotlib_cache")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
# Allow Apple Silicon MPS to utilize unified memory without premature OOM watermark abort
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import matplotlib.pyplot as plt
import numpy as np

from detect_drift import (
    ADWIN,
    DRIFT_ONSET_MESSAGE,
    DRIFT_ONSET_WINDOW,
    PRE_DRIFT_MESSAGES,
    PRE_DRIFT_WINDOWS,
    get_error_stream,
    load_stream,
)

DEFAULT_MODEL = "unsloth/Llama-3.2-3B"
FALLBACK_UNGATED_MODEL = "unsloth/Llama-3.2-3B"
SEED = 29822

STREAM_PATH = Path("data/processed/stream/stream.csv")
MANIFEST_PATH = Path("data/processed/stream/stream_manifest.json")
UCI_PATH = Path("data/processed/uci_messages.csv")
SUPER_PATH = Path("data/processed/super_messages.csv")
CLEANING_REPORT_PATH = Path("data/processed/cleaning_report.json")

DEFAULT_PREDICTIONS_PATH = Path("results/window_baseline/predictions.csv")
DEFAULT_WINDOW_METRICS_PATH = Path("results/window_baseline/window_metrics.csv")
DEFAULT_ALERTS_PATH = Path("results/drift_detection/alerts.csv")
DEFAULT_BASELINE_METRICS_PATH = Path("results/baseline_full/metrics.json")
DEFAULT_OUTPUT_DIR = Path("results/fine_tuning")

LABEL2ID = {"ham": 0, "spam": 1}
ID2LABEL = {0: "ham", 1: "spam"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_inputs() -> tuple[list[dict], list[dict], list[dict], dict]:
    """Validate stream, manifest, and dataset splits to guarantee zero data leakage."""
    stream = load_stream(STREAM_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    uci_rows = read_csv(UCI_PATH)
    super_rows = read_csv(SUPER_PATH)

    if sha256_file(STREAM_PATH) != manifest["stream_csv_sha256"]:
        raise ValueError("Stream SHA-256 hash does not match stream_manifest.json.")

    if len(stream) != manifest["messages"]:
        raise ValueError("Stream row count does not match stream_manifest.json.")

    if not all(manifest["checks"].values()):
        raise ValueError("stream_manifest.json contains failed integrity checks.")

    if CLEANING_REPORT_PATH.exists():
        report = json.loads(CLEANING_REPORT_PATH.read_text(encoding="utf-8"))
        if not report.get("leakage", {}).get("passed", False):
            raise ValueError("cleaning_report.json indicates failed leakage check.")

    split_map = {r["message_id"]: r["split"] for r in (uci_rows + super_rows)}
    for pos, row in enumerate(stream, start=1):
        if int(row["stream_position"]) != pos:
            raise ValueError(f"Stream position mismatch at row {pos}.")
        if split_map.get(row["message_id"]) != "train":
            raise ValueError(
                f"Data leakage detected: stream message {row['message_id']} is not from train split."
            )

    return stream, uci_rows, super_rows, manifest


def resolve_drift_trigger(
    stream: list[dict],
    predictions_path: Path,
    window_metrics_path: Path,
    alerts_path: Path,
    adwin_delta: float,
    trigger_window_override: int | None = None,
    trigger_position_override: int | None = None,
    strict_adwin: bool = False,
) -> dict:
    """Determine the drift trigger position and window using ADWIN with fallback."""
    if trigger_position_override is not None or trigger_window_override is not None:
        if trigger_position_override is not None:
            pos = trigger_position_override
            matching = [r for r in stream if int(r["stream_position"]) >= pos]
            win = int(matching[0]["window"]) if matching else DRIFT_ONSET_WINDOW
        else:
            win = trigger_window_override
            matching = [r for r in stream if int(r["window"]) >= win]
            pos = int(matching[0]["stream_position"]) if matching else DRIFT_ONSET_MESSAGE

        return {
            "trigger_source": "manual_override",
            "detector": "Manual Override",
            "trigger_window": win,
            "trigger_stream_position": pos,
            "detection_delay_messages": max(0, pos - DRIFT_ONSET_MESSAGE),
            "detection_delay_windows": max(0, win - DRIFT_ONSET_WINDOW),
            "adwin_delta_tested": adwin_delta,
            "adwin_triggered": False,
            "fallback_used": False,
        }

    # 1. Run ADWIN on the sequential error stream
    error_stream = get_error_stream(stream, predictions_path, window_metrics_path)
    adwin = ADWIN(delta=adwin_delta)
    adwin_post_alerts = []
    adwin_pre_alerts = []

    for idx, err in enumerate(error_stream):
        curr_pos = int(stream[idx]["stream_position"])
        curr_win = int(stream[idx]["window"])
        if adwin.add_element(float(err)):
            alert_info = {
                "detector": "ADWIN (Error Stream)",
                "stream_position": curr_pos,
                "window": curr_win,
                "phase": stream[idx]["phase"],
                "post_drift_fraction": float(stream[idx]["post_drift_fraction"]),
                "metric_value": round(adwin.mean, 4),
                "threshold": adwin_delta,
            }
            if curr_pos < DRIFT_ONSET_MESSAGE:
                adwin_pre_alerts.append(alert_info)
            else:
                adwin_post_alerts.append(alert_info)

    if adwin_post_alerts:
        first = adwin_post_alerts[0]
        return {
            "trigger_source": "adwin_primary",
            "detector": first["detector"],
            "trigger_window": int(first["window"]),
            "trigger_stream_position": int(first["stream_position"]),
            "detection_delay_messages": max(0, int(first["stream_position"]) - DRIFT_ONSET_MESSAGE),
            "detection_delay_windows": max(0, int(first["window"]) - DRIFT_ONSET_WINDOW),
            "adwin_delta_tested": adwin_delta,
            "adwin_triggered": True,
            "adwin_false_alarms": len(adwin_pre_alerts),
            "fallback_used": False,
        }

    if strict_adwin:
        raise RuntimeError(
            f"ADWIN did not trigger any post-drift alert at delta={adwin_delta} "
            "and --strict-adwin was set."
        )

    # 2. Fallback to earliest post-drift alert in alerts.csv (e.g., MMD at Window 7 or KS at Window 8)
    if alerts_path.exists():
        alerts = read_csv(alerts_path)
        post_alerts = [
            a for a in alerts
            if str(a.get("is_false_alarm", "")).lower() != "true"
            and int(a["stream_position"]) >= DRIFT_ONSET_MESSAGE
        ]
        if post_alerts:
            post_alerts.sort(key=lambda a: (int(a["stream_position"]), int(a["window"])))
            first = post_alerts[0]
            pos = int(first["stream_position"])
            win = int(first["window"])
            return {
                "trigger_source": "calibrated_detector_fallback",
                "detector": first["detector"],
                "trigger_window": win,
                "trigger_stream_position": pos,
                "detection_delay_messages": max(0, pos - DRIFT_ONSET_MESSAGE),
                "detection_delay_windows": max(0, win - DRIFT_ONSET_WINDOW),
                "adwin_delta_tested": adwin_delta,
                "adwin_triggered": False,
                "fallback_used": True,
                "fallback_reason": (
                    f"ADWIN had 0 alerts at delta={adwin_delta} because unadapted error rate "
                    "did not increase after drift onset; triggered via earliest calibrated "
                    f"drift alert ({first['detector']} at window {win}, pos {pos})."
                ),
            }

    # 3. Final fallback to drift onset window (Window 6, position 1001)
    return {
        "trigger_source": "drift_onset_fallback",
        "detector": "Schedule Drift Onset",
        "trigger_window": DRIFT_ONSET_WINDOW,
        "trigger_stream_position": DRIFT_ONSET_MESSAGE,
        "detection_delay_messages": 0,
        "detection_delay_windows": 0,
        "adwin_delta_tested": adwin_delta,
        "adwin_triggered": False,
        "fallback_used": True,
        "fallback_reason": "No ADWIN alerts or alerts.csv found; defaulted to drift onset.",
    }


def build_training_and_val_sets(
    stream: list[dict],
    uci_rows: list[dict],
    super_rows: list[dict],
    trigger_position: int,
    train_source: str,
    seed: int,
    limit: int = 0,
    val_limit: int = 200,
) -> tuple[list[dict], list[dict], dict]:
    """Select adaptation training examples and validation examples without test leakage."""
    rng = random.Random(seed)

    stream_post_trigger = [
        r for r in stream if int(r["stream_position"]) >= trigger_position
    ]
    stream_pre_trigger = [
        r for r in stream if int(r["stream_position"]) < trigger_position
    ]

    if train_source == "stream_post_trigger":
        train_rows = list(stream_post_trigger)
    elif train_source == "full_post_drift_train":
        train_rows = [r for r in super_rows if r["split"] == "train"]
    elif train_source == "hybrid":
        replay_size = min(len(stream_pre_trigger), max(200, len(stream_post_trigger) // 2))
        replay_sample = rng.sample(stream_pre_trigger, replay_size)
        train_rows = list(stream_post_trigger) + replay_sample
    else:
        raise ValueError(f"Unsupported --train-source: {train_source}")

    rng.shuffle(train_rows)

    # Validation split from held-out val splits (never test!)
    uci_val = [r for r in uci_rows if r["split"] == "val"]
    super_val = [r for r in super_rows if r["split"] == "val"]
    val_rows = uci_val + super_val
    rng.shuffle(val_rows)

    if limit > 0:
        train_rows = train_rows[:limit]
        val_rows = val_rows[: min(limit, len(val_rows))]
    elif val_limit > 0 and len(val_rows) > val_limit:
        val_rows = val_rows[:val_limit]

    # Verify zero overlap with test sets
    test_ids = {
        r["message_id"]
        for r in (uci_rows + super_rows)
        if r["split"] == "test"
    }
    leaked = [r["message_id"] for r in train_rows if r["message_id"] in test_ids]
    if leaked:
        raise RuntimeError(f"Test leakage detected in adaptation training set: {leaked[:5]}")

    stats = {
        "train_source": train_source,
        "train_messages": len(train_rows),
        "train_spam": sum(1 for r in train_rows if r["label"] == "spam"),
        "train_ham": sum(1 for r in train_rows if r["label"] == "ham"),
        "train_post_drift_messages": sum(
            1 for r in train_rows if r.get("dataset") == "super_sms"
        ),
        "train_pre_drift_messages": sum(
            1 for r in train_rows if r.get("dataset") == "uci_sms_spam"
        ),
        "val_messages": len(val_rows),
        "zero_test_leakage_verified": True,
    }
    return train_rows, val_rows, stats


def calculate_classification_metrics(records: list[dict]) -> dict:
    from sklearn.metrics import roc_auc_score

    tp = tn = fp = fn = invalid = 0
    y_true = []
    y_prob = []
    for row in records:
        actual = row["label"].strip().lower()
        pred = row["prediction"].strip().lower()

        if pred == "invalid":
            invalid += 1
        elif actual == "spam" and pred == "spam":
            tp += 1
        elif actual == "ham" and pred == "ham":
            tn += 1
        elif actual == "ham" and pred == "spam":
            fp += 1
        elif actual == "spam" and pred == "ham":
            fn += 1

        if "spam_probability" in row:
            y_true.append(1 if actual == "spam" else 0)
            y_prob.append(float(row["spam_probability"]))

    total = len(records)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    roc_auc = None
    if len(set(y_true)) == 2 and len(y_prob) == len(y_true):
        roc_auc = round(float(roc_auc_score(y_true, y_prob)), 6)

    result = {
        "total": total,
        "invalid": invalid,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if roc_auc is not None:
        result["roc_auc"] = roc_auc
    return result


def calibrate_decision_threshold(records: list[dict]) -> float:
    """Find the probability threshold on validation records that maximizes Spam F1."""
    if not records:
        return 0.5
    probs = sorted({float(r["spam_probability"]) for r in records})
    candidates = [0.5] + probs
    best_th = 0.5
    best_f1 = -1.0
    for th in candidates:
        if th <= 0.0 or th >= 1.0:
            continue
        tp = fp = fn = 0
        for r in records:
            actual_spam = r["label"].strip().lower() == "spam"
            pred_spam = float(r["spam_probability"]) >= th
            if actual_spam and pred_spam:
                tp += 1
            elif not actual_spam and pred_spam:
                fp += 1
            elif actual_spam and not pred_spam:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return round(float(best_th), 6)


def apply_decision_threshold(records: list[dict], threshold: float) -> list[dict]:
    """Update 'prediction' field in records using a calibrated spam probability threshold."""
    updated = []
    for r in records:
        prob = float(r["spam_probability"])
        updated.append({
            **r,
            "prediction": "spam" if prob >= threshold else "ham",
        })
    return updated


def detect_device(requested_device: str | None = None):
    import torch

    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_hardware_description(device) -> dict:
    import torch

    info = {
        "device_type": device.type,
        "torch_version": torch.__version__,
        "gpu_name": "Not detected",
    }
    if device.type == "cuda" and torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
    elif device.type == "mps":
        info["gpu_name"] = "Apple Silicon MPS (Metal Performance Shaders)"
    return info


def sync_device(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def empty_device_cache(device) -> None:
    import torch

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def format_input_text(text: str) -> str:
    return (
        "Classify the following SMS message as ham or spam.\n"
        f"SMS: {text.strip()}"
    )


def train_and_evaluate_pytorch(
    args,
    train_rows: list[dict],
    val_rows: list[dict],
    uci_test: list[dict],
    super_test: list[dict],
    output_dir: Path,
) -> tuple[list[dict], dict, list[dict], list[dict], dict]:
    """Perform full fine-tuning in PyTorch and evaluate on UCI and Super SMS test splits."""
    try:
        import torch
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and Hugging Face `transformers` are required for Phase 5 full fine-tuning.\n"
            "Install them via: `.venv/bin/pip install torch transformers accelerate`"
        ) from exc

    device = detect_device(args.device)
    hw_info = get_hardware_description(device)
    print(f"Accelerator selected: {device} ({hw_info['gpu_name']})")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Choose precision appropriate for the target device
    model_dtype = torch.float32
    if args.bf16 and device.type in ("cuda", "mps"):
        model_dtype = torch.bfloat16
    elif args.fp16 and device.type == "cuda":
        model_dtype = torch.float16

    resolved_model_name = args.model_name
    print(f"Loading tokenizer and model: {resolved_model_name} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_name,
            token=hf_token,
            trust_remote_code=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            resolved_model_name,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            dtype=model_dtype,
            token=hf_token,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    except OSError as exc:
        err_msg = str(exc).lower()
        if (
            ("gated repo" in err_msg or "401" in err_msg or "unauthorized" in err_msg)
            and resolved_model_name != FALLBACK_UNGATED_MODEL
        ):
            print(
                f"   [Note] '{resolved_model_name}' is gated on Hugging Face and requires auth. "
                f"Automatically falling back to ungated Llama 3.2 3B weights: '{FALLBACK_UNGATED_MODEL}' ..."
            )
            resolved_model_name = FALLBACK_UNGATED_MODEL
            args.model_name = resolved_model_name
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_model_name,
                trust_remote_code=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                resolved_model_name,
                num_labels=2,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                dtype=model_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            raise

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Full fine-tuning: ensure all 3.21B parameters are trainable
    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model loaded: {total_params:,} total params | "
        f"{trainable_params:,} trainable params (100% full fine-tuning)"
    )

    empty_device_cache(device)
    model.to(device)
    empty_device_cache(device)

    class SMSDataset(Dataset):
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> dict:
            r = self.rows[idx]
            return {
                "message_id": r["message_id"],
                "text": format_input_text(r["text"]),
                "raw_text": r["text"],
                "label_str": r["label"].strip().lower(),
                "label": LABEL2ID[r["label"].strip().lower()],
            }

    def collate_batch(batch: list[dict]) -> dict:
        texts = [b["text"] for b in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": labels,
            "message_ids": [b["message_id"] for b in batch],
            "label_strs": [b["label_str"] for b in batch],
        }

    # Select optimizer mode:
    # - On MPS (16GB unified memory) or when --optimizer lomo is chosen, use fused layer-wise
    #   in-place backward updates (LOMO: Low-Memory Optimization) via post-accumulate grad hooks.
    #   This updates 100% of the 3.21B parameters during backward() and immediately frees each
    #   layer's gradient tensor, keeping total memory at ~6.6 GB instead of >32 GB!
    optimizer_mode = args.optimizer
    if optimizer_mode == "auto":
        optimizer_mode = "lomo" if device.type == "mps" else "adamw"

    train_loader = DataLoader(
        SMSDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        SMSDataset(val_rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    def evaluate_loader(loader: DataLoader) -> tuple[float, list[dict], float]:
        model.eval()
        total_loss = 0.0
        total_items = 0
        records = []
        sync_device(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                logits = outputs.logits
                probs = torch.softmax(logits.float(), dim=-1)
                preds = torch.argmax(probs, dim=-1)

                bsz = labels.size(0)
                total_loss += float(loss.item()) * bsz
                total_items += bsz

                for i in range(bsz):
                    pred_id = int(preds[i].item())
                    spam_prob = float(probs[i, 1].item())
                    records.append({
                        "message_id": batch["message_ids"][i],
                        "label": batch["label_strs"][i],
                        "prediction": ID2LABEL[pred_id],
                        "spam_probability": round(spam_prob, 6),
                    })

                del input_ids, attention_mask, labels, outputs, logits, probs, preds

        sync_device(device)
        elapsed = time.perf_counter() - t0
        avg_loss = total_loss / max(1, total_items)
        return avg_loss, records, elapsed

    training_logs = []
    global_step = 0
    best_val_f1 = -1.0
    adapted_model_dir = output_dir / "adapted_model"
    adapted_model_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sync_device(device)
    train_start_wall = time.perf_counter()
    active_gpu_train_seconds = 0.0

    if optimizer_mode == "lomo":
        # Fused layer-wise full-parameter fine-tuning (LOMO)
        total_opt_steps = max(1, len(train_loader) * args.epochs)
        warmup_steps = int(total_opt_steps * args.warmup_ratio)
        lomo_state = {
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "last_grad_norm_sq": 0.0,
            "active": True,
        }

        def get_lomo_lr(step_num: int) -> float:
            if warmup_steps > 0 and step_num <= warmup_steps:
                return args.learning_rate * (step_num / max(1, warmup_steps))
            remaining = max(0.0, float(total_opt_steps - step_num))
            decay_span = max(1.0, float(total_opt_steps - warmup_steps))
            return args.learning_rate * (remaining / decay_span)

        def make_lomo_hook(is_head: bool):
            def _hook(param: "torch.Tensor") -> None:
                if not lomo_state["active"] or param.grad is None:
                    return
                with torch.no_grad():
                    g = param.grad
                    g_norm = float(torch.linalg.vector_norm(g.float()).item())
                    lomo_state["last_grad_norm_sq"] += g_norm * g_norm
                    max_norm = lomo_state["max_grad_norm"]
                    if math.isfinite(g_norm) and g_norm > 0:
                        scale = min(1.0, max_norm / (g_norm + 1e-6))
                        # Classification head receives 5x learning rate for rapid adaptation
                        lr_eff = lomo_state["lr"] * (5.0 if is_head else 1.0)
                        if lomo_state["weight_decay"] > 0 and param.ndim >= 2:
                            param.mul_(1.0 - lr_eff * lomo_state["weight_decay"])
                        param.add_(g, alpha=-lr_eff * scale)
                    param.grad = None

            return _hook

        hook_handles = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                is_head = "score" in name or "classifier" in name
                hook_handles.append(
                    param.register_post_accumulate_grad_hook(make_lomo_hook(is_head))
                )

        print(
            f"\nStarting 100% full-parameter fine-tuning (Fused LOMO optimizer) for {args.epochs} epoch(s) | "
            f"{len(train_rows)} train msgs | batch_size={args.batch_size} | lr={args.learning_rate}"
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            micro_steps = 0

            for step_idx, batch in enumerate(train_loader, start=1):
                sync_device(device)
                step_t0 = time.perf_counter()

                global_step += 1
                current_lr = get_lomo_lr(global_step)
                lomo_state["lr"] = current_lr
                lomo_state["last_grad_norm_sq"] = 0.0

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                loss_val = float(loss.item())
                running_loss += loss_val
                micro_steps += 1

                # Backward triggers per-layer fused weight updates and immediately frees layer gradients
                loss.backward()
                grad_norm = math.sqrt(lomo_state["last_grad_norm_sq"])

                del input_ids, attention_mask, labels, outputs, loss

                sync_device(device)
                step_elapsed = time.perf_counter() - step_t0
                active_gpu_train_seconds += step_elapsed

                avg_train_loss = running_loss / max(1, micro_steps)

                if (
                    global_step % args.log_every_steps == 0
                    or step_idx == len(train_loader)
                ):
                    empty_device_cache(device)
                    val_loss, val_records, val_elapsed = evaluate_loader(val_loader)
                    active_gpu_train_seconds += val_elapsed
                    val_metrics = calculate_classification_metrics(val_records)
                    model.train()

                    log_entry = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": round(avg_train_loss, 6),
                        "val_loss": round(val_loss, 6),
                        "val_accuracy": round(val_metrics["accuracy"], 6),
                        "val_f1": round(val_metrics["f1"], 6),
                        "learning_rate": float(f"{current_lr:.6e}"),
                        "grad_norm": round(grad_norm, 4),
                        "elapsed_compute_seconds": round(active_gpu_train_seconds, 2),
                    }
                    training_logs.append(log_entry)
                    print(
                        f"Epoch {epoch}/{args.epochs} | Step {global_step}/{total_opt_steps} | "
                        f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
                        f"val_F1={val_metrics['f1']:.2%} | compute={active_gpu_train_seconds:.1f}s",
                        flush=True,
                    )

                    if val_metrics["f1"] >= best_val_f1:
                        best_val_f1 = val_metrics["f1"]

        for h in hook_handles:
            h.remove()

    else:
        # Standard AdamW optimizer (for high-VRAM CUDA GPUs or smaller models)
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
        total_opt_steps = steps_per_epoch * args.epochs
        warmup_steps = int(total_opt_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_opt_steps,
        )

        print(
            f"\nStarting full fine-tuning (AdamW) for {args.epochs} epoch(s) | "
            f"{len(train_rows)} train msgs | batch_size={args.batch_size} | "
            f"grad_accum={args.grad_accum_steps} | lr={args.learning_rate}"
        )

        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            micro_steps = 0
            optimizer.zero_grad(set_to_none=True)

            for step_idx, batch in enumerate(train_loader, start=1):
                sync_device(device)
                step_t0 = time.perf_counter()

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss_val = float(outputs.loss.item())
                loss = outputs.loss / args.grad_accum_steps
                loss.backward()

                running_loss += loss_val
                micro_steps += 1
                del input_ids, attention_mask, labels, outputs, loss

                is_update_step = (
                    step_idx % args.grad_accum_steps == 0
                    or step_idx == len(train_loader)
                )
                if is_update_step:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        ).item()
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    sync_device(device)
                    step_elapsed = time.perf_counter() - step_t0
                    active_gpu_train_seconds += step_elapsed

                    avg_train_loss = running_loss / max(1, micro_steps)
                    current_lr = scheduler.get_last_lr()[0]

                    if (
                        global_step % args.log_every_steps == 0
                        or step_idx == len(train_loader)
                    ):
                        val_loss, val_records, val_elapsed = evaluate_loader(val_loader)
                        active_gpu_train_seconds += val_elapsed
                        val_metrics = calculate_classification_metrics(val_records)
                        model.train()

                        log_entry = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss": round(avg_train_loss, 6),
                            "val_loss": round(val_loss, 6),
                            "val_accuracy": round(val_metrics["accuracy"], 6),
                            "val_f1": round(val_metrics["f1"], 6),
                            "learning_rate": float(f"{current_lr:.6e}"),
                            "grad_norm": round(grad_norm, 4),
                            "elapsed_compute_seconds": round(active_gpu_train_seconds, 2),
                        }
                        training_logs.append(log_entry)
                        print(
                            f"Epoch {epoch}/{args.epochs} | Step {global_step}/{total_opt_steps} | "
                            f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
                            f"val_F1={val_metrics['f1']:.2%} | compute={active_gpu_train_seconds:.1f}s",
                            flush=True,
                        )

                        if val_metrics["f1"] >= best_val_f1:
                            best_val_f1 = val_metrics["f1"]
                else:
                    sync_device(device)
                    active_gpu_train_seconds += time.perf_counter() - step_t0

    sync_device(device)
    total_train_wall_seconds = time.perf_counter() - train_start_wall

    # Evaluate on held-out Old (Pre-drift UCI) and New (Post-drift Super SMS) test splits
    print("\nEvaluating adapted model on held-out test splits...")
    empty_device_cache(device)
    uci_test_loader = DataLoader(
        SMSDataset(uci_test),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )
    super_test_loader = DataLoader(
        SMSDataset(super_test),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    uci_test_loss, uci_test_preds, uci_eval_sec = evaluate_loader(uci_test_loader)
    super_test_loss, super_test_preds, super_eval_sec = evaluate_loader(super_test_loader)

    # Calibrate decision threshold on the held-out validation set to account for stream class prior (80% ham / 20% spam)
    _, final_val_records, val_cal_sec = evaluate_loader(val_loader)
    calibrated_threshold = calibrate_decision_threshold(final_val_records)
    print(f"Calibrated spam probability threshold on validation split: {calibrated_threshold:.6f}")
    uci_test_preds = apply_decision_threshold(uci_test_preds, calibrated_threshold)
    super_test_preds = apply_decision_threshold(super_test_preds, calibrated_threshold)

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
    elif device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        peak_memory_mb = round(torch.mps.driver_allocated_memory() / (1024**2), 2)

    print(f"Saving adapted model & tokenizer to: {adapted_model_dir} ...")
    model.save_pretrained(adapted_model_dir)
    tokenizer.save_pretrained(adapted_model_dir)

    total_gpu_seconds = active_gpu_train_seconds + uci_eval_sec + super_eval_sec
    compute_stats = {
        **hw_info,
        "optimizer": optimizer_mode,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "trainable_fraction": 1.0,
        "dtype": str(model_dtype).replace("torch.", ""),
        "training_compute_seconds": round(active_gpu_train_seconds, 2),
        "training_wall_seconds": round(total_train_wall_seconds, 2),
        "uci_test_eval_seconds": round(uci_eval_sec, 2),
        "super_sms_test_eval_seconds": round(super_eval_sec, 2),
        "total_gpu_compute_seconds": round(total_gpu_seconds, 2),
        "total_gpu_compute_hours": round(total_gpu_seconds / 3600.0, 6),
        "peak_memory_mb": peak_memory_mb,
        "uci_test_loss": round(uci_test_loss, 6),
        "super_sms_test_loss": round(super_test_loss, 6),
    }

    return training_logs, compute_stats, uci_test_preds, super_test_preds, {
        "adapted_model_dir": str(adapted_model_dir),
    }


def load_baseline_reference(baseline_metrics_path: Path) -> dict | None:
    if not baseline_metrics_path.exists():
        return None
    try:
        return json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def generate_plots(
    training_logs: list[dict],
    pre_drift_metrics: dict,
    post_drift_metrics: dict,
    baseline_ref: dict | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Subplot 1: Training & Validation Loss
    ax1 = axes[0]
    if training_logs:
        steps = [r["global_step"] for r in training_logs]
        train_losses = [r["train_loss"] for r in training_logs]
        val_losses = [r["val_loss"] for r in training_logs]
        ax1.plot(steps, train_losses, marker="o", color="#1f77b4", label="Train Loss")
        ax1.plot(steps, val_losses, marker="s", color="#ff7f0e", label="Val Loss")
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("1. Drift-Triggered Full Fine-Tuning Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Subplot 2: Pre-Drift Retention & Post-Drift F1 Comparison
    ax2 = axes[1]
    categories = ["Pre-Drift (UCI)\nSpam F1", "Post-Drift (Super SMS)\nSpam F1"]
    adapted_vals = [100 * pre_drift_metrics["f1"], 100 * post_drift_metrics["f1"]]

    x = np.arange(len(categories))
    width = 0.35

    if baseline_ref and "results" in baseline_ref:
        base_uci_f1 = 100 * float(baseline_ref["results"]["uci"]["f1"])
        base_super_f1 = 100 * float(baseline_ref["results"]["super_sms"]["f1"])
        ax2.bar(
            x - width / 2,
            [base_uci_f1, base_super_f1],
            width,
            label="Unadapted Baseline (Phase 2)",
            color="#7f7f7f",
        )
        ax2.bar(
            x + width / 2,
            adapted_vals,
            width,
            label="Adapted Model (Phase 5)",
            color="#2ca02c",
        )
    else:
        ax2.bar(x, adapted_vals, width, label="Adapted Model (Phase 5)", color="#2ca02c")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("F1 Score (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("2. Pre-Drift Retention vs. Post-Drift F1")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_markdown_report(summary: dict, report_path: Path) -> None:
    trig = summary["drift_trigger"]
    pre = summary["evaluation"]["pre_drift_uci_test"]
    post = summary["evaluation"]["post_drift_super_sms_test"]
    ret = summary["retention_and_adaptation"]
    comp = summary["compute"]
    data_cfg = summary["training_data"]

    base_pre_f1 = (
        f"{ret['baseline_pre_drift_f1']:.2%}"
        if ret["baseline_pre_drift_f1"] is not None
        else "N/A"
    )
    base_post_f1 = (
        f"{ret['baseline_post_drift_f1']:.2%}"
        if ret["baseline_post_drift_f1"] is not None
        else "N/A"
    )
    retention_ratio = (
        f"{ret['pre_drift_f1_retention_ratio']:.2%}"
        if ret["pre_drift_f1_retention_ratio"] is not None
        else "N/A"
    )
    pre_delta = (
        f"{ret['pre_drift_f1_delta']:+.2%}"
        if ret["pre_drift_f1_delta"] is not None
        else "N/A"
    )
    post_delta = (
        f"{ret['post_drift_f1_delta']:+.2%}"
        if ret["post_drift_f1_delta"] is not None
        else "N/A"
    )

    text = f"""# Phase 5: Drift-Triggered Full Fine-Tuning Report

## 1. Drift Trigger Summary
- **Trigger Source**: `{trig['trigger_source']}` ({trig['detector']})
- **Trigger Window**: Window {trig['trigger_window']} (Stream Position {trig['trigger_stream_position']})
- **Detection Delay**: {trig['detection_delay_messages']} messages ({trig['detection_delay_windows']} windows after Window {DRIFT_ONSET_WINDOW} onset)
- **ADWIN Delta Tested**: {trig['adwin_delta_tested']} (ADWIN fired directly: `{trig['adwin_triggered']}`)
{f"- **Fallback Note**: {trig['fallback_reason']}" if trig.get('fallback_reason') else ""}

## 2. Adaptation Dataset & Full Fine-Tuning Setup
- **Model**: `{summary['model_name']}`
- **Optimizer**: `{comp['optimizer']}` (100% Full Parameter Fine-Tuning)
- **Training Source**: `{data_cfg['train_source']}` ({data_cfg['train_messages']} messages: {data_cfg['train_spam']} spam, {data_cfg['train_ham']} ham)
- **Validation Set**: {data_cfg['val_messages']} held-out validation messages
- **Zero Test Leakage Verified**: `{data_cfg['zero_test_leakage_verified']}`
- **Total / Trainable Parameters**: {comp['total_parameters']:,} / {comp['trainable_parameters']:,} (100% Full Fine-Tuning)
- **Epochs / Batch Size / LR**: {summary['hyperparameters']['epochs']} / {summary['hyperparameters']['batch_size']} / {summary['hyperparameters']['learning_rate']}

## 3. Held-Out Test Split Evaluation (New vs. Old Splits)

| Split | Dataset | Messages | Baseline Spam F1 | Adapted Spam F1 | F1 Delta | Adapted Accuracy | Adapted Precision | Adapted Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **New Test Split (Post-Drift)** | Super SMS (`test`) | {post['total']} | {base_post_f1} | **{post['f1']:.2%}** | {post_delta} | {post['accuracy']:.2%} | {post['precision']:.2%} | {post['recall']:.2%} |
| **Old Test Split (Pre-Drift)** | UCI SMS (`test`) | {pre['total']} | {base_pre_f1} | **{pre['f1']:.2%}** | {pre_delta} | {pre['accuracy']:.2%} | {pre['precision']:.2%} | {pre['recall']:.2%} |

- **Post-Drift Spam F1**: **{post['f1']:.2%}** (TP={post['tp']}, TN={post['tn']}, FP={post['fp']}, FN={post['fn']})
- **Pre-Drift Spam F1**: **{pre['f1']:.2%}** (TP={pre['tp']}, TN={pre['tn']}, FP={pre['fp']}, FN={pre['fn']})
- **Pre-Drift F1 Retention**: **{retention_ratio}** of unadapted baseline F1 ({pre_delta})

## 4. Compute & GPU Time
- **Accelerator**: `{comp['device_type']}` ({comp['gpu_name']})
- **Precision**: `{comp['dtype']}`
- **Training Compute Time**: {comp['training_compute_seconds']:.2f} s (wall time: {comp['training_wall_seconds']:.2f} s)
- **Test Evaluation Time**: {comp['uci_test_eval_seconds'] + comp['super_sms_test_eval_seconds']:.2f} s
- **Total GPU / Compute Time**: {comp['total_gpu_compute_seconds']:.2f} s ({comp['total_gpu_compute_hours']:.6f} hours)
- **Peak Accelerator Memory**: {f"{comp['peak_memory_mb']} MB" if comp['peak_memory_mb'] is not None else "N/A"}

## 5. Generated Artifacts
- `adapted_model/`: Fine-tuned Hugging Face model weights and tokenizer
- `training_logs.csv` & `training_history.json`: Step-by-step optimization and validation logs
- `super_sms_test_predictions.csv`: Predictions on post-drift Super SMS test split
- `uci_test_predictions.csv`: Predictions on pre-drift UCI test split
- `metrics.json`: Full reproducible metrics and metadata
- `adaptation_evaluation_plot.png`: Loss curve and F1 retention bar chart
"""
    report_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Drift-triggered full fine-tuning and evaluation on new/old test splits."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        help="Hugging Face model ID or local path (default: unsloth/Llama-3.2-3B)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face access token for gated repos (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write adapted model, training logs, and test evaluation metrics",
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="Path to stream predictions.csv for ADWIN error stream",
    )
    parser.add_argument(
        "--window-metrics-path",
        type=Path,
        default=DEFAULT_WINDOW_METRICS_PATH,
        help="Fallback window_metrics.csv for reconstructing error stream",
    )
    parser.add_argument(
        "--alerts-path",
        type=Path,
        default=DEFAULT_ALERTS_PATH,
        help="Path to Phase 3 alerts.csv for fallback trigger resolution",
    )
    parser.add_argument(
        "--baseline-metrics-path",
        type=Path,
        default=DEFAULT_BASELINE_METRICS_PATH,
        help="Path to Phase 2 baseline metrics.json for computing pre-drift F1 retention",
    )
    parser.add_argument(
        "--adwin-delta",
        type=float,
        default=0.01,
        help="ADWIN confidence parameter delta",
    )
    parser.add_argument(
        "--trigger-window",
        type=int,
        default=None,
        help="Optional manual override for drift trigger window (1-14)",
    )
    parser.add_argument(
        "--trigger-position",
        type=int,
        default=None,
        help="Optional manual override for drift trigger stream position (1-2800)",
    )
    parser.add_argument(
        "--strict-adwin",
        action="store_true",
        help="Require ADWIN to fire directly without falling back to calibrated alerts",
    )
    parser.add_argument(
        "--train-source",
        choices=["stream_post_trigger", "full_post_drift_train", "hybrid"],
        default="stream_post_trigger",
        help="Training data pool used once drift is triggered",
    )
    parser.add_argument(
        "--optimizer",
        choices=["auto", "lomo", "adamw"],
        default="auto",
        help="Full fine-tuning optimizer: 'lomo' (fused layer-wise backward update for 16GB MPS) or 'adamw'",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-step training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (used with adamw)",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Linear warmup ratio")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--max-length", type=int, default=128, help="Max token sequence length")
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=25,
        help="Log and validate every N optimizer steps",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 precision when supported by CUDA/MPS (default: True)",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_false",
        dest="bf16",
        help="Disable bfloat16 and use float32",
    )
    parser.add_argument("--fp16", action="store_true", help="Use float16 on CUDA")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing to reduce memory usage (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit train/val/test message counts for fast smoke testing (0 = full)",
    )
    parser.add_argument(
        "--val-limit",
        type=int,
        default=200,
        help="Max validation messages per validation check during training (0 = all)",
    )
    parser.add_argument(
        "--recalibrate-only",
        action="store_true",
        help="Recalibrate classification threshold on validation split using saved adapted_model and update test metrics/plots without re-training",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be 0 or positive")

    set_reproducibility(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.recalibrate_only:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        print(f"Recalibrating existing Phase 5 artifacts in: {args.output_dir} ...")
        metrics_path = args.output_dir / "metrics.json"
        existing_summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        uci_pred_path = args.output_dir / "uci_test_predictions.csv"
        super_pred_path = args.output_dir / "super_sms_test_predictions.csv"
        logs_csv_path = args.output_dir / "training_logs.csv"

        uci_preds = read_csv(uci_pred_path)
        super_preds = read_csv(super_pred_path)
        training_logs = [
            {
                "epoch": int(r["epoch"]),
                "global_step": int(r["global_step"]),
                "train_loss": float(r["train_loss"]),
                "val_loss": float(r["val_loss"]),
                "val_accuracy": float(r["val_accuracy"]),
                "val_f1": float(r["val_f1"]),
                "learning_rate": float(r["learning_rate"]),
                "grad_norm": float(r["grad_norm"]),
                "elapsed_compute_seconds": float(r["elapsed_compute_seconds"]),
            }
            for r in read_csv(logs_csv_path)
        ] if logs_csv_path.exists() else []

        stream, uci_rows, super_rows, _ = validate_inputs()
        trigger_pos = int(existing_summary["drift_trigger"]["trigger_stream_position"])
        _, val_rows, _ = build_training_and_val_sets(
            stream=stream,
            uci_rows=uci_rows,
            super_rows=super_rows,
            trigger_position=trigger_pos,
            train_source=existing_summary["training_data"]["train_source"],
            seed=args.seed,
            limit=args.limit,
            val_limit=args.val_limit,
        )

        device = detect_device(args.device)
        adapted_model_dir = args.output_dir / "adapted_model"
        tokenizer = AutoTokenizer.from_pretrained(adapted_model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            adapted_model_dir,
            dtype=torch.bfloat16 if device.type in ("cuda", "mps") else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        model.eval()

        val_records = []
        with torch.no_grad():
            for r in val_rows:
                enc = tokenizer(
                    format_input_text(r["text"]),
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                logits = model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1)
                spam_prob = float(probs[0, 1].item())
                val_records.append({
                    "message_id": r["message_id"],
                    "label": r["label"].strip().lower(),
                    "prediction": "spam" if spam_prob >= 0.5 else "ham",
                    "spam_probability": spam_prob,
                })

        calibrated_threshold = calibrate_decision_threshold(val_records)
        print(f"Validation-calibrated spam probability threshold: {calibrated_threshold:.6f}")

        uci_preds = apply_decision_threshold(uci_preds, calibrated_threshold)
        super_preds = apply_decision_threshold(super_preds, calibrated_threshold)

        pre_drift_metrics = calculate_classification_metrics(uci_preds)
        post_drift_metrics = calculate_classification_metrics(super_preds)
        pre_drift_metrics["calibrated_threshold"] = calibrated_threshold
        post_drift_metrics["calibrated_threshold"] = calibrated_threshold

        baseline_ref = load_baseline_reference(args.baseline_metrics_path)
        base_pre_f1 = float(baseline_ref["results"]["uci"]["f1"])
        base_post_f1 = float(baseline_ref["results"]["super_sms"]["f1"])
        pre_retention_ratio = pre_drift_metrics["f1"] / base_pre_f1
        pre_f1_delta = pre_drift_metrics["f1"] - base_pre_f1
        post_f1_delta = post_drift_metrics["f1"] - base_post_f1

        existing_summary["retention_and_adaptation"].update({
            "calibrated_decision_threshold": calibrated_threshold,
            "post_drift_f1": post_drift_metrics["f1"],
            "pre_drift_f1": pre_drift_metrics["f1"],
            "post_drift_f1_delta": round(post_f1_delta, 6),
            "pre_drift_f1_delta": round(pre_f1_delta, 6),
            "pre_drift_f1_retention_ratio": round(pre_retention_ratio, 6),
            "pre_drift_f1_retention_percent": round(100.0 * pre_retention_ratio, 2),
        })
        existing_summary["evaluation"]["post_drift_super_sms_test"] = post_drift_metrics
        existing_summary["evaluation"]["pre_drift_uci_test"] = pre_drift_metrics

        pred_fields = ["message_id", "label", "prediction", "spam_probability"]
        with uci_pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pred_fields)
            writer.writeheader()
            writer.writerows(uci_preds)
        with super_pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pred_fields)
            writer.writeheader()
            writer.writerows(super_preds)

        write_json(metrics_path, existing_summary)
        generate_plots(
            training_logs=training_logs,
            pre_drift_metrics=pre_drift_metrics,
            post_drift_metrics=post_drift_metrics,
            baseline_ref=baseline_ref,
            output_path=args.output_dir / "adaptation_evaluation_plot.png",
        )
        write_markdown_report(existing_summary, args.output_dir / "FINETUNE_REPORT.md")
        print(f"Post-Drift F1    : {post_drift_metrics['f1']:.2%} (ROC-AUC={post_drift_metrics.get('roc_auc', 0):.4f})")
        print(f"Pre-Drift F1     : {pre_drift_metrics['f1']:.2%} (ROC-AUC={pre_drift_metrics.get('roc_auc', 0):.4f})")
        print(f"Pre-Drift Retent.: {100.0 * pre_retention_ratio:.2f}% of baseline ({pre_f1_delta:+.2%})")
        return

    print("1. Validating stream and dataset splits...")
    stream, uci_rows, super_rows, manifest = validate_inputs()
    print(
        f"   Verified stream ({len(stream)} messages), UCI ({len(uci_rows)} rows), "
        f"Super SMS ({len(super_rows)} rows)."
    )

    print("\n2. Resolving drift trigger...")
    trigger_info = resolve_drift_trigger(
        stream=stream,
        predictions_path=args.predictions_path,
        window_metrics_path=args.window_metrics_path,
        alerts_path=args.alerts_path,
        adwin_delta=args.adwin_delta,
        trigger_window_override=args.trigger_window,
        trigger_position_override=args.trigger_position,
        strict_adwin=args.strict_adwin,
    )
    print(
        f"   Trigger source: {trigger_info['trigger_source']} ({trigger_info['detector']})\n"
        f"   Trigger window: {trigger_info['trigger_window']} | "
        f"Stream position: {trigger_info['trigger_stream_position']} | "
        f"Delay: {trigger_info['detection_delay_messages']} messages"
    )
    if trigger_info.get("fallback_reason"):
        print(f"   Note: {trigger_info['fallback_reason']}")

    print("\n3. Building post-trigger adaptation dataset...")
    train_rows, val_rows, data_stats = build_training_and_val_sets(
        stream=stream,
        uci_rows=uci_rows,
        super_rows=super_rows,
        trigger_position=trigger_info["trigger_stream_position"],
        train_source=args.train_source,
        seed=args.seed,
        limit=args.limit,
        val_limit=args.val_limit,
    )
    print(
        f"   Adaptation train set: {data_stats['train_messages']} messages "
        f"({data_stats['train_spam']} spam, {data_stats['train_ham']} ham)\n"
        f"   Validation set: {data_stats['val_messages']} messages"
    )

    uci_test = [r for r in uci_rows if r["split"] == "test"]
    super_test = [r for r in super_rows if r["split"] == "test"]
    if args.limit > 0:
        uci_test = uci_test[: args.limit]
        super_test = super_test[: args.limit]

    print(
        f"   Held-out test splits: UCI (pre-drift) = {len(uci_test)} messages | "
        f"Super SMS (post-drift) = {len(super_test)} messages"
    )

    print("\n4. Running full fine-tuning and test evaluation...")
    training_logs, compute_stats, uci_preds, super_preds, model_artifacts = (
        train_and_evaluate_pytorch(
            args=args,
            train_rows=train_rows,
            val_rows=val_rows,
            uci_test=uci_test,
            super_test=super_test,
            output_dir=args.output_dir,
        )
    )

    pre_drift_metrics = calculate_classification_metrics(uci_preds)
    post_drift_metrics = calculate_classification_metrics(super_preds)

    # Compare with Phase 2 unadapted baseline for Pre-Drift F1 Retention and Post-Drift F1 Gain
    baseline_ref = load_baseline_reference(args.baseline_metrics_path)
    if baseline_ref and "results" in baseline_ref:
        base_pre_f1 = float(baseline_ref["results"]["uci"]["f1"])
        base_post_f1 = float(baseline_ref["results"]["super_sms"]["f1"])
        base_pre_acc = float(baseline_ref["results"]["uci"]["accuracy"])
        base_post_acc = float(baseline_ref["results"]["super_sms"]["accuracy"])

        pre_retention_ratio = (
            pre_drift_metrics["f1"] / base_pre_f1 if base_pre_f1 > 0 else None
        )
        pre_f1_delta = pre_drift_metrics["f1"] - base_pre_f1
        post_f1_delta = post_drift_metrics["f1"] - base_post_f1
    else:
        base_pre_f1 = None
        base_post_f1 = None
        base_pre_acc = None
        base_post_acc = None
        pre_retention_ratio = None
        pre_f1_delta = None
        post_f1_delta = None

    retention_summary = {
        "post_drift_f1": post_drift_metrics["f1"],
        "pre_drift_f1": pre_drift_metrics["f1"],
        "baseline_post_drift_f1": round(base_post_f1, 6) if base_post_f1 is not None else None,
        "baseline_pre_drift_f1": round(base_pre_f1, 6) if base_pre_f1 is not None else None,
        "post_drift_f1_delta": round(post_f1_delta, 6) if post_f1_delta is not None else None,
        "pre_drift_f1_delta": round(pre_f1_delta, 6) if pre_f1_delta is not None else None,
        "pre_drift_f1_retention_ratio": (
            round(pre_retention_ratio, 6) if pre_retention_ratio is not None else None
        ),
        "pre_drift_f1_retention_percent": (
            round(100.0 * pre_retention_ratio, 2) if pre_retention_ratio is not None else None
        ),
        "baseline_post_drift_accuracy": (
            round(base_post_acc, 6) if base_post_acc is not None else None
        ),
        "baseline_pre_drift_accuracy": (
            round(base_pre_acc, 6) if base_pre_acc is not None else None
        ),
    }

    # Save training_logs.csv and training_history.json
    logs_csv_path = args.output_dir / "training_logs.csv"
    if training_logs:
        with logs_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(training_logs[0].keys()))
            writer.writeheader()
            writer.writerows(training_logs)

    write_json(args.output_dir / "training_history.json", {"logs": training_logs})

    # Save test predictions
    pred_fields = ["message_id", "label", "prediction", "spam_probability"]
    uci_pred_path = args.output_dir / "uci_test_predictions.csv"
    with uci_pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(uci_preds)

    super_pred_path = args.output_dir / "super_sms_test_predictions.csv"
    with super_pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        writer.writerows(super_preds)

    summary = {
        "phase": "Phase 5: Drift-Triggered Full Fine-Tuning",
        "seed": args.seed,
        "model_name": args.model_name,
        "git_commit": git_commit(),
        "stream_sha256": sha256_file(STREAM_PATH),
        "drift_trigger": trigger_info,
        "training_data": data_stats,
        "hyperparameters": {
            "optimizer": compute_stats["optimizer"],
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_accum_steps": args.grad_accum_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "max_length": args.max_length,
            "limit": args.limit,
        },
        "retention_and_adaptation": retention_summary,
        "evaluation": {
            "post_drift_super_sms_test": post_drift_metrics,
            "pre_drift_uci_test": pre_drift_metrics,
        },
        "compute": compute_stats,
        "artifacts": {
            **model_artifacts,
            "training_logs_csv": str(logs_csv_path),
            "uci_test_predictions_csv": str(uci_pred_path),
            "super_sms_test_predictions_csv": str(super_pred_path),
        },
    }

    metrics_path = args.output_dir / "metrics.json"
    write_json(metrics_path, summary)

    plot_path = args.output_dir / "adaptation_evaluation_plot.png"
    generate_plots(
        training_logs=training_logs,
        pre_drift_metrics=pre_drift_metrics,
        post_drift_metrics=post_drift_metrics,
        baseline_ref=baseline_ref,
        output_path=plot_path,
    )

    report_path = args.output_dir / "FINETUNE_REPORT.md"
    write_markdown_report(summary, report_path)

    print("\n" + "=" * 72)
    print("PHASE 5 FULL FINE-TUNING SUMMARY")
    print("=" * 72)
    print(
        f"Trigger          : {trigger_info['detector']} "
        f"(Window {trigger_info['trigger_window']}, Pos {trigger_info['trigger_stream_position']})"
    )
    print(f"Post-Drift F1    : {post_drift_metrics['f1']:.2%} (Super SMS test, n={post_drift_metrics['total']})")
    print(f"Pre-Drift F1     : {pre_drift_metrics['f1']:.2%} (UCI test, n={pre_drift_metrics['total']})")
    if retention_summary["pre_drift_f1_retention_percent"] is not None:
        print(
            f"Pre-Drift Retent.: {retention_summary['pre_drift_f1_retention_percent']:.2f}% of baseline "
            f"({retention_summary['pre_drift_f1_delta']:+.2%})"
        )
    print(
        f"GPU/Compute Time : {compute_stats['total_gpu_compute_seconds']:.2f}s "
        f"({compute_stats['total_gpu_compute_hours']:.6f} hrs) on {compute_stats['device_type']}"
    )
    print(f"Artifacts saved  : {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
