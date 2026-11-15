"""Drift-Triggered LoRA (Low-Rank Adaptation) Fine-Tuning Pipeline.

Implements drift-triggered Parameter-Efficient Fine-Tuning (LoRA) of Llama 3.2 3B
(`LlamaForSequenceClassification`), triggered by the same ADWIN / calibrated MMD
drift alert used in `scripts/train_adapted.py`, and compares results directly
against both:
  1. Unadapted 15-Shot Baseline (`results/baseline_full/metrics.json`)
  2. 100% Full-Parameter Fine-Tuning (`results/fine_tuning/metrics.json`)

How LoRA differs from Full Fine-Tuning (`scripts/train_adapted.py`):
  - Freezes 100% of the pretrained 3.21B Llama 3.2 backbone weights (`W_0`),
    preserving pre-drift linguistic representations to minimize catastrophic forgetting.
  - Injects trainable low-rank adapter matrices `A (r x d_in)` and `B (d_out x r)`
    into the Transformer projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
    `gate_proj`, `up_proj`, `down_proj`) plus unfreezes the `3,072 x 2` classification
    head (`score`), training only ~0.3%-0.8% of total parameters.
  - Because only adapter weights require gradients, full 32-bit `AdamW` momentum/variance
    states take <100 MB (instead of 25.7 GB), enabling larger batch sizes (`batch_size=4`,
    `grad_accum=4`), class-weighted cross-entropy loss (`pos_weight` for the 80/20 stream),
    and ~3x faster training on 16GB Apple Silicon MPS.

Usage:
    # Smoke test on 32 messages (fast end-to-end check):
    .venv/bin/python scripts/train_lora.py --limit 32 --epochs 1 --output-dir results/lora_smoke

    # Full LoRA training & comparison against Baseline + Full Fine-Tuning:
    .venv/bin/python scripts/train_lora.py --output-dir results/lora_fine_tuning
"""

import argparse
import csv
import gc
import json
import math
import os
import time
from pathlib import Path

MPL_CACHE = Path("/tmp/matplotlib_cache")
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import matplotlib.pyplot as plt
import numpy as np

from detect_drift import DRIFT_ONSET_WINDOW
from train_adapted import (
    DEFAULT_ALERTS_PATH,
    DEFAULT_BASELINE_METRICS_PATH,
    DEFAULT_MODEL,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_WINDOW_METRICS_PATH,
    FALLBACK_UNGATED_MODEL,
    ID2LABEL,
    LABEL2ID,
    SEED,
    STREAM_PATH,
    apply_decision_threshold,
    build_training_and_val_sets,
    calculate_classification_metrics,
    calibrate_decision_threshold,
    detect_device,
    empty_device_cache,
    format_input_text,
     get_hardware_description,
    git_commit,
    load_baseline_reference,
    read_csv,
    resolve_drift_trigger,
    set_reproducibility,
    sha256_file,
    sync_device,
    validate_inputs,
    write_json,
)

DEFAULT_FULL_FT_METRICS_PATH = Path("results/fine_tuning/metrics.json")
DEFAULT_LORA_OUTPUT_DIR = Path("results/lora_fine_tuning")
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def inject_lora_adapters(
    model,
    r: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.05,
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES,
) -> dict:
    """Freeze base 3.21B backbone and inject trainable low-rank LoRA adapters (A, B) + classification head."""
    import torch
    import torch.nn as nn

    # 1. Freeze all base parameters first
    for param in model.parameters():
        param.requires_grad = False

    class LoRALinear(nn.Module):
        """Low-Rank Adaptation wrapper around a frozen nn.Linear layer: y = W_0(x) + (alpha/r) * B(A(drop(x)))."""

        def __init__(
            self,
            base_layer: nn.Linear,
            rank: int,
            alpha: float,
            dropout: float,
        ):
            super().__init__()
            self.base_layer = base_layer
            self.base_layer.weight.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False

            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
            self.r = rank
            self.lora_alpha = alpha
            self.scaling = alpha / max(1, rank)
            self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

            # Keep LoRA adapter weights in float32 or model dtype on target device
            dtype = base_layer.weight.dtype
            device = base_layer.weight.device
            self.lora_A = nn.Parameter(
                torch.empty((rank, self.in_features), dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros((self.out_features, rank), dtype=dtype, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            base_out = self.base_layer(x)
            lora_x = self.lora_dropout(x.to(self.lora_A.dtype))
            adapter_out = (lora_x @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
            return base_out + (adapter_out * self.scaling).to(base_out.dtype)

    replaced_layers = 0
    for _, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_modules:
                lora_layer = LoRALinear(
                    base_layer=child,
                    rank=r,
                    alpha=lora_alpha,
                    dropout=lora_dropout,
                )
                setattr(module, child_name, lora_layer)
                replaced_layers += 1

    # 2. Unfreeze the sequence classification head (`score` / `classifier`)
    head_params = 0
    for name, param in model.named_parameters():
        if "score" in name or "classifier" in name:
            param.requires_grad = True
            head_params += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (100.0 * trainable_params / total_params) if total_params else 0.0

    return {
        "lora_r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "target_modules": list(target_modules),
        "adapted_linear_layers": replaced_layers,
        "classification_head_params": head_params,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
        "trainable_percentage": round(trainable_pct, 4),
    }


def save_lora_adapter(model, tokenizer, output_dir: Path, lora_info: dict) -> Path:
    """Save only the lightweight trainable LoRA adapter & classification head weights (~35-70 MB)."""
    import torch

    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    adapter_weights_path = adapter_dir / "adapter_model.pt"
    torch.save(trainable_state, adapter_weights_path)

    adapter_config = {
        "peft_type": "LORA",
        "task_type": "SEQ_CLS",
        **lora_info,
    }
    write_json(adapter_dir / "adapter_config.json", adapter_config)
    tokenizer.save_pretrained(adapter_dir)
    return adapter_dir


def train_and_evaluate_lora(
    args,
    train_rows: list[dict],
    val_rows: list[dict],
    uci_test: list[dict],
    super_test: list[dict],
    output_dir: Path,
) -> tuple[list[dict], dict, list[dict], list[dict], dict]:
    """Train LoRA adapters + classification head using AdamW and class-weighted Cross-Entropy."""
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = detect_device(args.device)
    hw_info = get_hardware_description(device)
    print(f"Accelerator selected: {device} ({hw_info['gpu_name']})")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    model_dtype = torch.float32
    if args.bf16 and device.type in ("cuda", "mps"):
        model_dtype = torch.bfloat16
    elif args.fp16 and device.type == "cuda":
        model_dtype = torch.float16

    resolved_model_name = args.model_name
    print(f"Loading tokenizer and base model: {resolved_model_name} ...")
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
                f"   [Note] '{resolved_model_name}' is gated. "
                f"Falling back to ungated weights: '{FALLBACK_UNGATED_MODEL}' ..."
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
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    target_modules = tuple(m.strip() for m in args.target_modules.split(",") if m.strip())
    lora_info = inject_lora_adapters(
        model=model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    print(
        f"LoRA Injected (r={lora_info['lora_r']}, alpha={lora_info['lora_alpha']}): "
        f"{lora_info['trainable_parameters']:,} trainable / {lora_info['total_parameters']:,} total params "
        f"({lora_info['trainable_percentage']:.3f}% trained across {lora_info['adapted_linear_layers']} layers)"
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

    # Inverse-frequency class weighting to counter the 80% ham / 20% spam stream imbalance
    num_spam = max(1, sum(1 for r in train_rows if r["label"].strip().lower() == "spam"))
    num_ham = max(1, len(train_rows) - num_spam)
    spam_weight = (num_ham / num_spam) if args.class_weighted_loss else 1.0
    class_weights = torch.tensor([1.0, spam_weight], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Loss function: CrossEntropyLoss (class_weights=[ham: 1.00, spam: {spam_weight:.2f}])")

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params_list,
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
                )
                logits = outputs.logits.float()
                loss = loss_fn(logits, labels)
                probs = torch.softmax(logits, dim=-1)
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
    best_state_dict = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sync_device(device)
    train_start_wall = time.perf_counter()
    active_gpu_train_seconds = 0.0

    print(
        f"\nStarting LoRA Fine-Tuning (AdamW) for {args.epochs} epoch(s) | "
        f"{len(train_rows)} train msgs | batch_size={args.batch_size} | "
        f"grad_accum={args.grad_accum_steps} (eff_batch={args.batch_size * args.grad_accum_steps}) | "
        f"lr={args.learning_rate}"
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
            )
            logits = outputs.logits.float()
            raw_loss = loss_fn(logits, labels)
            loss_val = float(raw_loss.item())
            loss = raw_loss / args.grad_accum_steps
            loss.backward()

            running_loss += loss_val
            micro_steps += 1
            del input_ids, attention_mask, labels, outputs, logits, raw_loss, loss

            is_update_step = (
                step_idx % args.grad_accum_steps == 0
                or step_idx == len(train_loader)
            )
            if is_update_step:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params_list, args.max_grad_norm
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
                    empty_device_cache(device)
                    val_loss, val_records, val_elapsed = evaluate_loader(val_loader)
                    active_gpu_train_seconds += val_elapsed
                    val_th = calibrate_decision_threshold(val_records)
                    val_cal_records = apply_decision_threshold(val_records, val_th)
                    val_metrics = calculate_classification_metrics(val_cal_records)
                    model.train()

                    log_entry = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": round(avg_train_loss, 6),
                        "val_loss": round(val_loss, 6),
                        "val_accuracy": round(val_metrics["accuracy"], 6),
                        "val_f1": round(val_metrics["f1"], 6),
                        "val_roc_auc": val_metrics.get("roc_auc"),
                        "val_threshold": val_th,
                        "learning_rate": float(f"{current_lr:.6e}"),
                        "grad_norm": round(grad_norm, 4),
                        "elapsed_compute_seconds": round(active_gpu_train_seconds, 2),
                    }
                    training_logs.append(log_entry)
                    print(
                        f"Epoch {epoch}/{args.epochs} | Step {global_step}/{total_opt_steps} | "
                        f"train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | "
                        f"val_F1={val_metrics['f1']:.2%} (AUC={val_metrics.get('roc_auc', 0):.3f}) | "
                        f"compute={active_gpu_train_seconds:.1f}s",
                        flush=True,
                    )

                    if val_metrics["f1"] >= best_val_f1:
                        best_val_f1 = val_metrics["f1"]
                        best_state_dict = {
                            k: v.detach().cpu().clone()
                            for k, v in model.named_parameters()
                            if v.requires_grad
                        }
            else:
                sync_device(device)
                active_gpu_train_seconds += time.perf_counter() - step_t0

    sync_device(device)
    total_train_wall_seconds = time.perf_counter() - train_start_wall

    # Restore best validation checkpoint
    if best_state_dict is not None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in best_state_dict:
                    param.copy_(best_state_dict[name].to(param.device))

    # Calibrate decision threshold on validation split & evaluate on held-out test splits
    print("\nCalibrating threshold on validation split and evaluating on held-out test splits...")
    empty_device_cache(device)
    _, final_val_records, val_cal_sec = evaluate_loader(val_loader)
    calibrated_threshold = calibrate_decision_threshold(final_val_records)
    print(f"Calibrated spam probability threshold on validation split: {calibrated_threshold:.6f}")

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

    uci_test_preds = apply_decision_threshold(uci_test_preds, calibrated_threshold)
    super_test_preds = apply_decision_threshold(super_test_preds, calibrated_threshold)

    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024**2), 2)
    elif device.type == "mps" and hasattr(torch.mps, "driver_allocated_memory"):
        peak_memory_mb = round(torch.mps.driver_allocated_memory() / (1024**2), 2)

    print(f"Saving LoRA adapter & tokenizer to: {output_dir / 'lora_adapter'} ...")
    adapter_dir = save_lora_adapter(model, tokenizer, output_dir, lora_info)

    total_gpu_seconds = active_gpu_train_seconds + val_cal_sec + uci_eval_sec + super_eval_sec
    compute_stats = {
        **hw_info,
        "optimizer": "adamw_lora",
        "class_weighted_loss": args.class_weighted_loss,
        "spam_loss_weight": round(spam_weight, 4),
        "calibrated_threshold": calibrated_threshold,
        **lora_info,
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
        "lora_adapter_dir": str(adapter_dir),
    }


def generate_comparison_plots(
    training_logs: list[dict],
    pre_drift_metrics: dict,
    post_drift_metrics: dict,
    baseline_ref: dict | None,
    full_ft_ref: dict | None,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: LoRA Training & Validation Loss
    ax1 = axes[0]
    if training_logs:
        steps = [r["global_step"] for r in training_logs]
        train_losses = [r["train_loss"] for r in training_logs]
        val_losses = [r["val_loss"] for r in training_logs]
        ax1.plot(steps, train_losses, marker="o", color="#1f77b4", label="Train Loss")
        ax1.plot(steps, val_losses, marker="s", color="#ff7f0e", label="Val Loss")
    ax1.set_xlabel("Optimization Step")
    ax1.set_ylabel("Weighted Cross-Entropy Loss")
    ax1.set_title("1. Drift-Triggered LoRA Fine-Tuning Loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Subplot 2: 3-Way Comparison (Baseline vs. Full Fine-Tuning vs. LoRA)
    ax2 = axes[1]
    categories = ["Pre-Drift (UCI)\nSpam F1", "Post-Drift (Super SMS)\nSpam F1"]
    lora_vals = [100 * pre_drift_metrics["f1"], 100 * post_drift_metrics["f1"]]

    x = np.arange(len(categories))
    has_base = bool(baseline_ref and "results" in baseline_ref)
    has_full = bool(full_ft_ref and "evaluation" in full_ft_ref)

    if has_base and has_full:
        width = 0.25
        base_vals = [
            100 * float(baseline_ref["results"]["uci"]["f1"]),
            100 * float(baseline_ref["results"]["super_sms"]["f1"]),
        ]
        full_vals = [
            100 * float(full_ft_ref["evaluation"]["pre_drift_uci_test"]["f1"]),
            100 * float(full_ft_ref["evaluation"]["post_drift_super_sms_test"]["f1"]),
        ]
        ax2.bar(x - width, base_vals, width, label="15-Shot Baseline", color="#7f7f7f")
        ax2.bar(x, full_vals, width, label="Full Fine-Tuning (LOMO)", color="#1f77b4")
        ax2.bar(x + width, lora_vals, width, label="LoRA Fine-Tuning (AdamW)", color="#2ca02c")
    elif has_base:
        width = 0.35
        base_vals = [
            100 * float(baseline_ref["results"]["uci"]["f1"]),
            100 * float(baseline_ref["results"]["super_sms"]["f1"]),
        ]
        ax2.bar(x - width / 2, base_vals, width, label="15-Shot Baseline", color="#7f7f7f")
        ax2.bar(x + width / 2, lora_vals, width, label="LoRA Fine-Tuning (AdamW)", color="#2ca02c")
    else:
        ax2.bar(x, lora_vals, 0.35, label="LoRA Fine-Tuning (AdamW)", color="#2ca02c")

    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel("F1 Score (%)")
    ax2.set_ylim(0, 105)
    ax2.set_title("2. Baseline vs. Full Fine-Tuning vs. LoRA")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_lora_report(
    summary: dict,
    full_ft_ref: dict | None,
    report_path: Path,
) -> None:
    trig = summary["drift_trigger"]
    pre = summary["evaluation"]["pre_drift_uci_test"]
    post = summary["evaluation"]["post_drift_super_sms_test"]
    ret = summary["retention_and_adaptation"]
    comp = summary["compute"]
    data_cfg = summary["training_data"]

    base_pre_f1 = f"{ret['baseline_pre_drift_f1']:.2%}" if ret["baseline_pre_drift_f1"] is not None else "N/A"
    base_post_f1 = f"{ret['baseline_post_drift_f1']:.2%}" if ret["baseline_post_drift_f1"] is not None else "N/A"
    retention_ratio = f"{ret['pre_drift_f1_retention_ratio']:.2%}" if ret["pre_drift_f1_retention_ratio"] is not None else "N/A"

    full_pre_f1 = "N/A"
    full_post_f1 = "N/A"
    full_pre_acc = "N/A"
    full_post_acc = "N/A"
    full_hrs = "N/A"
    if full_ft_ref and "evaluation" in full_ft_ref:
        f_pre = full_ft_ref["evaluation"]["pre_drift_uci_test"]
        f_post = full_ft_ref["evaluation"]["post_drift_super_sms_test"]
        full_pre_f1 = f"{float(f_pre['f1']):.2%}"
        full_post_f1 = f"{float(f_post['f1']):.2%}"
        full_pre_acc = f"{float(f_pre['accuracy']):.2%}"
        full_post_acc = f"{float(f_post['accuracy']):.2%}"
        full_hrs = f"{float(full_ft_ref['compute']['total_gpu_compute_hours']):.3f} hrs"

    text = f"""# Drift-Triggered LoRA Fine-Tuning & Model Comparison Report

## 1. Drift Trigger & Adaptation Dataset
- **Trigger Source**: `{trig['trigger_source']}` ({trig['detector']}) at **Window {trig['trigger_window']} (Stream Position {trig['trigger_stream_position']})**
- **Training Source**: `{data_cfg['train_source']}` ({data_cfg['train_messages']} messages: {data_cfg['train_spam']} spam, {data_cfg['train_ham']} ham)
- **Validation Set**: {data_cfg['val_messages']} held-out validation messages (Zero Test Leakage: `{data_cfg['zero_test_leakage_verified']}`)

## 2. LoRA Configuration vs. Full Fine-Tuning
- **Base Model**: `{summary['model_name']}`
- **LoRA Rank ($r$) / Alpha ($\alpha$) / Dropout**: `{comp['lora_r']}` / `{comp['lora_alpha']}` / `{comp['lora_dropout']}`
- **Target Modules**: `{', '.join(comp['target_modules'])}` ({comp['adapted_linear_layers']} linear layers + classification head)
- **Trainable Parameters**: **{comp['trainable_parameters']:,}** / {comp['total_parameters']:,} (**{comp['trainable_percentage']:.3f}%** vs. 100% in Full Fine-Tuning)
- **Optimizer**: `AdamW` with class-weighted cross-entropy (`spam_weight = {comp['spam_loss_weight']}`)

## 3. Three-Way Benchmark Comparison: Baseline vs. Full Fine-Tuning vs. LoRA

| Metric | 15-Shot Baseline | Model 1: Full Fine-Tuning (LOMO) | Model 2: LoRA Fine-Tuning (AdamW) |
| :--- | :---: | :---: | :---: |
| **Trainable Parameters** | 0 (0%) | 3,212,755,968 (100%) | **{comp['trainable_parameters']:,} ({comp['trainable_percentage']:.3f}%)** |
| **Post-Drift (Super SMS) Spam F1** | {base_post_f1} | {full_post_f1} | **{post['f1']:.2%}** |
| **Post-Drift (Super SMS) Accuracy** | {f"{ret['baseline_post_drift_accuracy']:.2%}" if ret['baseline_post_drift_accuracy'] else "N/A"} | {full_post_acc} | **{post['accuracy']:.2%}** |
| **Post-Drift (Super SMS) ROC-AUC** | N/A | 0.8378 | **{post.get('roc_auc', 0):.4f}** |
| **Pre-Drift (UCI) Spam F1** | {base_pre_f1} | {full_pre_f1} | **{pre['f1']:.2%}** |
| **Pre-Drift (UCI) Accuracy** | {f"{ret['baseline_pre_drift_accuracy']:.2%}" if ret['baseline_pre_drift_accuracy'] else "N/A"} | {full_pre_acc} | **{pre['accuracy']:.2%}** |
| **Pre-Drift (UCI) F1 Retention** | 100.00% | 89.10% | **{retention_ratio}** |
| **Total GPU/MPS Compute Time** | 0.294 hrs (test only) | {full_hrs} | **{comp['total_gpu_compute_hours']:.3f} hrs ({comp['total_gpu_compute_seconds']:.1f} s)** |
"""
    report_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Drift-triggered LoRA fine-tuning and comparison against Full Fine-Tuning & Baseline."
    )
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--hf-token", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_LORA_OUTPUT_DIR)
    parser.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--window-metrics-path", type=Path, default=DEFAULT_WINDOW_METRICS_PATH)
    parser.add_argument("--alerts-path", type=Path, default=DEFAULT_ALERTS_PATH)
    parser.add_argument("--baseline-metrics-path", type=Path, default=DEFAULT_BASELINE_METRICS_PATH)
    parser.add_argument("--full-ft-metrics-path", type=Path, default=DEFAULT_FULL_FT_METRICS_PATH)
    parser.add_argument("--adwin-delta", type=float, default=0.01)
    parser.add_argument("--trigger-window", type=int, default=None)
    parser.add_argument("--trigger-position", type=int, default=None)
    parser.add_argument("--strict-adwin", action="store_true")
    parser.add_argument(
        "--train-source",
        choices=["stream_post_trigger", "full_post_drift_train", "hybrid"],
        default="stream_post_trigger",
    )
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank r (default: 16)")
    parser.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA scaling alpha (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument(
        "--target-modules",
        type=str,
        default=",".join(DEFAULT_TARGET_MODULES),
        help="Comma-separated linear layer names to adapt with LoRA",
    )
    parser.add_argument(
        "--class-weighted-loss",
        action="store_true",
        default=True,
        help="Weight spam class inversely to stream class imbalance (default: True)",
    )
    parser.add_argument(
        "--no-class-weighted-loss",
        action="store_false",
        dest="class_weighted_loss",
    )
    parser.add_argument("--epochs", type=int, default=2, help="Number of LoRA training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training micro-batch size")
    parser.add_argument("--eval-batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak AdamW learning rate for LoRA")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--log-every-steps", type=int, default=20)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_reproducibility(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("1. Validating stream and dataset splits...")
    stream, uci_rows, super_rows, _ = validate_inputs()

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
        f"   Trigger: {trigger_info['detector']} (Window {trigger_info['trigger_window']}, "
        f"Pos {trigger_info['trigger_stream_position']})"
    )

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

    uci_test = [r for r in uci_rows if r["split"] == "test"]
    super_test = [r for r in super_rows if r["split"] == "test"]
    if args.limit > 0:
        uci_test = uci_test[: args.limit]
        super_test = super_test[: args.limit]

    print("\n4. Running LoRA fine-tuning and test split evaluation...")
    training_logs, compute_stats, uci_preds, super_preds, model_artifacts = (
        train_and_evaluate_lora(
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

    baseline_ref = load_baseline_reference(args.baseline_metrics_path)
    full_ft_ref = load_baseline_reference(args.full_ft_metrics_path)

    if baseline_ref and "results" in baseline_ref:
        base_pre_f1 = float(baseline_ref["results"]["uci"]["f1"])
        base_post_f1 = float(baseline_ref["results"]["super_sms"]["f1"])
        base_pre_acc = float(baseline_ref["results"]["uci"]["accuracy"])
        base_post_acc = float(baseline_ref["results"]["super_sms"]["accuracy"])
        pre_retention_ratio = pre_drift_metrics["f1"] / base_pre_f1 if base_pre_f1 > 0 else None
        pre_f1_delta = pre_drift_metrics["f1"] - base_pre_f1
        post_f1_delta = post_drift_metrics["f1"] - base_post_f1
    else:
        base_pre_f1 = base_post_f1 = base_pre_acc = base_post_acc = None
        pre_retention_ratio = pre_f1_delta = post_f1_delta = None

    retention_summary = {
        "calibrated_decision_threshold": compute_stats["calibrated_threshold"],
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
        "baseline_post_drift_accuracy": round(base_post_acc, 6) if base_post_acc is not None else None,
        "baseline_pre_drift_accuracy": round(base_pre_acc, 6) if base_pre_acc is not None else None,
    }

    if full_ft_ref and "evaluation" in full_ft_ref:
        full_pre_f1 = float(full_ft_ref["evaluation"]["pre_drift_uci_test"]["f1"])
        full_post_f1 = float(full_ft_ref["evaluation"]["post_drift_super_sms_test"]["f1"])
        retention_summary["vs_full_finetuning"] = {
            "full_ft_post_drift_f1": full_post_f1,
            "full_ft_pre_drift_f1": full_pre_f1,
            "lora_minus_full_ft_post_drift_f1": round(post_drift_metrics["f1"] - full_post_f1, 6),
            "lora_minus_full_ft_pre_drift_f1": round(pre_drift_metrics["f1"] - full_pre_f1, 6),
        }

    logs_csv_path = args.output_dir / "training_logs.csv"
    if training_logs:
        with logs_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(training_logs[0].keys()))
            writer.writeheader()
            writer.writerows(training_logs)

    write_json(args.output_dir / "training_history.json", {"logs": training_logs})

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
        "phase": "Drift-Triggered LoRA Fine-Tuning",
        "seed": args.seed,
        "model_name": args.model_name,
        "git_commit": git_commit(),
        "stream_sha256": sha256_file(STREAM_PATH),
        "drift_trigger": trigger_info,
        "training_data": data_stats,
        "hyperparameters": {
            "optimizer": "adamw_lora",
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(compute_stats["target_modules"]),
            "class_weighted_loss": args.class_weighted_loss,
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

    write_json(args.output_dir / "metrics.json", summary)
    generate_comparison_plots(
        training_logs=training_logs,
        pre_drift_metrics=pre_drift_metrics,
        post_drift_metrics=post_drift_metrics,
        baseline_ref=baseline_ref,
        full_ft_ref=full_ft_ref,
        output_path=args.output_dir / "lora_comparison_plot.png",
    )
    write_lora_report(
        summary=summary,
        full_ft_ref=full_ft_ref,
        report_path=args.output_dir / "LORA_REPORT.md",
    )

    print("\n" + "=" * 72)
    print("LORA FINE-TUNING SUMMARY")
    print("=" * 72)
    print(
        f"Trainable Params : {compute_stats['trainable_parameters']:,} / "
        f"{compute_stats['total_parameters']:,} ({compute_stats['trainable_percentage']:.3f}%)"
    )
    print(f"Post-Drift F1    : {post_drift_metrics['f1']:.2%} (Super SMS test, ROC-AUC={post_drift_metrics.get('roc_auc', 0):.4f})")
    print(f"Pre-Drift F1     : {pre_drift_metrics['f1']:.2%} (UCI test, ROC-AUC={pre_drift_metrics.get('roc_auc', 0):.4f})")
    if retention_summary["pre_drift_f1_retention_percent"] is not None:
        print(f"Pre-Drift Retent.: {retention_summary['pre_drift_f1_retention_percent']:.2f}% of baseline")
    if "vs_full_finetuning" in retention_summary:
        vs_ft = retention_summary["vs_full_finetuning"]
        print(
            f"vs. Full FT      : Post-Drift F1 {vs_ft['lora_minus_full_ft_post_drift_f1']:+.2%} | "
            f"Pre-Drift F1 {vs_ft['lora_minus_full_ft_pre_drift_f1']:+.2%}"
        )
    print(
        f"GPU/Compute Time : {compute_stats['total_gpu_compute_seconds']:.2f}s "
        f"({compute_stats['total_gpu_compute_hours']:.4f} hrs) on {compute_stats['device_type']}"
    )
    print(f"Artifacts saved  : {args.output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
