"""Prepare/check Step 7 LoRA on CPU; only --run loads a model or trains."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from benchmark_common import (finite_number, load_inputs, read_csv, read_json, score,
                              sha256, spam_score, validate_predictions, write_csv, write_json)
from benchmark_detectors import DETECTORS, evaluate_detectors
from run_benchmark import ROOT, artifact_folder, audit_run, compute_records, populations, resolve

CODE_FILES = ("run_step7_lora.py", "train_lora.py", "train_adapted.py", "benchmark_common.py",
              "benchmark_detectors.py", "detect_drift.py", "run_benchmark.py", "run_harness.py", "evaluate_baseline.py")
INPUT_FILES = ("baseline_run_config.json", "baseline_predictions.csv", "detector_summary.json", "alerts.csv")
HELDOUTS = ("uci_test", "super_sms_test")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_hashes():
    # Git may convert line endings when transferring Windows work to a Linux GPU.
    return {name: hashlib.sha256((ROOT / "scripts" / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in CODE_FILES}


def repo_path(path):
    path = resolve(path).resolve()
    return path.relative_to(ROOT.resolve()).as_posix()


def equivalent(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(equivalent(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(equivalent(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        return isinstance(actual, (int, float)) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10)
    return actual == expected


def normalized_alerts(rows):
    result = [{"detector": r["detector"], "window": int(r["window"]),
               "stream_position": int(r["stream_position"]),
               "metric_value": finite_number(r["metric_value"], "alert statistic"),
               "threshold": finite_number(r["threshold"], "alert threshold", 0)} for r in rows]
    return sorted(result, key=lambda r: (r["stream_position"], r["detector"]))


def select_trigger(summary, alerts, stream, manifest, seed, detector):
    if summary["seed"] != seed or summary["stream_sha256"] != manifest["stream_csv_sha256"]:
        raise ValueError("Detector artifacts have the wrong seed or stream hash")
    if detector not in DETECTORS:
        raise ValueError("Unknown trigger detector")
    positions = {int(r["stream_position"]): r for r in stream}
    ends = {int(r["window"]): int(r["stream_position"]) for r in stream}
    events = normalized_alerts(alerts)
    for event in events:
        pos, window, variant = event["stream_position"], event["window"], event["detector"]
        if variant not in DETECTORS or pos not in positions or int(positions[pos]["window"]) != window:
            raise ValueError("Alert position/window does not match the stream")
        if pos < summary["monitoring_start_position"]:
            raise ValueError("Calibration-period alert cannot trigger adaptation")
        if variant != "adwin" and pos != ends[window]:
            raise ValueError("Window detector alert must occur at window end")
    selected = next((r for r in events if r["detector"] == detector), None)
    if selected is None:
        raise ValueError(f"No recorded {detector} monitoring alert for seed {seed}; no schedule-onset or other-detector fallback")
    if selected["stream_position"] >= len(stream):
        raise ValueError("No training messages remain strictly after the recorded alert")
    # Never filter using the simulated drift onset: a false alarm would also trigger.
    return {"trigger_source": "benchmark_recorded_alert", "selection_policy": "first_monitored_alert",
            "detector": detector, "trigger_window": selected["window"],
            "trigger_stream_position": selected["stream_position"], "recorded_alert": selected,
            "training_start_position": selected["stream_position"] + 1, "fallback_used": False}


def build_sets(stream, sources, trigger, seed, val_limit):
    # Use the team's seeded sampling, but its inclusive boundary receives alert+1.
    from train_adapted import build_training_and_val_sets
    train, val, stats = build_training_and_val_sets(
        stream, [r for r in sources if r["dataset"] == "uci_sms_spam"],
        [r for r in sources if r["dataset"] == "super_sms"],
        trigger["training_start_position"], "stream_post_trigger", seed, val_limit=val_limit)
    if not train or not val or {r["label"] for r in train} != {"ham", "spam"} or {r["label"] for r in val} != {"ham", "spam"}:
        raise ValueError("Training and validation must each contain ham and spam")
    if any(int(r["stream_position"]) <= trigger["trigger_stream_position"] for r in train):
        raise ValueError("Training uses a message at/before the recorded alert")
    return train, val, stats


def validate_settings(settings):
    if settings["method"] != "drift_triggered_lora" or settings["trigger_policy"] != "first_monitored_alert" or settings["training_policy"] != "offline_stream_strictly_after_alert":
        raise ValueError("Unsupported Step 7 LoRA protocol")
    if not re.fullmatch(r"[0-9a-f]{40}", settings["model_revision"]):
        raise ValueError("Pin a full immutable Hugging Face model revision")
    hp = settings["hyperparameters"]
    for key in ("lora_r", "epochs", "batch_size", "eval_batch_size", "grad_accum_steps", "max_length", "log_every_steps", "val_limit"):
        if isinstance(hp[key], bool) or not isinstance(hp[key], int) or hp[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("lora_alpha", "learning_rate", "max_grad_norm"):
        if finite_number(hp[key], key, 0) == 0:
            raise ValueError(f"{key} must be positive")
    for key in ("lora_dropout", "warmup_ratio"):
        if finite_number(hp[key], key, 0, 1) == 1:
            raise ValueError(f"{key} must be less than 1")
    finite_number(hp["weight_decay"], "weight_decay", 0)
    if hp["bf16"] == hp["fp16"] or hp["fp16"]:
        raise ValueError("Step 7 uses bfloat16; fp16 without loss scaling is not supported")


def prepare(config_path, settings_path, seed, detector_root, preparation_dir, baseline_stream_dir=None):
    config, settings = read_json(config_path), read_json(settings_path)
    validate_settings(settings)
    if seed not in config["seeds"]:
        raise ValueError("Seed is not in the benchmark run configuration")
    method = next(m for m in config["methods"] if m["name"] == settings["method"])
    if method.get("populations") != list(HELDOUTS):
        raise ValueError("Step 7 LoRA requires the two offline held-out populations")
    baseline = next(m for m in config["methods"] if m["name"] == config["baseline_method"])
    baseline_dir = Path(baseline_stream_dir) if baseline_stream_dir else artifact_folder(baseline, seed, "stream")
    detector_dir = Path(detector_root) / baseline["name"] / f"seed_{seed}"
    inputs = dict(zip(INPUT_FILES, (baseline_dir / "run_config.json", baseline_dir / "predictions.csv",
                                   detector_dir / "summary.json", detector_dir / "alerts.csv")))
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    stream, sources, manifest, hashes = load_inputs(resolve(config["processed_dir"]), resolve(config["stream_dir"]))
    base_config = audit_run(read_json(inputs[INPUT_FILES[0]]), seed, sources, manifest, hashes, stream, baseline=True)
    if base_config.get("limit", 0):
        raise ValueError("Smoke predictions cannot supply a LoRA trigger")
    predictions = validate_predictions(read_csv(inputs[INPUT_FILES[1]]), stream, stream=True)
    summary, alerts = read_json(inputs[INPUT_FILES[2]]), read_csv(inputs[INPUT_FILES[3]])
    print(f"CPU: verifying recorded detector artifacts for seed {seed}", flush=True)
    recomputed, new_alerts, _ = evaluate_detectors(
        stream, predictions, manifest, seed, calibration_windows=config.get("detector_calibration_windows", 3),
        bootstrap_samples=config.get("detector_bootstrap_samples", 100))
    for key in ("seed", "stream_sha256", "calibration_windows", "bootstrap_samples", "monitoring_start_position", "thresholds", "detectors"):
        if not equivalent(summary[key], recomputed[key]):
            raise ValueError(f"Recorded detector {key} differs from the benchmark recomputation")
    if not equivalent(normalized_alerts(alerts), normalized_alerts(new_alerts)):
        raise ValueError("Recorded alerts differ from the benchmark recomputation")
    trigger = select_trigger(summary, alerts, stream, manifest, seed, settings["trigger_detector"])
    train, val, stats = build_sets(stream, sources, trigger, seed, settings["hyperparameters"]["val_limit"])
    run_config = {
        "seed": seed, "model": settings["model_name"], "model_revision": settings["model_revision"],
        "stream_sha256": manifest["stream_csv_sha256"], "source_sha256": hashes,
        "evaluation_mode": "offline", "limit": 0,
        "training_ids": [r["message_id"] for r in train], "validation_ids": [r["message_id"] for r in val],
        "replay_ids": [], "retrieval_ids": [], "drift_trigger": trigger, "training_data": stats,
        "seed_controls": ["training/validation sampling and order", "classification head and LoRA initialization", "torch training shuffle and dropout"],
        "hyperparameters": settings["hyperparameters"], "trainer_source_ref": settings["trainer_source_ref"],
        "upstream_sha256": settings["upstream_sha256"], "implementation_sha256_lf": code_hashes(),
    }
    audit_run(run_config, seed, sources, manifest, hashes, stream, baseline=False)
    output = repo_path(resolve(method["runs_dir"]) / f"seed_{seed}")
    if output != f"results/benchmark_runs/drift_triggered_lora/seed_{seed}":
        raise ValueError("LoRA output must be its isolated benchmark_runs directory")
    plan = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "prepared_not_executed", "run_config": run_config, "settings": settings,
            "processed_dir": repo_path(config["processed_dir"]), "stream_dir": repo_path(config["stream_dir"]),
            "output_dir": output, "input_sha256": input_hashes,
            "expected_counts": {p: len(rows) for p, rows in populations(stream, sources).items()}}
    plan["plan_sha256"] = digest(plan)
    preparation_dir = Path(preparation_dir)
    preparation_dir.mkdir(parents=True, exist_ok=False)
    for name, source in inputs.items():
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != input_hashes[name]:
            raise ValueError("Source artifacts changed while preparing; retry with a new preparation directory")
        (preparation_dir / name).write_bytes(raw)
    write_json(preparation_dir / "plan.json", plan)
    return preparation_dir / "plan.json"


def check_plan(path):
    path = Path(path)
    plan = read_json(path)
    if digest({k: v for k, v in plan.items() if k != "plan_sha256"}) != plan["plan_sha256"]:
        raise ValueError("Prepared plan hash mismatch")
    settings, meta = plan["settings"], plan["run_config"]
    validate_settings(settings)
    if (meta["model"] != settings["model_name"] or meta["model_revision"] != settings["model_revision"]
            or meta["hyperparameters"] != settings["hyperparameters"]):
        raise ValueError("Prepared model/hyperparameter metadata does not match execution settings")
    if meta["implementation_sha256_lf"] != code_hashes():
        raise ValueError("Code changed since preparation; prepare a new plan")
    for name in INPUT_FILES:
        if sha256(path.parent / name) != plan["input_sha256"][name]:
            raise ValueError(f"Prepared input hash mismatch: {name}")
    stream, sources, manifest, hashes = load_inputs(resolve(plan["processed_dir"]), resolve(plan["stream_dir"]))
    audit_run(meta, meta["seed"], sources, manifest, hashes, stream, baseline=False)
    base = audit_run(read_json(path.parent / INPUT_FILES[0]), meta["seed"], sources, manifest, hashes, stream, baseline=True)
    if base.get("limit", 0):
        raise ValueError("Smoke predictions cannot supply a LoRA trigger")
    validate_predictions(read_csv(path.parent / INPUT_FILES[1]), stream, stream=True)
    trigger = select_trigger(read_json(path.parent / INPUT_FILES[2]), read_csv(path.parent / INPUT_FILES[3]),
                             stream, manifest, meta["seed"], settings["trigger_detector"])
    train, val, stats = build_sets(stream, sources, trigger, meta["seed"], settings["hyperparameters"]["val_limit"])
    if trigger != meta["drift_trigger"] or [r["message_id"] for r in train] != meta["training_ids"] or [r["message_id"] for r in val] != meta["validation_ids"] or stats != meta["training_data"]:
        raise ValueError("Prepared training IDs or trigger differ from deterministic selection")
    expected = populations(stream, sources)
    if plan["expected_counts"] != {p: len(rows) for p, rows in expected.items()}:
        raise ValueError("Prepared evaluation populations changed")
    if plan["output_dir"] != f"results/benchmark_runs/drift_triggered_lora/seed_{meta['seed']}":
        raise ValueError("Unsafe or incorrect LoRA output directory")
    return plan, train, val, expected


def finish_artifacts(plan, output, result, expected):
    logs, stats, uci_predictions, super_predictions, artifacts = result
    meta = {**plan["run_config"], "status": "complete", "prepared_plan_sha256": plan["plan_sha256"],
            "trigger_artifact_sha256": plan["input_sha256"], "completed_at": datetime.now(timezone.utc).isoformat()}
    if stats.get("model_name") != meta["model"] or stats.get("model_revision") != meta["model_revision"]:
        raise ValueError("Executed model does not match the prepared model/revision")
    threshold = finite_number(stats["calibrated_threshold"], "validation threshold", 0, 1)
    metrics = {}
    for population, rows in zip(HELDOUTS, (uci_predictions, super_predictions)):
        validate_predictions(rows, expected[population])
        for row in rows:
            probability = spam_score(row)
            if probability is None or row["prediction"] != ("spam" if probability >= threshold else "ham"):
                raise ValueError("Predictions must retain continuous scores and the validation-only decision threshold")
        metrics[population] = score(rows)
    adapter_dir = Path(artifacts["lora_adapter_dir"]).resolve()
    if adapter_dir.parent != Path(output).resolve():
        raise ValueError("Adapter was written outside the isolated output directory")
    weights = adapter_dir / "adapter_model.pt"
    if not weights.is_file() or weights.stat().st_size == 0:
        raise ValueError("Missing trained adapter weights")
    compute = {
        "hardware": stats["gpu_name"], "precision": stats["dtype"], "update_count": 1,
        "optimizer_steps": stats["optimizer_steps"], "training_wall_seconds": stats["training_wall_seconds"],
        "evaluation_wall_seconds": stats["uci_test_eval_seconds"] + stats["super_sms_test_eval_seconds"],
        "inference_seconds_by_population": {"uci_test": stats["uci_test_eval_seconds"], "super_sms_test": stats["super_sms_test_eval_seconds"]},
        "peak_memory_mb": stats.get("peak_memory_mb"), "memory_unit": "MiB (legacy _mb column name)",
        "trainable_parameters": stats["trainable_parameters"], "adapter_size_mb": weights.stat().st_size / 1024**2,
        "adapter_size_scope": "adapter_model.pt, LoRA matrices and classification head; MiB",
        "accelerator_hours_measured": None, "compute_cost": None,
        "measurement_method": "Device-synchronized wall-clock timers; CUDA max_memory_allocated; no utilization/cost estimate",
    }
    compute_records([{"method": "drift_triggered_lora", "seed": meta["seed"], "config": {"compute": compute}}])
    meta.update(compute=compute, decision_threshold=threshold, threshold_selection="validation_only_spam_f1",
                model_selection="best_validation_spam_f1", adapter_sha256=sha256(weights),
                adapter_path="lora_adapter/adapter_model.pt", training_statistics=stats,
                environment={"python": platform.python_version(), "platform": platform.platform(),
                             "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "torch", "transformers", "accelerate")}})
    for population, rows in zip(HELDOUTS, (uci_predictions, super_predictions)):
        write_csv(output / f"{population}_predictions.csv", rows)
    write_csv(output / "training_logs.csv", logs)
    write_json(output / "metrics.json", metrics)
    write_json(output / "plan.json", plan)
    write_json(output / "run_config.json", meta)
    return meta


def run_plan(path, device):
    plan, train, val, expected = check_plan(path)
    output = resolve(plan["output_dir"])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite LoRA artifacts: {output}")
    # Deliberately after all CPU-side checks; no torch/model import in --prepare/--check.
    import torch
    if device != "cuda" or not torch.cuda.is_available():
        raise ValueError("--run requires an available CUDA device; CPU fallback is disabled")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("This pinned run requires a CUDA GPU supporting bfloat16")
    from train_adapted import set_reproducibility
    from train_lora import train_and_evaluate_lora
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(".lock")
    with lock.open("x") as handle:
        handle.write(str(os.getpid()))
    pending = None
    try:
        if output.exists():
            raise FileExistsError(f"LoRA output already exists: {output}")
        pending = Path(tempfile.mkdtemp(prefix=f".{output.name}_inprogress_", dir=output.parent))
        write_json(pending / "plan.json", plan)
        settings = plan["settings"]
        args = SimpleNamespace(**settings["hyperparameters"], model_name=settings["model_name"],
                               model_revision=settings["model_revision"], allow_model_fallback=False,
                               hf_token=None, device=device, seed=plan["run_config"]["seed"])
        set_reproducibility(args.seed)
        result = train_and_evaluate_lora(args, train, val, expected["uci_test"], expected["super_sms_test"], pending)
        finish_artifacts(plan, pending, result, expected)
        # Publish only complete, validated artifacts; readers never see half a run.
        pending.rename(output)
        print(f"COMPLETE: {output}", flush=True)
    except Exception:
        if pending is not None:
            print(f"Incomplete artifacts preserved for diagnosis: {pending}", file=sys.stderr)
        raise
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true", help="CPU validation and portable input snapshot only")
    mode.add_argument("--check", action="store_true", help="CPU validation of a prepared plan only")
    mode.add_argument("--run", action="store_true", help="Explicitly download/load the pinned model and train on CUDA")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.json")
    parser.add_argument("--lora-config", type=Path, default=ROOT / "configs/step7_lora.json")
    parser.add_argument("--detector-root", type=Path, default=ROOT / "results/benchmark/detectors")
    parser.add_argument("--preparation-dir", type=Path)
    parser.add_argument("--baseline-stream-dir", type=Path)
    parser.add_argument("--device", choices=["cuda"], default="cuda")
    args = parser.parse_args(argv)
    if args.prepare:
        if args.seed is None or args.plan:
            parser.error("--prepare requires --seed and does not accept --plan")
        destination = args.preparation_dir or ROOT / "results/benchmark/lora_preflight" / f"seed_{args.seed}"
        if destination.exists():
            parser.error("Preparation directory exists; use --check --plan or choose a new --preparation-dir")
        path = prepare(args.config, args.lora_config, args.seed, args.detector_root, destination, args.baseline_stream_dir)
    else:
        if args.plan is None or args.seed is not None:
            parser.error("--check/--run require --plan; the seed is frozen in that plan")
        path = args.plan
    if args.run:
        run_plan(path, args.device)
    else:
        plan, train, val, expected = check_plan(path)
        trigger = plan["run_config"]["drift_trigger"]
        print(f"CPU CHECK PASS: seed {plan['run_config']['seed']}; {trigger['detector']} alert at position {trigger['trigger_stream_position']}; {len(train)} training / {len(val)} validation / {len(expected['uci_test'])} UCI / {len(expected['super_sms_test'])} Super SMS")
        print(f"Prepared plan: {path}\nNo model download, inference, or training performed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"Step 7 LoRA failed: {exc}", file=sys.stderr)
        sys.exit(1)
