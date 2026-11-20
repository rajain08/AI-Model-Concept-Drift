"""Step 7: validate and compare experiments; inference requires an explicit flag."""

import argparse
import importlib.metadata
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from benchmark_common import (
    aggregate, finite_number, load_inputs, read_csv, read_json, retention,
    score, sha256, validate_predictions, write_csv, write_json,
)
from benchmark_history import audit_existing_results, mark_pending_verification

ROOT = Path(__file__).resolve().parents[1]
POPULATIONS = {"stream": "predictions.csv", "uci_test": "uci_test_predictions.csv",
               "super_sms_test": "super_sms_test_predictions.csv"}
CLASSIFICATION_METRICS = ("f1", "accuracy", "precision", "recall", "false_positive_rate", "pr_auc")
METRICS = (*CLASSIFICATION_METRICS, "invalid_rate", "roc_auc",
           "mean_latency_seconds", "p95_latency_seconds", "inference_wall_seconds", "model_generation_seconds")


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def populations(stream, sources):
    return {"stream": stream,
            "uci_test": [r for r in sources if r["dataset"] == "uci_sms_spam" and r["split"] == "test"],
            "super_sms_test": [r for r in sources if r["dataset"] == "super_sms" and r["split"] == "test"]}


def artifact_folder(method, seed, population):
    """Population-specific external inputs are read-only and never inference destinations."""
    override = method.get("reuse_runs", {}).get(str(seed), {}).get(population)
    return resolve(override) if override else resolve(method["runs_dir"]) / f"seed_{seed}"


def read_run_config(folder, seed, sources, manifest, hashes, stream, *, baseline, limit=0):
    path = folder / "run_config.json"
    config = audit_run(read_json(path), seed, sources, manifest, hashes, stream, baseline=baseline)
    if config.get("limit", 0) != limit:
        raise ValueError("Smoke/full limit differs from benchmark mode")
    return config


def audit_run(config, seed, sources, manifest, hashes, stream, *, baseline):
    if config.get("seed") != seed or config.get("stream_sha256") != manifest["stream_csv_sha256"]:
        raise ValueError("Run seed or stream hash differs from benchmark")
    if not config.get("model"):
        raise ValueError("Run must identify its model")
    source_map = {r["message_id"]: r for r in sources}
    positions = {r["message_id"]: int(r["stream_position"]) for r in stream}
    if config.get("source_sha256") is not None and config["source_sha256"] != hashes:
        raise ValueError("Run source hashes differ from benchmark")

    def check_ids(ids, split, *, after=None):
        if not isinstance(ids, list) or len(ids) != len(set(ids)):
            raise ValueError("Usage IDs must be a list without duplicates")
        for mid in ids:
            if mid not in source_map or source_map[mid]["split"] != split:
                raise ValueError(f"Unknown or ineligible {split} usage ID: {mid}")
            if after is not None and positions.get(mid, 0) > after:
                raise ValueError(f"Future stream message used before observation: {mid}")

    if baseline:
        from evaluate_baseline import MODEL
        from run_harness import choose_examples
        if config["model"] != MODEL:
            raise ValueError(f"Baseline model must be {MODEL}")
        expected = [r["message_id"] for r in choose_examples(sources, set(positions), seed)]
        if config.get("prompt_examples") != expected:
            raise ValueError("Baseline examples differ from the seeded non-stream training examples")
    else:
        if config.get("source_sha256") != hashes:
            raise ValueError("Adapted runs must record source_sha256 for both processed datasets")
        for key in ("training_ids", "replay_ids", "retrieval_ids", "validation_ids"):
            if key not in config:
                raise ValueError(f"Adapted runs must declare {key} (an empty list is allowed)")
            check_ids(config[key], "val" if key == "validation_ids" else "train")
        if config.get("evaluation_mode") not in ("offline", "prequential"):
            raise ValueError("Declare evaluation_mode as offline or prequential")
        if config["evaluation_mode"] == "prequential":
            for key in ("training_ids", "replay_ids", "retrieval_ids"):
                check_ids(config[key], "train", after=0)
            if not isinstance(config.get("usage_events"), list):
                raise ValueError("Prequential runs must declare usage_events")
            previous = 0
            for event in config["usage_events"]:
                position = event["after_position"]
                if not isinstance(position, int) or not previous <= position <= len(stream):
                    raise ValueError("Usage event positions must be ordered and within the stream")
                previous = position
                for key in ("training_ids", "replay_ids", "retrieval_ids"):
                    check_ids(event.get(key, []), "train", after=position)
    return config


def baseline_inference(seed, output, expected, sources, manifest, hashes, *, limit=0, selected_populations=None):
    """Resume validated prediction prefixes, using the existing Ollama prompt."""
    from evaluate_baseline import MODEL
    from run_harness import choose_examples, git_commit, gpu_name, predict
    import csv
    import urllib.request

    selected_populations = list(POPULATIONS) if selected_populations is None else list(selected_populations)
    if not selected_populations:
        return
    output.mkdir(parents=True, exist_ok=True)
    examples = choose_examples(sources, {r["message_id"] for r in expected["stream"]}, seed)
    # Capture model digest and runtime identity before any inference.
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        models = json.load(response)["models"]
    model_info = next((m for m in models if m.get("name") == MODEL), None)
    if model_info is None:
        raise ValueError(f"Ollama does not have {MODEL}; install it before running inference")
    config = {
        "seed": seed, "model": MODEL, "model_digest": model_info.get("digest"),
        "prompt_examples": [r["message_id"] for r in examples],
        "stream_sha256": manifest["stream_csv_sha256"], "source_sha256": hashes,
        "limit": limit, "evaluation_mode": "prequential", "temperature": 0, "num_predict": 128,
        "git_commit": git_commit(), "platform": platform.platform(),
        "python": platform.python_version(), "gpu_detected": gpu_name(),
        "seed_controls": ["few-shot example sampling and order", "Ollama generation seed"],
        "stream_seed": manifest["seed"],
        "inference_code_sha256": {name: sha256(ROOT / "scripts" / name) for name in ("evaluate_baseline.py", "run_harness.py")},
    }
    config_path = output / "run_config.json"
    if config_path.exists():
        old = read_json(config_path)
        audit_run(old, seed, sources, manifest, hashes, expected["stream"], baseline=True)
        keys = ("limit", "model", "prompt_examples", "model_digest", "source_sha256", "temperature", "num_predict", "inference_code_sha256")
        if any(old.get(key) != config[key] for key in keys):
            raise ValueError("Cannot resume inference with different or incomplete provenance; use a new runs directory")
    else:
        if any((output / filename).exists() for filename in POPULATIONS.values()):
            raise ValueError("Prediction files exist without a run configuration")
        write_json(config_path, config)
    for population in selected_populations:
        filename = POPULATIONS[population]
        selected = expected[population][:limit] if limit else expected[population]
        path = output / filename
        completed = read_csv(path) if path.exists() else []
        validate_predictions(completed, selected, stream=population == "stream", allow_prefix=True)
        # Held-out files may be unordered when imported, but resumable inference is a prefix.
        if [r["message_id"] for r in completed] != [r["message_id"] for r in selected[:len(completed)]]:
            raise ValueError("Resume requires a prediction prefix in evaluation order")
        fields = ["message_id", "label", "prediction", "raw_response", "latency_seconds", "ollama_eval_seconds"]
        if population == "stream":
            fields = ["stream_position", "window", "phase", "post_drift_fraction", "dataset"] + fields
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if path.stat().st_size == 0:
                writer.writeheader()
            for index, row in enumerate(selected[len(completed):], len(completed) + 1):
                pred, raw, elapsed, generation = predict(row["text"], examples, seed)
                record = {key: row[key] for key in fields if key in row}
                record.update(prediction=pred, raw_response=raw, latency_seconds=elapsed, ollama_eval_seconds=generation)
                writer.writerow(record)
                handle.flush()
                if limit or index % 50 == 0 or index == len(selected):
                    print(f"{seed} {population}: {index}/{len(selected)}", flush=True)
        validate_predictions(read_csv(path), selected, stream=population == "stream")


def run_baseline_seed(method, seed, expected, sources, manifest, hashes, *, limit=0):
    """Validate reuse first, then infer only the remaining requested populations."""
    reuse = method.get("reuse_runs", {}).get(str(seed), {})
    requested = method.get("populations", list(POPULATIONS))
    destination = resolve(method["runs_dir"]) / f"seed_{seed}"
    for population, directory in reuse.items():
        if population not in requested:
            continue
        folder = resolve(directory)
        if folder.resolve() == destination.resolve():
            raise ValueError("Read-only reuse folder must differ from inference destination")
        read_run_config(folder, seed, sources, manifest, hashes, expected["stream"], baseline=True, limit=limit)
        target = expected[population][:limit] if limit else expected[population]
        validate_predictions(read_csv(folder / POPULATIONS[population]), target, stream=population == "stream")
        print(f"REUSE PASS: seed {seed} / {population} / {len(target)} records from {folder}", flush=True)
    remaining = [population for population in requested if population not in reuse]
    if remaining:
        baseline_inference(seed, destination, expected, sources, manifest, hashes,
                           limit=limit, selected_populations=remaining)


def collect(config, expected, sources, manifest, hashes, output, *, limit=0):
    runs, availability, windows, detectors, provenance = [], [], [], [], []
    for method in config["methods"]:
        name = method["name"]
        for seed in config["seeds"]:
            configs_seen = {}
            for population, filename in POPULATIONS.items():
                folder = artifact_folder(method, seed, population)
                path = folder / filename
                target = expected[population][:limit] if limit else expected[population]
                status = {"method": name, "seed": seed, "population": population,
                          "expected": len(target), "observed": 0, "status": "missing", "detail": "",
                          "path": str(path),
                          "source": "reused" if population in method.get("reuse_runs", {}).get(str(seed), {}) else "local"}
                if population not in method.get("populations", POPULATIONS):
                    status.update(status="not_requested", expected=0, detail="Population not requested in method configuration")
                    availability.append(status)
                    continue
                try:
                    if not path.exists():
                        status["detail"] = str(path)
                        continue
                    records = read_csv(path)
                    status["observed"] = len(records)
                    run_config = read_run_config(folder, seed, sources, manifest, hashes, expected["stream"],
                                                 baseline=name == config["baseline_method"], limit=limit)
                    if population == "stream" and name != config["baseline_method"] and run_config["evaluation_mode"] != "prequential":
                        raise ValueError("Offline adapted predictions cannot be scored as online stream recovery")
                    validate_predictions(records, target, stream=population == "stream", allow_prefix=True)
                    if len(records) != len(target):
                        status.update(status="incomplete", detail="Excluded from aggregates and detector evaluation")
                        continue
                    metrics = score(records)
                    status["status"] = "smoke" if limit else "complete"
                    config_path = folder / "run_config.json"
                    if str(config_path) not in configs_seen:
                        config_record = {"method": name, "seed": seed, "path": str(config_path),
                                         "sha256": sha256(config_path), "config": run_config, "populations": []}
                        configs_seen[str(config_path)] = config_record
                        provenance.append(config_record)
                    configs_seen[str(config_path)]["populations"].append(population)
                    provenance.append({"method": name, "seed": seed, "population": population,
                                       "path": str(path), "sha256": sha256(path)})
                    if population == "stream":
                        grouped = defaultdict(list)
                        window_rows = []
                        for row in records:
                            grouped[int(row["window"])].append(row)
                        for window, group in sorted(grouped.items()):
                            window_rows.append({"method": name, "seed": seed, "window": window,
                                                "phase": group[0]["phase"], **score(group)})
                        if not limit:
                            from benchmark_detectors import DETECTOR_FAMILIES, evaluate_detectors, save_detector_results
                            summary, alerts, signals = evaluate_detectors(
                                expected["stream"], records, manifest, seed,
                                calibration_windows=config.get("detector_calibration_windows", 3),
                                bootstrap_samples=config.get("detector_bootstrap_samples", 100))
                            save_detector_results(output / "detectors" / name / f"seed_{seed}", summary, alerts, signals)
                            for detector, values in summary["detectors"].items():
                                detectors.append({"method": name, "seed": seed, "family": DETECTOR_FAMILIES[detector], "detector": detector, **values})
                        windows.extend(window_rows)
                    runs.append({"method": name, "seed": seed, "population": population, **metrics})
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    status.update(status="invalid", detail=str(exc))
                finally:
                    availability.append(status)
    return runs, availability, windows, detectors, provenance


def complete_model_seeds(runs, method):
    required = set(method.get("populations", POPULATIONS)) - {"stream"}
    seeds = {r["seed"] for r in runs if r["method"] == method["name"]}
    return {seed for seed in seeds if required and required.issubset(
        {r["population"] for r in runs if r["method"] == method["name"] and r["seed"] == seed})}


def summarize(runs, config):
    summaries, paired = [], []
    for method in config["methods"]:
        name = method["name"]
        completed = complete_model_seeds(runs, method)
        for population in POPULATIONS:
            group = [r for r in runs if r["method"] == name and r["population"] == population
                     and (population == "stream" or r["seed"] in completed)]
            for metric in METRICS:
                summaries.append({"method": name, "population": population, "metric": metric,
                                  "expected_seeds": len(config["seeds"]),
                                  "complete_seeds": ",".join(str(r["seed"]) for r in group),
                                  "unavailable_reasons": "; ".join(sorted({r.get(f"{metric}_unavailable_reason") for r in group if r.get(f"{metric}_unavailable_reason")})),
                                  **aggregate([r[metric] for r in group])})
        if name != config["baseline_method"]:
            for seed in config["seeds"]:
                for population in ("uci_test", "super_sms_test"):
                    before = next((r for r in runs if r["method"] == config["baseline_method"] and r["seed"] == seed and r["population"] == population), None)
                    after = next((r for r in runs if r["method"] == name and r["seed"] == seed and r["population"] == population), None)
                    if before and after:
                        paired.append({"method": name, "seed": seed, "population": population, **retention(before, after)})
    return summaries, paired


def grouped_summary(rows, groups, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in groups)].append(row)
    return [{**dict(zip(groups, keys)), "metric": metric,
             **aggregate([row[metric] for row in group])}
            for keys, group in sorted(grouped.items(), key=lambda item: tuple("" if v is None else str(v) for v in item[0])) for metric in metrics]


def comparison_table(summary, config):
    """One row per model/population; error bars are across-seed sample SD."""
    table = []
    for population in POPULATIONS:
        for method in config["methods"]:
            metrics = {r["metric"]: r for r in summary
                       if r["method"] == method["name"] and r["population"] == population}
            f1, accuracy = metrics["f1"], metrics["accuracy"]
            requested = population in method.get("populations", POPULATIONS)
            table.append({"method": method["name"], "label": method.get("label", method["name"]),
                          "population": population, "n": f1["n"], "expected_seeds": len(config["seeds"]),
                          "seeds": f1["complete_seeds"], "f1_mean": f1["mean"], "f1_std": f1["std"],
                          "accuracy_mean": accuracy["mean"], "accuracy_std": accuracy["std"],
                          "status": "not_requested" if not requested else "pending" if not f1["n"]
                          else "complete" if f1["n"] == len(config["seeds"]) else "partial_seeds"})
            for metric in CLASSIFICATION_METRICS:
                table[-1].update({f"{metric}_{key}": metrics[metric][key] for key in ("n", "mean", "std")})
                table[-1][f"{metric}_unavailable_reasons"] = metrics[metric]["unavailable_reasons"]
    return table


def make_comparison_plot(output, table, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), sharey=True)
    titles = ("Training-stream diagnostic", "Held-out UCI", "Held-out Super SMS")
    for ax, population, title in zip(axes, POPULATIONS, titles):
        rows = [r for r in table if r["population"] == population]
        for x, row in enumerate(rows):
            if row["f1_mean"] is None:
                text = "Not requested" if row["status"] == "not_requested" else "Pending"
                ax.text(x, 52, text, ha="center", color="#666666", fontsize=10)
                continue
            mean = 100 * row["f1_mean"]
            if row["f1_std"] is None:
                ax.plot(x, mean, "o", color="#1665a7", markersize=7)
                note = f"{mean:.2f}%\nn={row['n']}; SD unavailable"
            else:
                ax.errorbar(x, mean, yerr=100 * row["f1_std"], fmt="o", capsize=6,
                            color="#1665a7", markersize=7)
                note = f"{mean:.2f}%\nn={row['n']}"
            ax.annotate(note, (x, mean), xytext=(0, 12 if mean < 80 else -35),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.set_xticks(range(len(rows)), [r["label"].replace(" Llama", "\nLlama").replace(" LoRA", "\nLoRA") for r in rows])
        ax.set(title=title, xlim=(-0.6, len(rows) - 0.4), ylim=(-3, 105))
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Spam F1 (%) — primary metric")
    fig.suptitle("SMOKE CHECK ONLY" if smoke else "15-shot baseline vs drift-triggered LoRA")
    fig.text(0.5, 0.015, "Error bars: sample SD across seeds. Missing runs are not zeros; n=1 has no error bar.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(output / "benchmark_f1.png", dpi=180)
    plt.close(fig)


def compute_records(provenance):
    records = []
    for item in provenance:
        if "config" not in item:
            continue
        config = item["config"]
        compute = config.get("compute", {})
        row = {"method": item["method"], "seed": item["seed"],
               "populations": ",".join(item.get("populations", [])),
               "hardware": compute.get("hardware", config.get("gpu_detected")),
               "precision": compute.get("precision"), "update_count": compute.get("update_count")}
        for key in ("training_wall_seconds", "evaluation_wall_seconds", "peak_memory_mb", "accelerator_hours_measured", "trainable_parameters", "adapter_size_mb", "compute_cost"):
            value = compute.get(key)
            row[key] = finite_number(value, key, 0) if value is not None else None
        if item["method"] == "baseline":
            row["training_wall_seconds"] = 0.0
            row["trainable_parameters"] = 0
        if row["trainable_parameters"] is not None and not float(row["trainable_parameters"]).is_integer():
            raise ValueError("trainable_parameters must be an integer count")
        row["cost_currency"] = compute.get("cost_currency")
        if row["compute_cost"] is not None and not row["cost_currency"]:
            raise ValueError("compute_cost requires cost_currency")
        if row["accelerator_hours_measured"] is not None and (not row["hardware"] or not compute.get("measurement_method")):
            raise ValueError("Measured accelerator-hours require hardware and measurement_method")
        row["measurement_method"] = compute.get("measurement_method")
        row["inference_seconds_by_population"] = {
            population: finite_number(value, "inference_seconds_by_population", 0)
            for population, value in compute.get("inference_seconds_by_population", {}).items()
        }
        if any(p not in POPULATIONS for p in row["inference_seconds_by_population"]):
            raise ValueError("Unknown population in inference_seconds_by_population")
        records.append(row)
    return records


def adaptation_records(runs, config):
    records, summaries = [], []
    baseline = config["baseline_method"]
    completed = {m["name"]: complete_model_seeds(runs, m) for m in config["methods"]}
    lookup = {(r["method"], r["seed"], r["population"]): r for r in runs}
    for method in config["methods"]:
        name = method["name"]
        for seed in config["seeds"]:
            post = lookup.get((name, seed, "super_sms_test"))
            pre = lookup.get((name, seed, "uci_test"))
            base_post = lookup.get((baseline, seed, "super_sms_test"))
            base_pre = lookup.get((baseline, seed, "uci_test"))
            records.append({"method": name, "seed": seed,
                            "complete_model_seed": seed in completed[name],
                            "complete_paired_seed": seed in completed[name] & completed[baseline],
                            "super_sms_f1": post["f1"] if post else None,
                            "super_sms_f1_change_pp": 100 * (post["f1"] - base_post["f1"]) if post and base_post else None,
                            "uci_f1_retention_pct": 100 * pre["f1"] / base_pre["f1"] if pre and base_pre and base_pre["f1"] else None})
        for metric in ("super_sms_f1", "super_sms_f1_change_pp", "uci_f1_retention_pct"):
            eligible = "complete_model_seed" if metric == "super_sms_f1" else "complete_paired_seed"
            summaries.append({"method": name, "metric": metric,
                              **aggregate([r[metric] for r in records if r["method"] == name and r[eligible]])})
    return records, summaries


def resource_records(availability, runs, compute, config):
    """Keep population inference and shared run-level training measurements explicit."""
    lookup = {(r["method"], r["seed"], r["population"]): r for r in runs}
    completed = {m["name"]: complete_model_seeds(runs, m) for m in config["methods"]}
    records = []
    for item in availability:
        if item["status"] == "not_requested":
            continue
        name, seed, population = item["method"], item["seed"], item["population"]
        run = lookup.get((name, seed, population), {})
        resource = next((r for r in compute if r["method"] == name and r["seed"] == seed
                         and population in r["populations"].split(",")), {})
        row = {"method": name, "seed": seed, "population": population, "status": item["status"],
               "complete_model_seed": seed in completed[name],
               "hardware": resource.get("hardware") or "unknown",
               "measurement_scope": resource.get("populations"),
               "inference_wall_seconds": run.get("inference_wall_seconds")}
        if row["inference_wall_seconds"] is None:
            row["inference_wall_seconds"] = resource.get("inference_seconds_by_population", {}).get(population)
        for metric in ("training_wall_seconds", "peak_memory_mb", "accelerator_hours_measured", "compute_cost", "cost_currency", "trainable_parameters", "adapter_size_mb"):
            row[metric] = resource.get(metric)
        if name == config["baseline_method"]:
            row.update(training_wall_seconds=0, trainable_parameters=0)
        records.append(row)
    return records


def make_plot(output, windows, manifest, smoke):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for method in sorted({r["method"] for r in windows}):
        rows = [r for r in windows if r["method"] == method]
        x = sorted({r["window"] for r in rows})
        for ax, metric in zip(axes, ("f1", "accuracy")):
            stats = [aggregate([r[metric] for r in rows if r["window"] == window]) for window in x]
            means = [100 * s["mean"] for s in stats]
            ax.plot(x, means, marker="o", label=f"{method} (n={len({r['seed'] for r in rows})})")
            if all(s["std"] is not None for s in stats):
                ax.fill_between(x, [m - 100*s["std"] for m, s in zip(means, stats)],
                                [m + 100*s["std"] for m, s in zip(means, stats)], alpha=0.15)
    for ax, metric in zip(axes, ("Spam F1 (primary)", "Accuracy")):
        if not smoke:
            ax.axvline(manifest["drift_onset_window"] - 0.5, linestyle="--", color="gray", label="Simulated onset")
        ax.set(xlabel="Stream window", ylabel=f"{metric} (%)", ylim=(0, 100))
        if windows:
            ticks = sorted({r["window"] for r in windows})
            ax.set_xticks(ticks)
            ax.set_xlim(ticks[0] - 0.5, ticks[-1] + 0.5)
        ax.grid(alpha=0.2)
        if windows:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "No complete stream runs available", transform=ax.transAxes, ha="center")
    fig.suptitle("SMOKE CHECK ONLY" if smoke else "Training-stream diagnostics: mean and sample SD across seeds")
    fig.tight_layout()
    fig.savefig(output / "window_performance.png", dpi=160)
    plt.close(fig)


def write_report(output, data):
    from benchmark_reporting import classification_sections, adaptation_sections, resource_sections, history_sections
    mode = "Smoke validation (not experiment evidence)" if data["smoke_limit"] else "Step 7 benchmark"
    lines = [f"# {mode}", "", "No inference is performed by the default report command.", "",
             "Comparison: 15-shot Llama 3.2 3B versus the team's drift-triggered LoRA. **Spam F1 is primary.**",
             "Three detector families: ADWIN; statistical KS/PSI (two variants); embedding MMD. KS and PSI are not counted as separate families.", "",
             "Spam is positive. Invalid outputs count as incorrect; spam abstentions count as false negatives.",
             "Confusion matrix rows: ham, spam; columns: ham, spam, invalid. Metrics are recomputed from predictions.", "",
             "FPR = false positives / all actual ham (including invalid ham). PR-AUC is trapezoidal area under the precision-recall curve from continuous spam scores, not average precision; label-only runs report N/A with a reason.", "",
             "## Coverage", "", "| Method | Seed | Stream | UCI test | Super SMS test |", "|---|---:|---|---|---|"]
    for method in data["config"]["methods"]:
        for seed in data["config"]["seeds"]:
            group = [s for s in data["availability"] if s["method"] == method["name"] and s["seed"] == seed]
            cells = [f"{s['status']} ({s['observed']}/{s['expected']})" for s in group]
            lines.append(f"| {method['name']} | {seed} | " + " | ".join(cells) + " |")
    lines += [""] + classification_sections(data)
    lines += adaptation_sections(data) + resource_sections(data) + history_sections(data)
    lines += ["", "![Spam F1 comparison](benchmark_f1.png)", "", "## Reused evidence", ""]
    reused = [r for r in data["availability"] if r["source"] == "reused"]
    for row in reused:
        lines.append(f"- {row['method']} / {row['seed']} / {row['population']}: {row['status']} ({row['observed']}/{row['expected']}); `{row['path']}`. Evaluated through the same run-config and prediction checks as local inputs.")
    if reused:
        lines += ["", "The external stream is read-only. Held-out UCI and Super SMS files are resolved independently from the local run directory. Detector results are recomputed with this benchmark's calibration protocol; historical detector summaries are not imported.",
                  "Legacy stream metadata identifies model tag and examples but not the original model digest; do not infer byte-identical historical weights from a matching tag."]
    lines += ["", "## Detection, retention, and compute", "",
              "Detector results are in detector_metrics.csv and detectors/. Calibration uses early pre-drift windows; those windows are excluded from false-alarm denominators. Window alerts occur at window end. Missing detections retain null delay and detected=false.",
              "", "paired_comparisons.csv compares the same seed and held-out examples. Retention is after/before x 100 (uncapped); zero baseline denominators produce null. Changes are percentage points.",
              "", "compute.csv keeps training, evaluation, and measured accelerator-hours separate. Missing measurements remain null; inference wall time is never renamed GPU-hours.",
              "", "## Limitations", "",
              "The UCI-to-Super-SMS stream simulates dataset shift, not verified chronological concept drift. First-five-window false-alarm exposure is small after excluding calibration. The project ADWIN implementation is approximate; MMD uses TF-IDF/SVD features.",
              "", "The team's LoRA is an offline post-trigger sequence classifier. It is evaluated on held-out sets and is not presented as online recovery. Its classification head/precision differ from the prompt-based Ollama baseline, so the comparison is between systems.",
              "", "The professor-reviewed Step 7 checklist supplied by the user is the requirements source of truth. No additional numeric pass/fail target is invented.",
              "", "## Follow-up inputs", ""]
    if data["paired_comparisons"]:
        lines[-2:-2] = ["## Paired held-out results", "",
                        "| Method | Seed | Population | Accuracy change (pp) | F1 change (pp) | F1 retention (%) |",
                        "|---|---:|---|---:|---:|---:|"] + [
                            f"| {r['method']} | {r['seed']} | {r['population']} | {r['accuracy_change_pp']:.2f} | {r['f1_change_pp']:.2f} | "
                            + (f"{r['f1_retention_pct']:.2f}" if r["f1_retention_pct"] is not None else "N/A") + " |"
                            for r in data["paired_comparisons"]] + [""]
    if data["detectors"]:
        lines[-2:-2] = ["## Detector results by seed", "",
                        "| Method | Seed | Family | Variant | Detected | Delay (messages) | False alerts / 1,000 | False-alarm windows (%) |",
                        "|---|---:|---|---|---|---:|---:|---:|"] + [
                            f"| {r['method']} | {r['seed']} | {r['family']} | {r['detector']} | {r['detected']} | {r['detection_delay_messages'] if r['detected'] else 'miss'} | {r['false_alarms_per_1000']:.2f} | {r['false_alarm_window_pct']:.2f} |"
                            for r in data["detectors"]] + [""]
        lines[-2:-2] = ["Detector mean/SD and observed counts are in detector_summary.csv. Delay averages include detected runs only; detection rate includes misses. No across-seed error bars can be estimated from one seed.", ""]
    for method in data["config"]["methods"]:
        trainer = method.get("trainer")
        if trainer:
            lines[-2:-2] = ["## Team LoRA trainer", "",
                            f"Located `{trainer['path']}` in `{trainer['source_branch']}` at `{trainer['source_ref']}`; dependency: `" + "`, `".join(trainer["dependencies"]) + "`.",
                            f"Status: `{trainer['status']}`. {trainer['note']}",
                            "See the trainer integration notes in BENCHMARK_README.md. Team summary-only results are not substituted for audited per-message predictions.", ""]
    for status in data["availability"]:
        if status["status"] not in ("complete", "smoke", "not_requested"):
            lines.append(f"- {status['method']} / {status['seed']} / {status['population']}: {status['status']}. {status['detail']}")
    pending_baseline = [r for r in data["availability"] if r["method"] == data["config"]["baseline_method"] and r["status"] == "missing"]
    needs_verification = any(r["status"] == "needs_verification" for r in data["availability"])
    if needs_verification and not data["smoke_limit"]:
        lines += ["", "## Next command: readiness check only", "", "```bash",
                  "python scripts/run_benchmark.py --require-complete", "```", "",
                  "Expected exit code: 2 until missing/verification-pending artifacts are resolved. This launches no model inference or training. Recover and verify seed-29822 held-out artifacts, align the comparison protocol, and integrate the located trainer before scheduling repeats. Paid compute awaits the user's approval."]
    elif pending_baseline and not data["smoke_limit"]:
        seed = pending_baseline[0]["seed"]
        lines += ["", "## Next local inference command (not executed)", "", "```bash",
                  f"python scripts/run_benchmark.py --run-baseline --seeds {seed}", "```", "",
                  "For seed 29822 the configured reuse entry skips stream inference and runs only the two held-out populations. External inputs are validated before any inference. LoRA and paid compute remain pending."]
    (output / "BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/benchmark")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--baseline-runs-dir", type=Path, help="Override baseline artifact location")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run-baseline", action="store_true", help="Explicitly run full Ollama inference for selected seeds")
    mode.add_argument("--smoke", action="store_true", help="Run a small isolated Ollama integration check")
    parser.add_argument("--smoke-limit", type=int, default=3, help="Messages per population per seed (1-20); used only with --smoke")
    parser.add_argument("--require-complete", action="store_true", help="Exit 2 if any method/seed/population is missing or incomplete")
    args = parser.parse_args(argv)
    if not 1 <= args.smoke_limit <= 20:
        parser.error("--smoke-limit must be between 1 and 20")
    config = read_json(resolve(args.config))
    if args.seeds:
        config["seeds"] = args.seeds
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        parser.error("Seeds must be a nonempty unique list")
    names = [m["name"] for m in config["methods"]]
    if config.get("primary_metric", "f1") != "f1":
        parser.error("Step 7 uses spam F1 as its primary metric")
    if "detector_families" in config:
        from benchmark_detectors import DETECTOR_FAMILIES
        variants = [(variant, family) for family, items in config["detector_families"].items() for variant in items]
        if len(variants) != 4 or dict(variants) != DETECTOR_FAMILIES:
            parser.error("Use the three detector families: ADWIN, statistical KS/PSI, and embedding MMD")
    if len(names) != len(set(names)) or config["baseline_method"] not in names:
        parser.error("Method names must be unique and include the baseline")
    if any(not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name) for name in names):
        parser.error("Method names must use only letters, numbers, underscores or hyphens")
    for method in config["methods"]:
        requested = method.get("populations", list(POPULATIONS))
        if not requested or len(set(requested)) != len(requested) or any(p not in POPULATIONS for p in requested):
            parser.error("Each method must request unique, known evaluation populations")
        reuse = method.get("reuse_runs", {})
        if not isinstance(reuse, dict) or any(not isinstance(overrides, dict) or any(p not in requested for p in overrides) for overrides in reuse.values()):
            parser.error("reuse_runs must map seed strings to requested populations and read-only run directories")
    output = resolve(args.output_dir)
    if args.smoke:
        # Keep smoke results physically separate even if the default output is used.
        output = output / "smoke"
        config["seeds"] = args.seeds or [config["seeds"][0]]
        config["methods"] = [m for m in config["methods"] if m["name"] == config["baseline_method"]]
        config["methods"][0]["runs_dir"] = str(output / "runs")
        config["methods"][0].pop("reuse_runs", None)
    elif args.baseline_runs_dir:
        next(m for m in config["methods"] if m["name"] == config["baseline_method"])["runs_dir"] = str(args.baseline_runs_dir)
    output.mkdir(parents=True, exist_ok=True)
    stream, sources, manifest, hashes = load_inputs(resolve(config["processed_dir"]), resolve(config["stream_dir"]))
    expected = populations(stream, sources)
    print(f"INPUT PASS: {len(stream)} stream messages; {len(expected['uci_test'])} UCI and {len(expected['super_sms_test'])} Super SMS test messages", flush=True)
    limit = args.smoke_limit if args.smoke else 0
    if args.smoke or args.run_baseline:
        method = next(m for m in config["methods"] if m["name"] == config["baseline_method"])
        for seed in config["seeds"]:
            run_baseline_seed(method, seed, expected, sources, manifest, hashes, limit=limit)
    runs, availability, windows, detectors, provenance = collect(config, expected, sources, manifest, hashes, output, limit=limit)
    historical_audit = [] if limit else audit_existing_results(config, expected, sources, manifest, ROOT)
    mark_pending_verification(availability, historical_audit)
    summary, paired = summarize(runs, config)
    detector_summary = grouped_summary(detectors, ("method", "family", "detector"),
                                       ("detected", "detection_delay_messages", "detection_delay_windows", "false_alarms_per_1000", "false_alarm_window_pct"))
    completed = {m["name"]: complete_model_seeds(runs, m) for m in config["methods"]}
    complete_pairs = [r for r in paired if r["seed"] in completed[r["method"]] & completed[config["baseline_method"]]]
    paired_summary = grouped_summary(complete_pairs, ("method", "population"),
                                     ("accuracy_retention_pct", "accuracy_change_pp", "f1_retention_pct", "f1_change_pp"))
    table = comparison_table(summary, config)
    adaptation, adaptation_summary = adaptation_records(runs, config)
    compute = compute_records(provenance)
    resources = resource_records(availability, runs, compute, config)
    resource_summary = grouped_summary(
        [r for r in resources if r["status"] in ("complete", "smoke") and (r["complete_model_seed"] or r["population"] == "stream")],
        ("method", "population", "hardware", "cost_currency"),
        ("training_wall_seconds", "inference_wall_seconds", "peak_memory_mb", "accelerator_hours_measured", "compute_cost", "trainable_parameters", "adapter_size_mb"))
    data = {"schema_version": 3, "smoke_limit": limit, "config": config,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "implementation_sha256": {name: sha256(ROOT / "scripts" / name) for name in
                                      ("benchmark_common.py", "benchmark_detectors.py", "benchmark_history.py", "benchmark_reporting.py", "run_benchmark.py", "run_harness.py", "evaluate_baseline.py", "detect_drift.py")},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "matplotlib")}},
            "stream_manifest": manifest, "source_sha256": hashes, "provenance": provenance,
            "availability": availability, "runs": runs, "summary": summary, "benchmark_table": table,
            "paired_comparisons": paired, "paired_summary": paired_summary,
            "detectors": detectors, "detector_summary": detector_summary, "compute": compute,
            "adaptation": adaptation, "adaptation_summary": adaptation_summary,
            "resources": resources, "resource_summary": resource_summary, "historical_audit": historical_audit}
    write_json(output / "benchmark.json", data)
    write_csv(output / "availability.csv", availability)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "benchmark_table.csv", table)
    write_csv(output / "per_seed_metrics.csv", [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in runs])
    write_json(output / "historical_audit.json", historical_audit)
    write_csv(output / "adaptation.csv", adaptation)
    write_csv(output / "adaptation_summary.csv", adaptation_summary)
    write_csv(output / "resources.csv", resources)
    write_csv(output / "resource_summary.csv", resource_summary)
    write_csv(output / "paired_summary.csv", paired_summary, ["method", "population", "metric", "n", "mean", "std"])
    write_csv(output / "detector_summary.csv", detector_summary, ["method", "family", "detector", "metric", "n", "mean", "std"])
    write_csv(output / "paired_comparisons.csv", paired, ["method", "seed", "population", "accuracy_retention_pct", "accuracy_change_pp", "f1_retention_pct", "f1_change_pp"])
    write_csv(output / "detector_metrics.csv", detectors, ["method", "seed", "family", "detector", "false_alert_count", "pre_drift_messages_evaluated", "pre_drift_windows_evaluated", "false_alarms_per_1000", "false_alarm_window_pct", "detected", "detection_delay_messages", "detection_delay_windows", "first_alert_position", "total_alerts"])
    write_csv(output / "compute.csv", data["compute"])
    write_csv(output / "window_metrics.csv", [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in windows])
    make_plot(output, windows, manifest, bool(limit))
    make_comparison_plot(output, table, bool(limit))
    write_report(output, data)
    print(f"Saved {output / 'BENCHMARK_REPORT.md'}", flush=True)
    complete = all(r["status"] in ("complete", "smoke", "not_requested") for r in availability)
    return 2 if args.require_complete and not complete else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        sys.exit(1)
