"""Readable Step 7 tables; missing evidence is never rendered as a measured zero."""

METRICS = ("f1", "accuracy", "precision", "recall", "false_positive_rate", "pr_auc")
HEADERS = ("Spam F1 (primary)", "Accuracy", "Precision", "Spam recall", "FPR", "PR-AUC")


def number(value, scale=1):
    return "N/A" if value is None else f"{scale * value:.2f}"


def mean_sd(row, scale=1):
    if row["mean"] is None:
        return "N/A (n=0)"
    spread = f" +/- {scale * row['std']:.2f}" if row["std"] is not None else " (SD N/A)"
    return f"{scale * row['mean']:.2f}{spread}; n={row['n']}"


def classification_sections(data):
    lines = ["## Completed populations by seed", "",
             "Percentages, including PR-AUC x 100. A completed held-out model seed requires both UCI and Super SMS; individual completed populations remain visible below. Stream diagnostics are independent.", "",
             "| Method | Seed | Population | " + " | ".join(HEADERS) + " |",
             "|---|---:|---|" + "---:|" * len(METRICS)]
    for row in data["runs"]:
        values = []
        for metric in METRICS:
            value = number(row[metric], 100)
            if row[metric] is None:
                value += ": " + row.get(f"{metric}_unavailable_reason", "Not recorded")
            values.append(value)
        lines.append(f"| {row['method']} | {row['seed']} | {row['population']} | " + " | ".join(values) + " |")
    if not data["runs"]:
        lines += ["", "No completed populations."]
    lines += ["", "## Benchmark table with error bars", "",
              "Mean +/- sample SD across completed model seeds for held-out populations; stream seeds are counted separately. Values are percentages and SD is in percentage points. SD is N/A for n=1. PR-AUC reports its own scored-seed count; label-only runs have no invented scores.", "",
              "| Model | Population | Completed seeds | " + " | ".join(HEADERS) + " |",
              "|---|---|---|" + "---:|" * len(METRICS)]
    for row in data["benchmark_table"]:
        values = []
        for metric in METRICS:
            cell = mean_sd({key: row[f"{metric}_{key}"] for key in ("n", "mean", "std")}, 100)
            if row["status"] == "not_requested":
                cell = "Not requested"
            elif row[f"{metric}_mean"] is None:
                cell += ": " + (row.get(f"{metric}_unavailable_reasons") or "Pending verified runs")
            values.append(cell)
        lines.append(f"| {row['label']} | {row['population']} | {row['seeds'] or 'none'} ({row['n']}/{row['expected_seeds']}) | " + " | ".join(values) + " |")
    return lines


def adaptation_sections(data):
    lines = ["", "## Held-out adaptation outcomes", "",
             "Super SMS measures post-drift generalization; UCI measures original-domain retention. Delta is relative to the same-seed baseline. Retention is 100 x adapted F1 / baseline F1 (uncapped). N/A means predictions or a nonzero baseline denominator are unavailable.", "",
             "| Method | Seed | Complete model / paired seed | Super SMS F1 (%) | Change from baseline (pp) | UCI F1 retention (%) |",
             "|---|---:|---|---:|---:|---:|"]
    for row in data["adaptation"]:
        lines.append(f"| {row['method']} | {row['seed']} | {row['complete_model_seed']} / {row['complete_paired_seed']} | {number(row['super_sms_f1'], 100)} | {number(row['super_sms_f1_change_pp'])} | {number(row['uci_f1_retention_pct'])} |")
    lines += ["", "| Method | Outcome | Mean +/- sample SD; completed seeds |", "|---|---|---|"]
    for row in data["adaptation_summary"]:
        lines.append(f"| {row['method']} | {row['metric']} | {mean_sd(row, 100 if row['metric'] == 'super_sms_f1' else 1)} |")
    return lines


def resource_sections(data):
    lines = ["", "## Time, memory, and adapter resources", "",
             "Inference is scoped to each population. Training, peak memory, adapter size, and cost describe the shared run named by measurement_scope in resources.csv; repeated cells must not be summed. N/A means not measured, except the baseline has no training or trainable parameters and no adapter. Hardware differs across historical runs; wall time is not GPU time.", "",
             "| Method | Seed | Population / status | Training (s) | Inference (s) | Peak memory (MB) | Measured accelerator (h) | Cost | Trainable parameters | Adapter (MB) |",
             "|---|---:|---|---:|---:|---:|---:|---|---:|---:|"]
    for row in data["resources"]:
        cost = "N/A" if row["compute_cost"] is None else f"{number(row['compute_cost'])} {row['cost_currency']}"
        count = "N/A" if row["trainable_parameters"] is None else f"{int(row['trainable_parameters']):,}"
        values = [number(row[key]) for key in ("training_wall_seconds", "inference_wall_seconds", "peak_memory_mb", "accelerator_hours_measured")]
        lines.append(f"| {row['method']} | {row['seed']} | {row['population']} / {row['status']} | " + " | ".join(values + [cost, count, number(row['adapter_size_mb'])]) + " |")
    lines += ["", "Resource mean/sample SD and measurement counts are exported to resource_summary.csv, grouped by method, population, hardware, and currency. Held-out aggregates require complete model seeds; stream diagnostics remain separate. Missing measurements have n=0."]
    return lines


def history_sections(data):
    audits = data["historical_audit"]
    if not audits:
        return []
    lines = ["", "## Existing seed-29822 held-out results: verification pending", "",
             "These are team-reported historical results, excluded from audited tables and error bars. Count arithmetic can be checked against current class totals; message membership, scores, and execution provenance require the original artifacts. PR-AUC below is reported, not independently recomputed.", "",
             "| Method | Population | Arithmetic | Verification status | " + " | ".join(HEADERS) + " |",
             "|---|---|---|---|" + "---:|" * len(METRICS)]
    for row in audits:
        values = []
        for metric in METRICS:
            value = row.get("derived_from_summary_counts", {}).get(metric) if metric != "pr_auc" else row["reported"].get(metric)
            values.append(number(value, 100) if value is not None else "N/A: no continuous-score artifacts")
        lines.append(f"| {row['method']} | {row['population']} | {'PASS' if row['summary_arithmetic_passed'] else 'FAIL'} | {row['status']} | " + " | ".join(values) + " |")
    lookup = {(a["method"], a["population"]): a for a in audits}
    baseline = data["config"]["baseline_method"]
    for method in data["config"]["methods"]:
        name = method["name"]
        if name == baseline:
            continue
        pairs = [(lookup.get((baseline, p)), lookup.get((name, p))) for p in ("uci_test", "super_sms_test")]
        if all(a and b and a["summary_arithmetic_passed"] and b["summary_arithmetic_passed"] for a, b in pairs):
            (pre_base, pre), (post_base, post) = pairs
            pre_den = pre_base["derived_from_summary_counts"]["f1"]
            ratio = 100 * pre["derived_from_summary_counts"]["f1"] / pre_den if pre_den else None
            change = 100 * (post["derived_from_summary_counts"]["f1"] - post_base["derived_from_summary_counts"]["f1"])
            lines += ["", f"Historical summary arithmetic only: {name} Super SMS F1 change **{change:+.2f} pp**; UCI F1 retention **{number(ratio)}%**. These are not accepted paired benchmark results; protocol alignment remains unresolved."]
    lines += ["", "| Historical method | Population | Reported training (s) | Reported inference (s) | Peak memory (MB) | Trainable parameters | Adapter (MB) |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in audits:
        resources = row["reported_resources"]
        vals = [number(resources.get(k)) for k in ("training_wall_seconds", "inference_wall_seconds", "peak_memory_mb", "trainable_parameters", "adapter_size_mb")]
        lines.append(f"| {row['method']} | {row['population']} | " + " | ".join(vals) + " |")
    for row in audits:
        lines += ["", f"- {row['method']} / {row['population']}: `{row['source_ref']}:{row['summary_path']}`."]
        if row.get("required_prediction_path"):
            lines.append(f"  Required original predictions: `{row['required_prediction_path']}`.")
        for note in row["issues"] + row["protocol_notes"]:
            lines.append(f"  {note}")
    lines += ["", "Retrieve the existing predictions, continuous LoRA scores, exact executed trainer, training/validation IDs, input/model hashes, and trigger provenance before deciding whether seed 29822 needs a repeat. Historical baseline prompt examples and LoRA trigger protocol differ from this benchmark. All detailed checks and source hashes are in historical_audit.json."]
    return lines
