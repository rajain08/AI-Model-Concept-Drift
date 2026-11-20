"""Audit historical team summaries without treating them as prediction evidence."""

import csv
import hashlib
import io
import json
import math
import subprocess
from collections import Counter

from benchmark_common import score, validate_predictions


def git_bytes(root, ref, path):
    result = subprocess.run(["git", f"--git-dir={root / '.git'}", f"--work-tree={root}",
                             "show", f"{ref}:{path}"], capture_output=True, timeout=15)
    if result.returncode:
        raise FileNotFoundError(f"Not available in fetched Git objects: {ref}:{path}")
    return result.stdout


def check_summary(metrics, expected):
    """Check counts against current class totals, including invalid ham/spam."""
    totals = Counter(r["label"] for r in expected)
    issues = []
    counts = {key: metrics[key] for key in ("tp", "tn", "fp", "fn", "invalid")}
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in counts.values()):
        raise ValueError("Confusion counts must be nonnegative integers")
    tp, tn, fp, fn, invalid = (counts[key] for key in ("tp", "tn", "fp", "fn", "invalid"))
    invalid_spam = totals["spam"] - tp - fn
    invalid_ham = totals["ham"] - tn - fp
    if metrics["total"] != len(expected) or sum(counts.values()) != len(expected):
        issues.append("Summary count does not match the held-out population")
    if invalid_spam < 0 or invalid_ham < 0 or invalid_spam + invalid_ham != invalid:
        issues.append("Confusion matrix and invalid counts do not match actual ham/spam totals")
    # The historical scripts excluded invalid outputs from their binary FN count.
    missed_spam = fn + max(0, invalid_spam)
    derived = {
        "accuracy": (tp + tn) / len(expected),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + missed_spam) if tp + missed_spam else 0.0,
        "f1": 2 * tp / (2 * tp + fp + missed_spam) if 2 * tp + fp + missed_spam else 0.0,
        "false_positive_rate": fp / totals["ham"] if totals["ham"] else None,
    }
    for key, value in derived.items():
        if key in metrics and value is not None and not math.isclose(float(metrics[key]), value, abs_tol=1e-6):
            issues.append(f"Reported {key} differs from the current scoring policy/counts")
    return derived, issues


def audit_existing_results(config, expected, sources, manifest, root):
    audits = []
    source_map = {r["message_id"]: r for r in sources}
    from run_harness import choose_examples

    for entry in config.get("historical_results", []):
        if entry["seed"] not in config["seeds"]:
            continue
        method = entry["method"]
        base = {"method": method, "seed": entry["seed"], "source_ref": entry["git_ref"],
                "summary_path": entry["metrics_path"], "accepted_for_aggregation": False}
        try:
            raw = git_bytes(root, entry["git_ref"], entry["metrics_path"])
            summary = json.loads(raw)
            source_hash = hashlib.sha256(raw).hexdigest()
            metadata_issues = []
            protocol_notes = []
            if summary.get("seed") != entry["seed"]:
                metadata_issues.append("Summary seed mismatch")
            if entry["kind"] == "baseline":
                sections = {"uci_test": summary["results"]["uci"], "super_sms_test": summary["results"]["super_sms"]}
                ids = summary.get("shot_message_ids", [])
                if summary.get("model") != "llama3.2:3b" or summary.get("shots") != 15 or len(ids) != 15 or len(set(ids)) != 15:
                    metadata_issues.append("Expected a 15-shot Llama 3.2 3B baseline")
                if any(mid not in source_map or source_map[mid]["split"] != "train" for mid in ids):
                    metadata_issues.append("Prompt includes unknown or non-training IDs")
                current = [r["message_id"] for r in choose_examples(sources, {r["message_id"] for r in expected["stream"]}, entry["seed"])]
                if ids != current:
                    protocol_notes.append("Historical held-out prompt examples differ from the current stream-excluding prompt. Align the held-out protocol before combining seeds.")
                resources = {"training_wall_seconds": 0, "trainable_parameters": 0}
            else:
                sections = {"uci_test": summary["evaluation"]["pre_drift_uci_test"],
                            "super_sms_test": summary["evaluation"]["post_drift_super_sms_test"]}
                if summary.get("stream_sha256") != manifest["stream_csv_sha256"]:
                    metadata_issues.append("LoRA stream hash mismatch")
                trigger = summary.get("drift_trigger", {})
                protocol_notes.append(f"Historical trigger: {trigger.get('detector')} at position {trigger.get('trigger_stream_position')}. This used the legacy detector protocol; it is not equivalent to the current calibration/alert-timing protocol.")
                protocol_notes.append("Training/validation ID lists and source hashes are not present in the summary; the zero-leakage claim cannot be independently confirmed.")
                run_ref = summary.get("git_commit")
                if run_ref:
                    try:
                        git_bytes(root, run_ref, "scripts/train_lora.py")
                    except FileNotFoundError:
                        protocol_notes.append("The recorded run commit lacks train_lora.py; confirm the exact executed trainer source.")
                resources = summary.get("compute", {})
            for population, metrics in sections.items():
                derived, arithmetic_issues = check_summary(metrics, expected[population])
                issues = metadata_issues + arithmetic_issues
                prediction_path = entry["prediction_paths"][population]
                prediction_raw = None
                prediction_source = None
                # Search the declared local artifact and the pinned team's Git tree.
                local = root / prediction_path
                if local.exists():
                    prediction_raw = local.read_bytes()
                    prediction_source = str(local)
                else:
                    try:
                        prediction_raw = git_bytes(root, entry["git_ref"], prediction_path)
                        prediction_source = f"{entry['git_ref']}:{prediction_path}"
                    except FileNotFoundError:
                        pass
                verified = None
                status = "needs_prediction_artifacts"
                if prediction_raw is not None:
                    rows = list(csv.DictReader(io.StringIO(prediction_raw.decode("utf-8-sig"))))
                    try:
                        validate_predictions(rows, expected[population])
                        verified = score(rows)
                        for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "pr_auc"):
                            if metrics.get(key) is not None and verified.get(key) is not None and not math.isclose(float(metrics[key]), verified[key], abs_tol=1e-6):
                                issues.append(f"Predictions disagree with reported {key}")
                        status = "predictions_validated_needs_provenance"
                    except (ValueError, KeyError) as exc:
                        status = "prediction_validation_failed"
                        issues.append(str(exc))
                if issues:
                    status = "verification_issues"
                per_population_resources = dict(resources)
                if entry["kind"] == "baseline":
                    per_population_resources["inference_wall_seconds"] = metrics.get("runtime_seconds")
                else:
                    per_population_resources["inference_wall_seconds"] = resources.get(
                        "uci_test_eval_seconds" if population == "uci_test" else "super_sms_test_eval_seconds")
                audits.append({**base, "population": population, "summary_sha256": source_hash,
                               "status": status, "summary_arithmetic_passed": not arithmetic_issues,
                               "issues": issues, "protocol_notes": protocol_notes,
                               "reported": metrics, "derived_from_summary_counts": derived,
                               "reported_resources": per_population_resources,
                               "prediction_source": prediction_source,
                               "prediction_sha256": hashlib.sha256(prediction_raw).hexdigest() if prediction_raw else None,
                               "verified_prediction_metrics": verified,
                               "required_prediction_path": prediction_path})
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            for population in ("uci_test", "super_sms_test"):
                audits.append({**base, "population": population, "status": "audit_unavailable", "issues": [str(exc)],
                               "protocol_notes": [], "summary_arithmetic_passed": False, "reported": {},
                               "reported_resources": {}, "verified_prediction_metrics": None})
    return audits


def mark_pending_verification(availability, audits):
    """A historical run awaiting artifacts is not automatically a new experiment."""
    lookup = {(a["method"], a["seed"], a["population"]): a for a in audits}
    for row in availability:
        audit = lookup.get((row["method"], row["seed"], row["population"]))
        if row["status"] == "missing" and audit and audit["status"] != "audit_unavailable":
            row.update(status="needs_verification", detail="Existing team result found; retrieve predictions/provenance and resolve protocol notes before deciding on a repeat.")
