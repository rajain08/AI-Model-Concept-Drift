"""Scientific correctness and integration checks; no model downloads or inference."""

import copy
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_common import aggregate, retention, score, sha256, validate_predictions, write_csv, write_json
from benchmark_detectors import evaluate_detectors, summarize_alerts
from benchmark_history import audit_existing_results, check_summary, mark_pending_verification
from benchmark_reporting import classification_sections
from run_benchmark import adaptation_records, artifact_folder, audit_run, baseline_inference, collect, comparison_table, compute_records, grouped_summary, main, resource_records, run_baseline_seed, summarize
from run_harness import choose_examples


def fixture():
    stream = []
    for pos in range(1, 71):
        window = (pos - 1) // 10 + 1
        post = window >= 6
        stream.append({"message_id": f"stream-{pos}", "stream_position": str(pos),
                       "window": str(window), "phase": "post_drift" if post else "pre_drift",
                       "post_drift_fraction": "1.0" if post else "0.0",
                       "dataset": "super_sms" if post else "uci_sms_spam",
                       "label": "spam" if pos % 5 == 0 else "ham",
                       "text": f"{'urgent prize offer' if post else 'meeting friend plans'} token{pos} topic{pos % 7}",
                       "split": "train"})
    sources = stream + [{"message_id": f"pool-{label}-{i}", "dataset": "uci_sms_spam", "split": "train",
                         "label": label, "text": f"example {label} {i}"}
                        for label in ("spam", "ham") for i in range(20)]
    targets = {"stream": stream}
    for dataset, name in (("uci_sms_spam", "uci_test"), ("super_sms", "super_sms_test")):
        targets[name] = [{"message_id": f"{dataset}-{label}", "dataset": dataset, "label": label,
                          "split": "test", "text": f"test {dataset} {label}"} for label in ("ham", "spam")]
        sources += targets[name]
    sources.append({"message_id": "val-1", "dataset": "uci_sms_spam", "label": "ham", "split": "val", "text": "validation"})
    manifest = {"seed": 29822, "stream_csv_sha256": "stream-hash", "drift_onset_position": 51, "drift_onset_window": 6}
    hashes = {"uci_sms_spam": "uci-hash", "super_sms": "super-hash"}
    return stream, sources, targets, manifest, hashes


def predictions(rows):
    return [{**r, "prediction": r["label"], "latency_seconds": "0.5", "ollama_eval_seconds": "0.2"} for r in rows]


def adapted_config(seed, manifest, hashes):
    return {"seed": seed, "model": "test-classifier", "stream_sha256": manifest["stream_csv_sha256"],
            "source_sha256": hashes, "training_ids": [], "replay_ids": [], "retrieval_ids": [],
            "validation_ids": ["val-1"], "evaluation_mode": "offline"}


class ScoringTests(unittest.TestCase):
    def test_fpr_denominator_includes_invalid_actual_ham(self):
        rows = [{"label": "ham", "prediction": pred} for pred in ("spam", "ham", "invalid")]
        self.assertAlmostEqual(score(rows)["false_positive_rate"], 1 / 3)
        only_spam = score([{"label": "spam", "prediction": "spam"}])
        self.assertIsNone(only_spam["false_positive_rate"])
        self.assertIn("No actual ham", only_spam["false_positive_rate_unavailable_reason"])

    def test_pr_auc_uses_continuous_scores_and_trapezoidal_area(self):
        rows = [{"label": y, "prediction": "ham", "spam_probability": p}
                for y, p in (("spam", .9), ("ham", .8), ("spam", .1))]
        self.assertAlmostEqual(score(rows)["pr_auc"], 19 / 24)
        self.assertNotAlmostEqual(score(rows)["pr_auc"], 5 / 6)  # AP is different.
        for row in rows:
            row["prediction"] = "spam"
        self.assertAlmostEqual(score(rows)["pr_auc"], 19 / 24)

    def test_pr_auc_unavailable_reasons_are_explicit(self):
        rows = [{"label": "ham", "prediction": "ham"}, {"label": "spam", "prediction": "spam"}]
        self.assertIn("Label-only", score(rows)["pr_auc_unavailable_reason"])
        rows[0]["spam_score"] = .2
        self.assertIn("missing", score(rows)["pr_auc_unavailable_reason"])
        self.assertIsNone(score(rows)["pr_auc"])
        self.assertIn("No actual spam", score(rows[:1])["pr_auc_unavailable_reason"])

    def test_team_lora_probability_column_is_supported_and_checked(self):
        rows = [{"message_id": "ham", "label": "ham", "prediction": "ham", "spam_probability": "0.2"},
                {"message_id": "spam", "label": "spam", "prediction": "spam", "spam_probability": "0.8"}]
        self.assertEqual(score(rows)["roc_auc"], 1)
        rows[0]["spam_score"] = "0.3"
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_predictions(rows, rows)
        rows[0].pop("spam_score")
        rows[0]["spam_probability"] = "nan"
        with self.assertRaises(ValueError):
            validate_predictions(rows, rows)

    def test_abstentions_and_confusion_matrix(self):
        records = [{"label": y, "prediction": p} for y, p in
                   (("spam", "spam"), ("spam", "ham"), ("spam", "invalid"),
                    ("ham", "spam"), ("ham", "ham"), ("ham", "invalid"))]
        result = score(records)
        self.assertAlmostEqual(result["accuracy"], 1 / 3)
        self.assertEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertEqual(result["f1"], 0.4)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 1], [1, 1, 1]])
        self.assertEqual(result["fn"], 2)
        self.assertIsNone(result["roc_auc"])
        self.assertIsNone(result["inference_wall_seconds"])

    def test_auc_requires_real_scores(self):
        rows = [{"label": "ham", "prediction": "spam", "spam_score": "0.2"},
                {"label": "spam", "prediction": "spam", "spam_score": "0.7"}]
        self.assertEqual(score(rows)["roc_auc"], 1)
        rows[1].pop("spam_score")
        self.assertIsNone(score(rows)["roc_auc"])

    def test_empty_and_bad_labels_rejected(self):
        with self.assertRaises(ValueError):
            score([])
        with self.assertRaises(ValueError):
            score([{"label": "spam", "prediction": "maybe"}])

    def test_sample_sd_not_window_spread(self):
        self.assertEqual(aggregate([0.5]), {"n": 1, "mean": 0.5, "std": None})
        self.assertAlmostEqual(aggregate([0.2, 0.4])["std"], math.sqrt(0.02))
        self.assertEqual(aggregate([None])["n"], 0)

    def test_retention_uncapped_and_zero_denominator(self):
        result = retention({"accuracy": 0.5, "f1": 0}, {"accuracy": 0.75, "f1": 0.5})
        self.assertEqual(result["accuracy_retention_pct"], 150)
        self.assertEqual(result["accuracy_change_pp"], 25)
        self.assertIsNone(result["f1_retention_pct"])


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.stream, self.sources, self.targets, self.manifest, self.hashes = fixture()

    def test_prediction_coverage_order_and_labels(self):
        valid = predictions(self.stream)
        validate_predictions(valid, self.stream, stream=True)
        mutations = [valid[:-1], [valid[1], valid[0]] + valid[2:], valid + [valid[0]],
                     [{**valid[0], "label": "spam"}] + valid[1:],
                     [{**valid[0], "window": "9"}] + valid[1:]]
        for records in mutations:
            with self.subTest(records=len(records)), self.assertRaises(ValueError):
                validate_predictions(records, self.stream, stream=True)

    def test_nan_negative_latency_and_out_of_range_scores(self):
        for field, value in (("latency_seconds", "nan"), ("latency_seconds", "-1"), ("spam_score", "1.1")):
            rows = predictions(self.stream)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                validate_predictions(rows, self.stream, stream=True)

    def test_heldout_order_does_not_change_population(self):
        validate_predictions(list(reversed(predictions(self.targets["uci_test"]))), self.targets["uci_test"])

    def test_leakage_and_source_mismatch_rejected(self):
        base = adapted_config(1, self.manifest, self.hashes)
        audit_run(base, 1, self.sources, self.manifest, self.hashes, self.stream, baseline=False)
        for key in ("training_ids", "replay_ids", "retrieval_ids", "validation_ids"):
            bad = copy.deepcopy(base)
            bad[key] = ["uci_sms_spam-spam"]
            with self.assertRaisesRegex(ValueError, "ineligible"):
                audit_run(bad, 1, self.sources, self.manifest, self.hashes, self.stream, baseline=False)
        with self.assertRaises(ValueError):
            audit_run({**base, "source_sha256": {}}, 1, self.sources, self.manifest, self.hashes, self.stream, baseline=False)

    def test_future_training_and_retrieval_rejected(self):
        base = adapted_config(1, self.manifest, self.hashes)
        base.update(evaluation_mode="prequential", usage_events=[{"after_position": 5, "training_ids": ["stream-5"]}])
        audit_run(base, 1, self.sources, self.manifest, self.hashes, self.stream, baseline=False)
        for key in ("training_ids", "retrieval_ids", "replay_ids"):
            bad = copy.deepcopy(base)
            bad["usage_events"] = [{"after_position": 5, key: ["stream-6"]}]
            with self.assertRaisesRegex(ValueError, "Future"):
                audit_run(bad, 1, self.sources, self.manifest, self.hashes, self.stream, baseline=False)

    def test_gpu_hours_require_measurement_provenance(self):
        base = {"method": "test", "seed": 1, "config": {"compute": {"accelerator_hours_measured": 1}}}
        with self.assertRaises(ValueError):
            compute_records([base])
        base["config"]["compute"] = {"training_wall_seconds": 3600}
        self.assertIsNone(compute_records([base])[0]["accelerator_hours_measured"])

    def test_adapter_and_cost_measurements_are_validated(self):
        for measurement in ({"trainable_parameters": 2.5}, {"adapter_size_mb": -1}, {"compute_cost": 3},
                            {"inference_seconds_by_population": {"uci_test": -1}},
                            {"inference_seconds_by_population": {"typo": 1}}):
            with self.subTest(measurement=measurement), self.assertRaises(ValueError):
                compute_records([{"method": "lora", "seed": 1, "config": {"compute": measurement}}])
        result = compute_records([{"method": "lora", "seed": 1, "config": {"compute": {
            "trainable_parameters": 100, "adapter_size_mb": 4, "compute_cost": 3, "cost_currency": "USD"}}}])[0]
        self.assertEqual(result["trainable_parameters"], 100)
        self.assertIsNone(result["training_wall_seconds"])


class HistoricalAuditTests(unittest.TestCase):
    def test_historical_counts_and_invalid_ham_denominator(self):
        expected = [{"label": "ham"}] * 3 + [{"label": "spam"}]
        metrics = {"total": 4, "tp": 1, "tn": 1, "fp": 1, "fn": 0, "invalid": 1,
                   "accuracy": .5, "f1": 2 / 3, "precision": .5, "recall": 1}
        derived, issues = check_summary(metrics, expected)
        self.assertEqual(issues, [])
        self.assertAlmostEqual(derived["false_positive_rate"], 1 / 3)
        self.assertTrue(check_summary({**metrics, "f1": .9}, expected)[1])
        self.assertTrue(check_summary({**metrics, "total": 5}, expected)[1])

    def test_summary_only_run_is_not_accepted_or_scheduled_as_missing(self):
        stream, sources, targets, manifest, _ = fixture()
        metrics = {"total": 2, "tp": 1, "tn": 1, "fp": 0, "fn": 0, "invalid": 0, "f1": 1}
        summary = {"seed": 1, "model": "llama3.2:3b", "shots": 15,
                   "shot_message_ids": [r["message_id"] for r in choose_examples(sources, {r["message_id"] for r in stream}, 1)],
                   "results": {"uci": metrics, "super_sms": metrics}}
        config = {"seeds": [1], "historical_results": [{"method": "baseline", "kind": "baseline", "seed": 1,
                  "git_ref": "pinned", "metrics_path": "metrics.json", "prediction_paths": {
                      "uci_test": "uci.csv", "super_sms_test": "super.csv"}}]}
        def read_git(root, ref, path):
            if path == "metrics.json":
                return json.dumps(summary).encode()
            raise FileNotFoundError(path)
        with tempfile.TemporaryDirectory() as temp, patch("benchmark_history.git_bytes", side_effect=read_git):
            audits = audit_existing_results(config, targets, sources, manifest, Path(temp))
        self.assertEqual(len(audits), 2)
        self.assertTrue(all(a["summary_arithmetic_passed"] for a in audits))
        self.assertTrue(all(not a["accepted_for_aggregation"] and a["verified_prediction_metrics"] is None for a in audits))
        availability = [{"method": "baseline", "seed": 1, "population": p, "status": "missing"} for p in targets]
        mark_pending_verification(availability, audits)
        self.assertEqual([a["status"] for a in availability], ["missing", "needs_verification", "needs_verification"])

    def test_bad_prediction_artifacts_are_rejected_even_when_summary_matches(self):
        _, sources, targets, manifest, _ = fixture()
        summary = {"seed": 1, "stream_sha256": manifest["stream_csv_sha256"], "evaluation": {
            p: {"total": 2, "tp": 1, "tn": 1, "fp": 0, "fn": 0, "invalid": 0}
            for p in ("pre_drift_uci_test", "post_drift_super_sms_test")}}
        config = {"seeds": [1], "historical_results": [{"method": "lora", "kind": "lora", "seed": 1,
                  "git_ref": "pinned", "metrics_path": "metrics.json", "prediction_paths": {
                      "uci_test": "uci.csv", "super_sms_test": "super.csv"}}]}
        def read_git(root, ref, path):
            return json.dumps(summary).encode() if path.endswith("json") else b"message_id,label,prediction\nwrong,ham,ham\n"
        with tempfile.TemporaryDirectory() as temp, patch("benchmark_history.git_bytes", side_effect=read_git):
            audits = audit_existing_results(config, targets, sources, manifest, Path(temp))
        self.assertTrue(all(a["status"] == "verification_issues" and a["issues"] for a in audits))


class DetectorTests(unittest.TestCase):
    def test_delay_uses_observation_time_and_excludes_calibration(self):
        stream, _, _, manifest, _ = fixture()
        alerts = [{"detector": "ks", "stream_position": 40, "window": 4},
                  {"detector": "ks", "stream_position": 60, "window": 6}]
        result = summarize_alerts(alerts, stream, manifest["drift_onset_position"], 31)
        self.assertEqual(result["ks"]["detection_delay_messages"], 9)
        self.assertEqual(result["ks"]["false_alarms_per_1000"], 50)
        self.assertEqual(result["ks"]["false_alarm_window_pct"], 50)
        self.assertIsNone(result["adwin"]["detection_delay_messages"])
        self.assertFalse(result["adwin"]["detected"])

    def test_real_detector_pipeline_is_causal(self):
        stream, _, _, manifest, _ = fixture()
        first, alerts, signals = evaluate_detectors(stream, predictions(stream), manifest, 42, bootstrap_samples=10)
        future = copy.deepcopy(stream)
        for row in future[50:]:
            row["text"] = "unseen catastrophic change !" * 20
        second, _, _ = evaluate_detectors(future, predictions(future), manifest, 42, bootstrap_samples=10)
        self.assertEqual(first["thresholds"], second["thresholds"])
        self.assertTrue(all(s["ks"] is None for s in signals[:3]))
        self.assertEqual(first["detectors"]["ks"]["pre_drift_messages_evaluated"], 20)
        self.assertTrue(all(a["stream_position"] == a["window"] * 10 for a in alerts if a["detector"] != "adwin"))
        with self.assertRaises(ValueError):
            evaluate_detectors(stream, predictions(stream)[:-1], manifest, 42, bootstrap_samples=10)

    def test_legacy_detector_cannot_invent_missing_errors(self):
        from detect_drift import get_error_stream
        stream, _, _, _, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            write_csv(path / "metrics.csv", [{"window": 1, "accuracy": 1}])
            with self.assertRaises(FileNotFoundError):
                get_error_stream(stream, path / "missing.csv", path / "metrics.csv")
            write_csv(path / "predictions.csv", predictions(stream)[:-1])
            with self.assertRaises(ValueError):
                get_error_stream(stream, path / "predictions.csv", path / "metrics.csv")


class IntegrationTests(unittest.TestCase):
    def test_completed_model_seed_statistics_exclude_partial_holdouts(self):
        config = {"seeds": [1, 2], "baseline_method": "baseline", "methods": [
            {"name": "baseline"}, {"name": "lora", "populations": ["uci_test", "super_sms_test"]}]}
        perfect = score([{"label": "ham", "prediction": "ham"}, {"label": "spam", "prediction": "spam"}])
        poor = score([{"label": "ham", "prediction": "spam"}, {"label": "spam", "prediction": "ham"}])
        runs = [{"method": method, "seed": seed, "population": pop, **perfect}
                for method in ("baseline", "lora") for seed in (1, 2) for pop in ("uci_test", "super_sms_test")
                if not (method == "lora" and seed == 2)]
        runs.append({"method": "lora", "seed": 2, "population": "super_sms_test", **poor})
        summary, _ = summarize(runs, config)
        selected = next(r for r in summary if (r["method"], r["population"], r["metric"]) == ("lora", "super_sms_test", "f1"))
        self.assertEqual((selected["n"], selected["mean"], selected["std"]), (1, 1, None))
        adaptation, stats = adaptation_records(runs, config)
        partial = next(r for r in adaptation if r["method"] == "lora" and r["seed"] == 2)
        self.assertEqual(partial["super_sms_f1_change_pp"], -100)
        self.assertFalse(partial["complete_paired_seed"])
        self.assertTrue(all(r["n"] == 1 and r["std"] is None for r in stats if r["method"] == "lora"))
        table = comparison_table(summary, config)
        report = "\n".join(classification_sections({"runs": runs, "benchmark_table": table}))
        for header in ("Spam F1", "Accuracy", "Precision", "Spam recall", "FPR", "PR-AUC", "Label-only"):
            self.assertIn(header, report)

    def test_resources_keep_population_timing_and_unknowns_separate(self):
        config = {"seeds": [1], "baseline_method": "baseline", "methods": [{"name": "baseline"}, {"name": "lora"}]}
        available = [{"method": "lora", "seed": 1, "population": p, "status": "complete"}
                     for p in ("uci_test", "super_sms_test")]
        runs = [{**r, "inference_wall_seconds": None} for r in available]
        compute = compute_records([{"method": "lora", "seed": 1, "populations": ["uci_test", "super_sms_test"],
                                    "config": {"compute": {"training_wall_seconds": 10,
                                        "inference_seconds_by_population": {"uci_test": 2, "super_sms_test": 3}}}}])
        records = resource_records(available, runs, compute, config)
        self.assertEqual([r["inference_wall_seconds"] for r in records], [2, 3])
        self.assertTrue(all(r["training_wall_seconds"] == 10 and r["peak_memory_mb"] is None for r in records))
        self.assertTrue(all(r["measurement_scope"] == "uci_test,super_sms_test" for r in records))

    def test_external_stream_reuse_keeps_holdouts_local_and_never_writes_external(self):
        stream, sources, targets, manifest, hashes = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            external = root / "external" / "seed_1"
            local = root / "local" / "seed_1"
            external.mkdir(parents=True)
            local.mkdir(parents=True)
            meta = {"seed": 1, "model": "llama3.2:3b", "stream_sha256": manifest["stream_csv_sha256"],
                    "prompt_examples": [r["message_id"] for r in choose_examples(sources, {r["message_id"] for r in stream}, 1)]}
            for folder in (external, local):
                write_json(folder / "run_config.json", meta)
            write_csv(external / "predictions.csv", predictions(stream))
            write_csv(local / "predictions.csv", predictions(stream)[:-1])
            for population in ("uci_test", "super_sms_test"):
                write_csv(local / f"{population}_predictions.csv", predictions(targets[population]))
            method = {"name": "baseline", "runs_dir": str(root / "local"), "reuse_runs": {"1": {"stream": str(external)}}}
            config = {"seeds": [1], "baseline_method": "baseline", "methods": [method], "detector_bootstrap_samples": 10}
            before = {p.name: sha256(p) for p in external.iterdir()}
            runs, available, _, detectors, provenance = collect(config, targets, sources, manifest, hashes, root / "report")
            self.assertEqual(len(runs), 3)
            self.assertEqual([r["source"] for r in available], ["reused", "local", "local"])
            self.assertTrue(all(r["status"] == "complete" for r in available))
            self.assertEqual({r["family"] for r in detectors}, {"adwin", "statistical", "embedding"})
            self.assertEqual(len(detectors), 4)
            self.assertEqual(artifact_folder(method, 1, "uci_test"), local)
            self.assertEqual(len([r for r in provenance if "config" in r]), 2)
            with patch("run_benchmark.baseline_inference") as infer:
                run_baseline_seed(method, 1, targets, sources, manifest, hashes)
                self.assertEqual(infer.call_args.kwargs["selected_populations"], ["uci_test", "super_sms_test"])
                self.assertEqual(infer.call_args.args[1], local)
            self.assertEqual(before, {p.name: sha256(p) for p in external.iterdir()})
            # Invalid or missing external inputs must fail; never silently use the Desktop copy.
            write_csv(external / "predictions.csv", predictions(stream)[:-1])
            with patch("run_benchmark.baseline_inference") as infer:
                with self.assertRaisesRegex(ValueError, "Incomplete"):
                    run_baseline_seed(method, 1, targets, sources, manifest, hashes)
                infer.assert_not_called()
            runs, available, _, _, _ = collect(config, targets, sources, manifest, hashes, root / "report")
            self.assertEqual(len(runs), 2)
            self.assertEqual(available[0]["status"], "incomplete")

    def test_comparison_table_exposes_missing_seeds_and_unknown_error_bars(self):
        config = {"seeds": [1, 2, 3], "baseline_method": "baseline", "methods": [
            {"name": "baseline"}, {"name": "drift_triggered_lora", "populations": ["uci_test", "super_sms_test"]}]}
        metrics = score([{"label": "spam", "prediction": "spam"}, {"label": "ham", "prediction": "ham"}])
        summary, _ = summarize([{"method": "baseline", "seed": 1, "population": "stream", **metrics}], config)
        table = comparison_table(summary, config)
        baseline = table[0]
        self.assertEqual((baseline["n"], baseline["expected_seeds"], baseline["f1_mean"]), (1, 3, 1))
        self.assertIsNone(baseline["f1_std"])
        self.assertEqual(table[1]["status"], "not_requested")
        self.assertEqual(table[-1]["status"], "pending")
        self.assertIsNone(table[-1]["f1_mean"])

    def test_default_cli_never_launches_inference(self):
        stream, sources, targets, manifest, hashes = fixture()
        with tempfile.TemporaryDirectory() as temp, \
                patch("run_benchmark.load_inputs", return_value=(stream, sources, manifest, hashes)), \
                patch("run_benchmark.make_plot"), \
                patch("run_benchmark.make_comparison_plot"), \
                patch("run_benchmark.baseline_inference") as inference:
            root = Path(temp)
            config = {"seeds": [1], "baseline_method": "baseline", "processed_dir": "unused",
                      "stream_dir": "unused", "methods": [{"name": "baseline", "runs_dir": str(root / "runs")} ]}
            write_json(root / "config.json", config)
            code = main(["--config", str(root / "config.json"), "--output-dir", str(root / "report"), "--require-complete"])
            self.assertEqual(code, 2)
            inference.assert_not_called()
            self.assertTrue((root / "report" / "BENCHMARK_REPORT.md").exists())

    def test_offline_method_can_request_only_heldout_populations(self):
        stream, sources, targets, manifest, hashes = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"seeds": [1], "baseline_method": "baseline", "methods": [
                {"name": "adapted", "runs_dir": str(root), "populations": ["uci_test", "super_sms_test"]}]}
            folder = root / "seed_1"
            folder.mkdir()
            write_json(folder / "run_config.json", adapted_config(1, manifest, hashes))
            for population in ("uci_test", "super_sms_test"):
                write_csv(folder / f"{population}_predictions.csv", predictions(targets[population]))
            runs, available, _, detectors, _ = collect(config, targets, sources, manifest, hashes, root)
            self.assertEqual(len(runs), 2)
            self.assertEqual([r["status"] for r in available], ["not_requested", "complete", "complete"])
            self.assertEqual(detectors, [])

    def test_artifact_collection_and_paired_seeds(self):
        stream, sources, targets, manifest, hashes = fixture()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"seeds": [1, 2], "baseline_method": "baseline", "methods": [
                {"name": name, "runs_dir": str(root / name)} for name in ("baseline", "adapted")]}
            for method in config["methods"]:
                for seed in config["seeds"]:
                    folder = Path(method["runs_dir"]) / f"seed_{seed}"
                    folder.mkdir(parents=True)
                    meta = adapted_config(seed, manifest, hashes)
                    if method["name"] == "baseline":
                        meta["model"] = "llama3.2:3b"
                        meta["prompt_examples"] = [r["message_id"] for r in choose_examples(sources, {r["message_id"] for r in stream}, seed)]
                    write_json(folder / "run_config.json", meta)
                    for population in ("uci_test", "super_sms_test"):
                        rows = predictions(targets[population])
                        if method["name"] == "baseline":
                            rows[0]["prediction"] = "spam"
                        write_csv(folder / f"{population}_predictions.csv", rows)
            runs, availability, _, _, _ = collect(config, targets, sources, manifest, hashes, root)
            self.assertEqual(len(runs), 8)
            self.assertEqual(sum(a["status"] == "missing" for a in availability), 4)
            summary, paired = summarize(runs, config)
            self.assertEqual(len(paired), 4)
            self.assertTrue(all(p["accuracy_change_pp"] == 50 for p in paired))
            values = next(r for r in summary if r["method"] == "adapted" and r["population"] == "uci_test" and r["metric"] == "accuracy")
            self.assertEqual((values["n"], values["mean"], values["std"]), (2, 1, 0))
            # A missing row must remove that seed, not silently shrink its population.
            folder = root / "adapted" / "seed_1"
            write_csv(folder / "uci_test_predictions.csv", predictions(targets["uci_test"])[:1])
            runs, availability, _, _, _ = collect(config, targets, sources, manifest, hashes, root)
            self.assertEqual(len(runs), 7)
            self.assertEqual(sum(a["status"] == "incomplete" for a in availability), 1)

    def test_inference_resume_does_not_duplicate_or_requery(self):
        stream, sources, targets, manifest, hashes = fixture()
        with tempfile.TemporaryDirectory() as temp, \
                patch("urllib.request.urlopen", side_effect=lambda *a, **k: io.BytesIO(json.dumps({"models": [{"name": "llama3.2:3b", "digest": "test-digest"}]}).encode())), \
                patch("run_harness.predict", return_value=("ham", "ham", 0.1, 0.01)) as predict, \
                patch("run_harness.git_commit", return_value="test-commit"), \
                patch("run_harness.gpu_name", return_value="test-hardware"):
            baseline_inference(1, Path(temp), targets, sources, manifest, hashes, limit=2)
            self.assertEqual(predict.call_count, 6)
            baseline_inference(1, Path(temp), targets, sources, manifest, hashes, limit=2)
            self.assertEqual(predict.call_count, 6)
            with self.assertRaisesRegex(ValueError, "Cannot resume"):
                baseline_inference(1, Path(temp), targets, sources, manifest, hashes, limit=1)

    def test_existing_harness_writes_aggregate_without_inference(self):
        import run_harness
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(sys, "argv", ["run_harness.py", "--limit", "1", "--skip-detectors", "--output-dir", temp]), \
                patch.object(run_harness, "validate_inputs", return_value=([], [], {})), \
                patch.object(run_harness, "run_seed", return_value={k: 0.5 for k in ("accuracy", "precision", "recall", "f1")}):
            run_harness.main()
            result = json.loads((Path(temp) / "aggregate_smoke1.json").read_text())
            self.assertEqual(result["runs"], 1)
            self.assertIsNone(result["metrics"]["f1"]["std"])

    def test_misses_not_averaged_as_zero_delay(self):
        rows = [{"method": "baseline", "detector": "ks", "detected": True, "delay": 99},
                {"method": "baseline", "detector": "ks", "detected": False, "delay": None}]
        result = grouped_summary(rows, ("method", "detector"), ("detected", "delay"))
        self.assertEqual(result[0]["mean"], 0.5)
        self.assertEqual(result[1]["n"], 1)
        self.assertEqual(result[1]["mean"], 99)


if __name__ == "__main__":
    unittest.main()
