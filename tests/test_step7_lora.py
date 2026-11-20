"""CPU-only Step 7 trainer integration checks; no model download or CUDA use."""

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_common import read_json, write_csv, write_json
from run_benchmark import collect, compute_records
import run_step7_lora as lora
from test_benchmark import fixture, predictions
from run_harness import choose_examples


class LoRAIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.stream, self.sources, self.expected, self.manifest, self.hashes = fixture()
        self.sources.append({"message_id": "val-spam", "dataset": "super_sms", "split": "val",
                             "label": "spam", "text": "validation spam"})
        self.summary = {"seed": 1, "stream_sha256": self.manifest["stream_csv_sha256"],
                        "calibration_windows": 3, "bootstrap_samples": 10,
                        "monitoring_start_position": 31, "thresholds": {"mmd": .1}, "detectors": {}}
        self.alerts = [{"detector": "mmd", "window": 6, "stream_position": 60,
                        "metric_value": .2, "threshold": .1}]
        self.settings = read_json(lora.ROOT / "configs/step7_lora.json")
        self.settings["hyperparameters"]["val_limit"] = 2

    def trigger(self):
        return lora.select_trigger(self.summary, self.alerts, self.stream, self.manifest, 1, "mmd")

    def make_plan(self, root):
        baseline = root / "baseline" / "seed_1"
        detector = root / "detectors" / "baseline" / "seed_1"
        baseline.mkdir(parents=True)
        detector.mkdir(parents=True)
        write_json(baseline / "run_config.json", {"seed": 1, "model": "llama3.2:3b",
                   "stream_sha256": self.manifest["stream_csv_sha256"],
                   "prompt_examples": [r["message_id"] for r in choose_examples(self.sources, {r["message_id"] for r in self.stream}, 1)]})
        write_csv(baseline / "predictions.csv", predictions(self.stream))
        write_json(detector / "summary.json", self.summary)
        write_csv(detector / "alerts.csv", self.alerts)
        config = {"seeds": [1, 2], "processed_dir": "data/processed", "stream_dir": "data/processed/stream",
                  "baseline_method": "baseline", "detector_bootstrap_samples": 10,
                  "methods": [{"name": "baseline", "runs_dir": str(root / "baseline")},
                              {"name": "drift_triggered_lora", "runs_dir": "results/benchmark_runs/drift_triggered_lora",
                               "populations": list(lora.HELDOUTS)}]}
        write_json(root / "benchmark.json", config)
        write_json(root / "lora.json", self.settings)
        with patch.object(lora, "evaluate_detectors", return_value=(self.summary, self.alerts, [])):
            return lora.prepare(root / "benchmark.json", root / "lora.json", 1, root / "detectors", root / "prepared")

    def inputs_patch(self):
        return patch.object(lora, "load_inputs", return_value=(self.stream, self.sources, self.manifest, self.hashes))

    def test_trigger_is_observed_window_end_and_training_is_strictly_later(self):
        trigger = self.trigger()
        train, val, _ = lora.build_sets(self.stream, self.sources, trigger, 1, 2)
        self.assertEqual(trigger["training_start_position"], 61)
        self.assertEqual({int(r["stream_position"]) for r in train}, set(range(61, 71)))
        self.assertTrue(all(r["split"] == "val" for r in val))
        self.assertFalse(trigger["fallback_used"])

    def test_false_alarm_triggers_without_using_known_drift_onset(self):
        self.alerts.insert(0, {**self.alerts[0], "window": 4, "stream_position": 40})
        self.assertEqual(self.trigger()["trigger_stream_position"], 40)

    def test_wrong_seed_wrong_stream_early_timestamp_and_no_alert_are_rejected(self):
        cases = [({**self.summary, "seed": 2}, self.alerts),
                 ({**self.summary, "stream_sha256": "wrong"}, self.alerts),
                 (self.summary, [{**self.alerts[0], "stream_position": 51}]),
                 (self.summary, [{**self.alerts[0], "window": 3, "stream_position": 30}]),
                 (self.summary, []),
                 (self.summary, [{**self.alerts[0], "window": 7, "stream_position": 70}])]
        for summary, alerts in cases:
            with self.subTest(summary=summary, alerts=alerts), self.assertRaises(ValueError):
                lora.select_trigger(summary, alerts, self.stream, self.manifest, 1, "mmd")

    def test_cpu_preparation_portable_snapshot_and_deterministic_check(self):
        with tempfile.TemporaryDirectory() as temp, self.inputs_patch():
            root = Path(temp)
            plan_path = self.make_plan(root)
            plan, train, val, expected = lora.check_plan(plan_path)
            self.assertEqual((len(train), len(val), len(expected["uci_test"])), (10, 2, 2))
            self.assertNotIn(str(root), json.dumps(plan))
            self.assertEqual(set(plan["input_sha256"]), set(lora.INPUT_FILES))
            self.assertFalse((root / "prepared" / "run_config.json").exists())
            self.assertEqual(plan["status"], "prepared_not_executed")
            # Checking consumes the portable snapshot, not the live baseline file.
            (root / "baseline/seed_1/predictions.csv").write_text("changed during later inference")
            lora.check_plan(plan_path)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("transformers", sys.modules)

    def test_prepared_input_and_plan_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, self.inputs_patch():
            path = self.make_plan(Path(temp))
            data = read_json(path)
            data["run_config"]["seed"] = 2
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "plan hash"):
                lora.check_plan(path)
            data["run_config"]["seed"] = 1
            write_json(path, data)
            with (path.parent / "alerts.csv").open("a") as handle:
                handle.write("mmd,7,70,0.9,0.1\n")
            with self.assertRaisesRegex(ValueError, "input hash"):
                lora.check_plan(path)

    def test_wrong_training_boundary_rejected_even_with_updated_plan_hash(self):
        with tempfile.TemporaryDirectory() as temp, self.inputs_patch():
            path = self.make_plan(Path(temp))
            data = read_json(path)
            data["run_config"]["training_ids"][0] = "stream-60"
            data["plan_sha256"] = lora.digest({k: v for k, v in data.items() if k != "plan_sha256"})
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "training IDs"):
                lora.check_plan(path)

    def result_fixture(self, plan, output):
        adapter = output / "lora_adapter"
        adapter.mkdir()
        (adapter / "adapter_model.pt").write_bytes(b"synthetic-test-adapter-only")
        stats = {"model_name": plan["run_config"]["model"], "model_revision": plan["run_config"]["model_revision"],
                 "gpu_name": "synthetic-test-device", "dtype": "bfloat16", "optimizer_steps": 2,
                 "training_wall_seconds": 10, "uci_test_eval_seconds": 2, "super_sms_test_eval_seconds": 3,
                 "peak_memory_mb": 123, "trainable_parameters": 456, "calibrated_threshold": .5}
        rows = [[{"message_id": r["message_id"], "label": r["label"], "prediction": r["label"],
                  "spam_probability": .9 if r["label"] == "spam" else .1} for r in self.expected[p]] for p in lora.HELDOUTS]
        return [], stats, *rows, {"lora_adapter_dir": str(adapter)}

    def test_produced_artifacts_pass_actual_benchmark_collection(self):
        with tempfile.TemporaryDirectory() as temp, self.inputs_patch(), patch("importlib.metadata.version", return_value="test"):
            root = Path(temp)
            path = self.make_plan(root)
            plan, _, _, expected = lora.check_plan(path)
            output = root / "lora" / "seed_1"
            output.mkdir(parents=True)
            metadata = lora.finish_artifacts(plan, output, self.result_fixture(plan, output), expected)
            config = {"seeds": [1], "baseline_method": "baseline", "methods": [{"name": "drift_triggered_lora",
                      "runs_dir": str(root / "lora"), "populations": list(lora.HELDOUTS)}]}
            runs, available, _, detectors, provenance = collect(config, expected, self.sources, self.manifest, self.hashes, root)
            self.assertEqual([r["status"] for r in available], ["not_requested", "complete", "complete"])
            self.assertTrue(all(r["pr_auc"] == 1 for r in runs))
            self.assertEqual(detectors, [])
            compute = compute_records(provenance)[0]
            self.assertEqual(compute["inference_seconds_by_population"], {"uci_test": 2, "super_sms_test": 3})
            self.assertIsNone(compute["accelerator_hours_measured"])
            self.assertEqual(metadata["threshold_selection"], "validation_only_spam_f1")
            self.assertGreater(metadata["compute"]["adapter_size_mb"], 0)

    def test_incomplete_unscored_wrong_threshold_and_wrong_model_results_not_published(self):
        for case in ("partial", "missing_score", "threshold", "model"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temp, self.inputs_patch():
                root = Path(temp)
                path = self.make_plan(root)
                plan, _, _, expected = lora.check_plan(path)
                output = root / "output"
                output.mkdir()
                result = self.result_fixture(plan, output)
                if case == "partial":
                    result[2].pop()
                elif case == "missing_score":
                    result[2][0].pop("spam_probability")
                elif case == "threshold":
                    result[2][0]["prediction"] = "spam"
                else:
                    result[1]["model_revision"] = "wrong"
                with self.assertRaises(ValueError):
                    lora.finish_artifacts(plan, output, result, expected)
                self.assertFalse((output / "run_config.json").exists())
                self.assertFalse((output / "uci_test_predictions.csv").exists())

    def test_run_refuses_existing_output_before_importing_torch(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(lora, "check_plan", return_value=({"output_dir": temp}, [], [], {})):
            with self.assertRaises(FileExistsError):
                lora.run_plan(Path("unused"), "cuda")
            self.assertNotIn("torch", sys.modules)

    def test_execution_passes_frozen_settings_and_publishes_only_complete_output(self):
        with tempfile.TemporaryDirectory() as temp, self.inputs_patch(), patch("importlib.metadata.version", return_value="test"):
            root = Path(temp)
            path = self.make_plan(root)
            checked = lora.check_plan(path)
            plan = checked[0]
            output = root / "finished" / "seed_1"
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True))
            def fake_train(args, train, val, uci, super_sms, pending):
                self.assertEqual(args.seed, 1)
                self.assertEqual(args.model_revision, self.settings["model_revision"])
                self.assertFalse(args.allow_model_fallback)
                self.assertEqual((len(train), len(val), len(uci), len(super_sms)), (10, 2, 2, 2))
                self.assertFalse(output.exists())
                return self.result_fixture(plan, pending)
            with patch.object(lora, "check_plan", return_value=checked), patch.object(lora, "resolve", return_value=output), \
                    patch.dict(sys.modules, {"torch": fake_torch}), patch("train_adapted.set_reproducibility") as seeded, \
                    patch("train_lora.train_and_evaluate_lora", side_effect=fake_train) as trainer:
                lora.run_plan(path, "cuda")
                trainer.assert_called_once()
                seeded.assert_called_once_with(1)
            self.assertTrue((output / "run_config.json").exists())
            self.assertEqual(read_json(output / "run_config.json")["status"], "complete")
            self.assertFalse(output.with_suffix(".lock").exists())
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_cli_needs_explicit_mode_and_validated_model_revision(self):
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit):
            lora.main([])
        self.settings["model_revision"] = "main"
        with self.assertRaisesRegex(ValueError, "immutable"):
            lora.validate_settings(self.settings)


if __name__ == "__main__":
    unittest.main()
