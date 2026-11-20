# Drift-triggered LoRA execution guide

The team trainer and helper were imported from commit
`8ac3bedfe207dfbe21fc76ba2afbf4c32e540aec`. Use **`run_step7_lora.py`** for the
benchmark. The original `train_lora.py` / `train_adapted.py` command-line programs
retain their historical trigger/comparison logic and are not the benchmark entry
point. Following CPU validation, all three seeds were executed on RunPod on
2026-11-20. All three downloaded runs passed local artifact validation.

## Completed runs

Seeds 29822, 29823, and 29824 are saved locally under
`results/benchmark_runs/drift_triggered_lora/seed_<seed>/`. The exact prepared
plans match the executed metadata. All six held-out prediction files have full
coverage and continuous spam scores; adapter hashes and validation-threshold
consistency passed. No additional training is scheduled.

Hardware/software: NVIDIA RTX PRO 4500 Blackwell, PyTorch 2.8.0+cu128,
Transformers 4.57.6, Accelerate 1.16.0, Python 3.12.3, bfloat16. Training wall
time averaged 105.54 +/- 2.09 seconds per seed; the two held-out evaluations
together averaged 39.23 +/- 1.09 seconds. Peak allocated CUDA memory averaged
6,522.01 MiB. Each adapter has 24,320,000 trainable parameters and a 46.51 MiB
weight file. Pod setup/download/idle time and billed cost are not measured by
these timers.

The original ZIP, extracted logs/environment, and import audit are retained in
`results/benchmark/imports/runpod_lora_20261120/`. See
[the combined report](results/benchmark/BENCHMARK_REPORT.md) for held-out metrics.
Commands below document the workflow; existing completed outputs must not be
overwritten or rerun inadvertently.

## Frozen protocol

- Model: `unsloth/Llama-3.2-3B`, revision
  `d4446454d87d51aa42e1fb174f25acc5f8762331` (Hugging Face metadata verified).
- Team rank-16 LoRA, alpha 32, dropout 0.05; seven projection modules plus the
  sequence-classification head. Two epochs, batch 4, accumulation 4, maximum
  length 128, bfloat16, gradient checkpointing, class-weighted cross-entropy.
- Trigger: first **MMD** monitoring alert recorded by the benchmark for that
  seed. This selection is fixed across seeds and does not inspect held-out F1
  or use the known drift onset to filter alerts. A false alarm would trigger too.
- Training: all stream rows strictly after the alert, collected offline. The
  alert-producing message is excluded. This is not an online-recovery claim.
- Validation: the team's seeded sample of 200 records from the existing
  validation splits, used for checkpoint and spam-F1 threshold selection.
- Evaluation: every held-out UCI and Super SMS message, kept separate from the
  training stream. No test message is used for sampling, tuning, or training.

For seed **29822**, the recorded MMD alert is **window 10, position 2000**.
The selected training set is **800 messages, positions 2001-2800** (640 ham,
160 spam), with 200 validation messages, 1,031 UCI test messages, and 2,000
Super SMS test messages. The older team's 1,600-row run used a different trigger
and remains historical evidence awaiting verification.

## CPU preparation and checks

From the repository root, with `requirements.txt` installed:

```powershell
python -m unittest discover -s tests -v
python scripts/run_step7_lora.py --prepare --seed 29822
python scripts/run_step7_lora.py --check --plan results/benchmark/lora_preflight/seed_29822/plan.json
```

All three seeds (29822, 29823, 29824) have been prepared and passed CPU checks
as of 2026-11-20. Each seed's recorded MMD alert is at position 2000 and selects
800 post-alert training messages. Use `--check` on the existing plans;
preparation refuses to overwrite them. After intentional code or
protocol changes, prepare into a new directory using `--preparation-dir`.

Preparation validates processed data and complete baseline predictions through
the benchmark's existing checks. It independently recomputes the detectors on
CPU and requires the saved thresholds, summaries, and alerts to match. It then
copies only the necessary baseline stream and detector inputs into its own
preparation directory and freezes their hashes, the exact usage IDs, trainer
source, code hashes, model revision, and hyperparameters in `plan.json`.

`--check` verifies that frozen plan, its input copies, current processed data,
source code, and deterministic dataset selection. Neither CPU mode imports
PyTorch/Transformers, downloads a model, contacts Ollama, or writes baseline
outputs. All three baseline runs are now complete.

The preparation commands below have already completed for the later seeds;
use them only when preparing a new snapshot into an absent directory:

```powershell
python scripts/run_step7_lora.py --prepare --seed 29823
python scripts/run_step7_lora.py --prepare --seed 29824
```

Each seed must have its own complete stream predictions and detector artifacts.
No alert, wrong seed, altered artifacts, an early window timestamp, or no
remaining training rows causes an error. There is no schedule-onset fallback.
Use `--baseline-stream-dir` only when preparing from a different verified
location; `--detector-root` can point to another benchmark report's detector tree.

## GPU execution

On the designated GPU host, copy the current working code (including local,
unstaged additions), configs, requirements files, **the existing `data/processed/`
tree including split files**, and the entire directory:

```text
results/benchmark/lora_preflight/seed_29822/
  plan.json
  baseline_run_config.json
  baseline_predictions.csv
  detector_summary.json
  alerts.csv
```

These processed data and preparation files are Git-ignored and will not arrive
with a normal clone. A D: drive is not needed on the GPU host: the plan uses the
portable snapshot and repository-relative paths. Code hashes normalize CRLF/LF
for Windows-to-Linux transfers; input-file hashes remain byte-exact.

Use a CUDA-enabled PyTorch installation and an available GPU supporting
bfloat16. The entry point refuses CPU fallback. With that environment ready:

```bash
python -m pip install -r requirements-lora.txt
python scripts/run_step7_lora.py --check --plan results/benchmark/lora_preflight/seed_29822/plan.json
python scripts/run_step7_lora.py --run --plan results/benchmark/lora_preflight/seed_29822/plan.json --device cuda
```

The last command downloads/loads the pinned weights and starts training.
Execution is complete for all three seeds. Existing final output directories
prevent accidental reruns. Additional billed experiments require authorization.
If an HF token is required, use the `HF_TOKEN` environment variable; it is not serialized
into the plan or run metadata. GPU package versions are recorded in the output.

The CPU suite validates orchestration, data/provenance, and artifact ingestion
using synthetic trainer outputs. Actual GPU execution has now succeeded for all
three seeds on the recorded RunPod configuration. The local benchmark environment
does not need PyTorch, Transformers, or Accelerate to validate and report the
downloaded predictions. This success does not imply compatibility or memory fit
on every other GPU/software configuration.

## Output artifacts

Successful execution writes:

```text
results/benchmark_runs/drift_triggered_lora/seed_29822/
  run_config.json
  uci_test_predictions.csv
  super_sms_test_predictions.csv
  metrics.json
  training_logs.csv
  plan.json
  lora_adapter/adapter_model.pt
  lora_adapter/adapter_config.json
  lora_adapter/<tokenizer files>
```

CSV rows retain continuous `spam_probability`. The benchmark recomputes spam
F1, accuracy, precision, spam recall, all-actual-ham FPR, and trapezoidal PR-AUC.
Metadata records training/validation IDs, empty replay/retrieval sets, the exact
recorded trigger and artifact hashes, frozen model/code identity, selected
threshold, training and population-specific inference times, CUDA peak allocated
memory, trainable parameters, and adapter weight-file size/hash. The legacy
`_mb` memory/adapter fields use MiB; this is declared explicitly in metadata.
Synchronized wall times are not advertised as measured utilization/GPU-hours
or cost; those fields remain null without a separate measurement.

The final output directory must be absent. Execution uses an exclusive lock and
a sibling temporary directory, validates both complete prediction files, and
publishes the directory only at completion. Failed output stays in a clearly
named `.seed_29822_inprogress_*` directory for diagnosis; it is not ingested by
the benchmark. There is no automatic resume or overwrite of a trained adapter.

Copy the completed LoRA directory back to the local repository, then regenerate
the combined benchmark (all three baseline seeds are already complete):

```powershell
python scripts/run_benchmark.py --require-complete
```

All configured baseline and LoRA held-out runs are now present. Missing or
invalid artifacts would still produce a nonzero exit. Existing historical
summary-only runs are never silently converted into completed benchmark evidence.

## Changes to the pinned trainer

The training algorithm and helper's seeded sampler are retained. Small local
patches make the Matplotlib cache portable, pin and verify the loaded model
revision, disable model fallback for benchmark runs, use the built-in Llama implementation,
retain unrounded continuous prediction probabilities, and return optimizer-step
and validation-calibration timing metadata. The wrapper bypasses the historical
main functions, uses alert+1 with the helper's inclusive training boundary, and
emits benchmark artifacts. Original source hashes are in `configs/step7_lora.json`;
actual executed source hashes are in each prepared plan and final run metadata.

All changes remain local and unstaged. Selective staging, review, commit, and
push commands are documented in `BENCHMARK_README.md`.
