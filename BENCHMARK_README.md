# Concept drift benchmark

This benchmark compares **15-shot Llama 3.2 3B with drift-triggered LoRA**
for SMS concept drift, with **spam F1 as the primary metric**.
It runs baseline inference only when explicitly requested, validates
prediction artifacts, evaluates three detector families, and writes the benchmark
table with across-seed sample-standard-deviation error bars. Report generation
uses saved artifacts and does not initiate training or provision compute.

The benchmark scope consists of the 15-shot baseline, drift-triggered LoRA,
three detector families, and evaluation across three seeds. Supporting
references include the
[evaluation plan](https://docs.google.com/document/d/1cuZ1VwqH31ZZAkmN0CePFqakVAKyibip-XNMjrRlq0g/edit)
and [seed selection plan](https://docs.google.com/presentation/d/1Y2cGna9RiFkTJmwJWNVVYYdK_s03HMajHeOcDWWm0Cc/edit).

## Completed core benchmark: 2026-11-20

All three seeds (29822, 29823, 29824) have complete, validated baseline and
drift-triggered LoRA held-out results. LoRA runs used an NVIDIA RTX PRO
4500 Blackwell; the downloaded outputs were checked locally against the frozen
plans, processed data, recorded detector alerts, continuous scores, validation
thresholds, and adapter hashes. All 43 tests pass. No core model runs remain
missing, and no further GPU run is scheduled.

| Model | Held-out Super SMS spam F1 (%) | Held-out UCI spam F1 (%) |
|---|---:|---:|
| 15-shot Llama 3.2 3B | 81.86 +/- 9.73 | 41.11 +/- 20.37 |
| Drift-triggered LoRA | 98.54 +/- 0.16 | 93.32 +/- 0.99 |

Values are mean +/- sample SD across three completed model seeds; SD is in
percentage points. See [the full report](results/benchmark/BENCHMARK_REPORT.md)
and [validation evidence](results/benchmark/VALIDATION.md) for all six metric
columns, per-seed results, paired changes/retention, resources, and detector
comparisons. These compare two systems: the baseline uses 15-shot Ollama
inference on the laptop, while LoRA uses a sequence-classification head and
bfloat16 on RunPod. Hardware-dependent timings are reported separately.

ADWIN, KS/PSI, and MMD evaluations are complete for the three baseline streams.
The evaluated stream already contains a stepwise gradual transition: windows
1-5 contain only UCI; windows 6-9 contain 10%, 25%, 50%, and 75% Super SMS;
windows 10-14 contain only Super SMS. Each window has 200 messages, so the four
mixed-source windows contain 800 messages. This schedule is recorded in
`configs/drift_stream.yaml` and the validated `stream_manifest.json`; the saved
stream's actual source counts match it. All completed stream predictions and
detector evaluations use this schedule. Earlier notes describing gradual drift
as unrun were incorrect.

The implementation specifies fractions per window rather than an explicit
transition-width parameter `w`. An abrupt-versus-gradual comparison or a study
of different transition widths would require a separate protocol and artifacts;
neither is included in the current results. No additional GPU run is required
to report the gradual schedule already evaluated.
Ablation studies, failure analysis, and independent reproduction are outside
this benchmark's scope. Historical summary-only results remain separate
and do not block completion of the current benchmark.

## Validate before full experiments

Use Python 3.10+ with the repository's `requirements.txt`. The benchmark itself
uses NumPy, SciPy, scikit-learn, and Matplotlib; its configuration is JSON.

```bash
python -m unittest discover -s tests -v
python scripts/run_benchmark.py
```

The report command performs no inference or training. It audits local data and
existing artifacts. Complete stream artifacts also receive CPU detector
evaluation. Outputs go to `results/benchmark/`:

- `BENCHMARK_REPORT.md`: readiness, coverage, classification results, limitations.
- `benchmark.json`: configuration, source/code/artifact hashes, environment,
  per-run metrics, confusion matrices, provenance, aggregate results.
- `availability.csv`: missing, needs-verification, incomplete, invalid, complete, or not-requested
  method/seed/population combinations.
- `summary.csv`: mean and sample standard deviation across seeds, with counts.
- `benchmark_table.csv` and `benchmark_f1.png`: baseline-versus-LoRA comparison,
  with spam F1 first, missing cells explicit, and error bars only when n >= 2.
- `paired_comparisons.csv` and `paired_summary.csv`: paired held-out changes and retention.
- `per_seed_metrics.csv`: each validated population/seed with all six metric
  columns and PR-AUC/FPR unavailability reasons.
- `adaptation.csv` and `adaptation_summary.csv`: Super SMS F1, change from
  baseline, and UCI F1 retention, with completed-model and paired-seed counts.
- `historical_audit.json`: pinned team summary arithmetic, artifact checks,
  source hashes, reported resources, and unresolved protocol/provenance gaps.
- `window_metrics.csv` and `window_performance.png`: training-stream diagnostics.
- `detector_metrics.csv`, `detector_summary.csv`, `detectors/`: per-seed detector
  results, aggregate results, alert logs, thresholds, and window signals.
- `compute.csv`: separately declared training/evaluation and accelerator measurements.
- `resources.csv` and `resource_summary.csv`: per-population inference time,
  shared run training time, peak memory/cost, trainable parameters, and adapter
  size; mean/sample SD grouped by method, population, hardware, and currency.

For a bounded integration check, start Ollama with `llama3.2:3b` already installed:

```bash
python scripts/run_benchmark.py --smoke --smoke-limit 2 --output-dir results/benchmark/step7_revision --require-complete
```

This makes at most six predictions: two stream messages and two from each test
set, using one seed. This command writes beneath `results/benchmark/step7_revision/smoke/`, skips detectors,
and labels all outputs as smoke validation rather than experiment evidence.
Smoke mode clears external reuse entries and cannot consume full-run evidence.
Re-running with the same settings resumes validated prediction prefixes. Changing
the limit requires a different `--output-dir`. Unit tests use synthetic fixtures
in temporary directories and never contact Ollama.

`--require-complete` exits with code 2 when a requested artifact is missing,
incomplete, or invalid; default report mode exits 0 after writing a readiness
report even when experiments are pending. Invalid benchmark inputs exit 1.

## Protocol

The paired seeds are **29822, 29823, 29824**. All methods use the same frozen stream
and held-out examples. Baseline seeds control the 15-shot sample/order and the
Ollama seed; they do not regenerate the stream. Training owners must record the
operations their seeds control in run metadata. One seed has no sample SD;
variation across windows is not reported as across-run uncertainty.
Held-out means require both held-out populations to be complete for that model
seed. Paired changes/retention additionally require the baseline model seed to
be complete. Individually validated populations remain visible in per-seed
outputs even if their companion population is pending. Stream aggregates are
independent and never used as substitutes for held-out scores.
Detector seeds control bootstrap resampling and SVD; the stream remains fixed.
Their error bars describe variability from those operations, not independent
draws of new SMS data.

There are three distinct evaluation populations:

| Population | Source | Purpose |
|---|---|---|
| `stream` | 2,800 training records in 14 windows | Predict-before-update diagnostics and detector evaluation |
| `uci_test` | 1,031 UCI test records | Original-domain accuracy and spam-F1 retention |
| `super_sms_test` | 2,000 Super SMS test records | Post-shift generalization |

The default config requests all three populations for the baseline, and both
held-out populations for drift-triggered LoRA. The team's trainer consumes a
collected post-trigger buffer and evaluates held-out tests. This offline classifier does not
provide evidence of online recovery. To evaluate a genuine online implementation,
add `stream` to that method's `populations` and provide its prequential provenance.
All held-out threshold tuning must use validation data only.

### Reuse the complete seed-29822 stream

`configs/benchmark.json` contains a population-specific, read-only override:

```json
"reuse_runs": {"29822": {"stream": "D:/DATA298A_Harness/seed_29822"}}
```

The benchmark reads both `predictions.csv` and `run_config.json` from that folder.
It validates the model tag, seed, fixed 15-shot example IDs, stream hash, all
2,800 IDs/labels/positions/window metadata, and prediction coverage. It recomputes
metrics and all detector results under the current calibration protocol. It does
not trust the historical `metrics.json` or copy the historical detector summaries.
The reused file's SHA-256 is
`5ef9ef9b01a4531f56508a5c55919dc9de62422ac4e6e45e5677a0a560281e03`.

The Desktop `results/harness/seed_29822` copy has 2,637 rows and is preserved but
not used. Held-out UCI/Super SMS files are resolved separately from
`results/benchmark_runs/baseline/seed_29822`. No population is silently substituted
for another. Invalid or unavailable reuse inputs fail validation rather than
falling back to the partial Desktop copy. Seed 29822's held-out populations
have now completed under the current benchmark protocol. A future
`--run-baseline --seeds 29822` validates and skips reused stream inference and
resumes only compatible local held-out outputs; no repeat is needed now.
External directories are never inference targets.
Historical metadata lacks a model digest; a matching model tag cannot establish
byte-identical historical weights. That provenance limitation remains visible.

Input checks verify the stream's SHA-256, row count, contiguous positions/windows,
schedule, source composition, labels/text, train membership, split-file hashes,
and source ID/text uniqueness. Predictions must cover the exact population, with
unique IDs and matching labels. Stream predictions must also match order and
window metadata. Incomplete runs are excluded rather than scored on a smaller
population. The historical 2,788-message window baseline is a different population
and cannot be silently combined with the 2,800-message harness.

### Scoring policy

Spam is positive. Accuracy counts every invalid response as incorrect. Spam
abstentions contribute to false negatives, so recall and F1 penalize them. Ham
abstentions count against accuracy and appear in the explicit invalid column;
they are not incorrectly counted as spam false positives. Zero-denominator
precision/recall/F1 are 0. The 2x3 confusion matrix has actual rows `[ham, spam]`
and predicted columns `[ham, spam, invalid]`.

The headline columns, in order, are **spam F1, accuracy, precision, spam recall,
false-positive rate (FPR), and PR-AUC**. FPR is `FP / all actual ham`, including
ham abstentions in the denominator. It is N/A if there are no actual ham.
PR-AUC is the trapezoidal area from scikit-learn's precision-recall curve using
continuous spam probabilities; it is distinct from average precision (AP).
Every message must supply a score, and at least one actual spam must be present.
Label-only runs report **N/A: Label-only run: no continuous spam scores**;
partially scored runs have a different explicit reason. No confidence scores
are invented from predicted labels. Additional ROC-AUC requires both classes.
NaN, infinite, or out-of-range scores fail validation. Numeric precision is
retained in JSON/CSV; Markdown tables display percentages, including PR-AUC x 100.

Retention = `100 * adapted_metric / baseline_metric`, on the same held-out
population and seed. Ratios may exceed 100%; a zero baseline gives null.
Absolute changes are percentage points. Both accuracy and spam-F1 retention are
reported. UCI retention measures forgetting; Super SMS paired changes measure
post-shift benefit.

### Detector evaluation

The three families are **ADWIN**, **statistical KS/PSI**, and **embedding MMD**.
KS and PSI are two variants within the statistical family, not two additional
families. They retain separate rows rather than averaging incompatible scores.
`benchmark_detectors.py` reuses the existing ADWIN-style detector, structural
KS/PSI features, and TF-IDF/SVD MMD implementation. It uses a new evaluation
protocol; historical detector outputs are not reused as comparable evidence:

1. Windows 1-2 fit the reference features and embeddings.
2. Window 3 calibrates KS/PSI/MMD thresholds using 100 seeded bootstrap samples
   and their 99th percentiles. The ADWIN parameter is fixed at delta=0.01.
3. Windows 4-5 provide disjoint no-drift monitoring exposure (400 messages).
4. Windows 6-14 measure detection after the known simulated onset.

ADWIN observes real aligned prediction errors and warms up during calibration.
Calibration-period alarms are excluded. Statistical alerts become available at
the **last message** of their window. For example, an alert at the end of window
6 has delay `1200 - 1001 = 199` messages, not zero. Delay in windows is the alert
window minus the onset window; misses have `detected=false` and null delays.
Mean delay is conditional on detection, with its own `n`; detection rate includes
misses so a detector that misses shifts cannot appear fast by averaging in zeros.

False alarms are reported as both event count per 1,000 eligible pre-drift
messages and percentage of eligible pre-drift windows with at least one alert.
The two windows provide limited evidence. The ADWIN implementation is approximate,
and MMD features are TF-IDF/SVD vectors, not pretrained semantic embeddings.
These limitations remain visible in outputs. The older `detect_drift.py` retains
its legacy calibration protocol but now refuses missing/partial prediction files;
it can no longer invent an error sequence from aggregate accuracy.

## Adapted-model artifact contract

Configure run locations in `configs/benchmark.json`. For each requested seed,
the adaptation owner writes:

```text
results/benchmark_runs/<method>/seed_<seed>/
  run_config.json
  uci_test_predictions.csv
  super_sms_test_predictions.csv
  predictions.csv                 # only for a requested online stream evaluation
```

Held-out CSVs require `message_id,label,prediction`, with lowercase labels `ham`
or `spam`, and predictions `ham`, `spam`, or `invalid`. Extra columns are allowed.
Optional columns are `raw_response`, `spam_score` (probability in [0,1]),
`latency_seconds`, and `ollama_eval_seconds` (only for actual Ollama generation
time). A timing sum is null if any row lacks that timing measurement.
The team's `spam_probability` column is also accepted for PR-AUC and ROC-AUC; if both
probability columns are present they must agree. Both are checked for finite
values within [0,1].

Stream CSVs additionally require `stream_position,window,phase,post_drift_fraction,dataset`
matching the frozen stream. They must be emitted in stream order.

An adapted run configuration has this structure (replace placeholder values):

```json
{
  "seed": 29822,
  "model": "model identifier and checkpoint/revision",
  "stream_sha256": "SHA-256 from stream_manifest.json",
  "source_sha256": {
    "uci_sms_spam": "SHA-256 of uci_messages.csv",
    "super_sms": "SHA-256 of super_messages.csv"
  },
  "evaluation_mode": "offline",
  "training_ids": [],
  "replay_ids": [],
  "retrieval_ids": [],
  "validation_ids": [],
  "seed_controls": ["initialization", "training order"],
  "git_commit": "source commit",
  "compute": {
    "hardware": "accelerator model or CPU",
    "precision": "bfloat16",
    "update_count": 1,
    "training_wall_seconds": null,
    "evaluation_wall_seconds": null,
    "peak_memory_mb": null,
    "accelerator_hours_measured": null,
    "trainable_parameters": null,
    "adapter_size_mb": null,
    "compute_cost": null,
    "cost_currency": null,
    "inference_seconds_by_population": {},
    "measurement_method": null
  }
}
```

Declare the **actual complete ID sets**, including the empty sets. Training,
replay, and retrieval IDs must belong to training; validation IDs must belong to
validation. Test IDs are forbidden in all of them. Hash values are also available
in `results/benchmark/benchmark.json`. Metadata declares what the producer did;
the benchmark can verify membership and timing declarations but cannot prove an
external trainer obeyed its own logging. Keep checkpoint/config evidence.

For online evaluation, set `evaluation_mode` to `prequential`. Top-level training,
replay, and retrieval IDs then describe initialization and must exclude the
entire stream. Provide ordered `usage_events`:

```json
{
  "usage_events": [
    {
      "after_position": 1200,
      "training_ids": ["an ID observed at or before position 1200"],
      "replay_ids": [],
      "retrieval_ids": []
    }
  ]
}
```

An event after position N can affect prediction N+1, never prediction N. Retrieval
for query N must therefore be logged after position N-1. Future stream records
and test records are rejected. Initial eligible non-stream training data is
available from position zero. Log all retrieval/update usage, not only drift
alerts. Offline configurations cannot be scored as online stream runs.

Training wall time, evaluation wall time, model-generation time, and measured
accelerator-hours are separate measurements. Missing data stays null. A non-null
accelerator-hour value requires hardware and measurement-method provenance;
wall time is never automatically converted into GPU-hours. MPS and CUDA are not
assumed equivalent. Record adaptation-event detail in additional metadata when
comparing per-event budgets.
Record an integer trainable-parameter count and measured adapter size in MB.
Cost requires a currency. If prediction rows lack latency, record measured
whole-population inference seconds under `inference_seconds_by_population`
(keys `uci_test`, `super_sms_test`, or `stream`). Training, peak memory, adapter
size, and cost are shared run-level measurements: resource rows repeat them
for each population and must not be added together. Baseline training time and
trainable parameters are zero; an absent baseline adapter is N/A.

## Regenerate the completed benchmark

Recompute the report from validated local artifacts without additional model
inference or training:

```bash
python scripts/run_benchmark.py --require-complete
```

The completed dataset contains 2,800 stream messages and 3,031 held-out
predictions per baseline seed, plus 3,031 held-out predictions per LoRA seed.
Exit 0 means all configured populations passed; missing or verification-pending
artifacts return 2. The historical summary-only audit is separate from these
new, complete current-protocol runs. No baseline or LoRA rerun is needed.

New inference writes under `results/benchmark_runs/baseline`. It resumes only
if model digest, input hashes, inference code hashes, prompt examples, seed,
and options match. CSV progress is flushed after every prediction. The
`--baseline-runs-dir` option changes the local destination, not read-only reuse
entries; change the config explicitly to select different external evidence.

`--seeds 29822` selects one seed explicitly. Paths in the benchmark configuration
are relative to the repository root; absolute paths are also supported. No raw
SMS text is copied into aggregate reports. Generated benchmark artifacts are
ignored by Git; publish selected summaries deliberately after review.

## Integrated team LoRA trainer and completed GPU runs

`git fetch origin` succeeded on 2026-11-18. The trainer is in
[`origin/main` at 8ac3bedfe207dfbe21fc76ba2afbf4c32e540aec](https://github.com/rajain08/AI-Model-Concept-Drift/blob/8ac3bedfe207dfbe21fc76ba2afbf4c32e540aec/scripts/train_lora.py),
and was introduced by commit `cefefc1a375c7ddbe6a88118c782b0e912b4d48b`
on `origin/feature/lora-fine-tuning`. It imports
`scripts/train_adapted.py` and the detector primitives. Both trainer/helper files
are now imported locally from that pinned revision, without merging the branch
or replacing baseline/detector code. The benchmark entry point is
`scripts/run_step7_lora.py`; see [LORA_README.md](LORA_README.md) for preparation,
CPU checks, the exact GPU command, and transfer instructions.

The trainer implements rank-16 LoRA plus a sequence-classification head, supports
`--seed`, `--predictions-path`, `--alerts-path`, `--trigger-window`, and
`--trigger-position`, and emits `uci_test_predictions.csv` and
`super_sms_test_predictions.csv` with `spam_probability`. It uses PyTorch,
Transformers, and Accelerate, beyond the baseline/detector dependencies.

The integration emits the benchmark `run_config.json`, complete held-out
prediction CSVs with continuous scores, training/validation IDs, input/model/code
hashes, validation-selected threshold, and measured time/memory/adapter metadata.
It publishes a run only after both prediction files pass the benchmark checks.
The original trainer CLIs are preserved as team code; use the **benchmark entry
point** for these runs, since it bypasses their legacy trigger and comparison
logic, including the schedule-onset fallback.

`configs/step7_lora.json` fixes MMD as the adaptation trigger for every seed:
the first recorded monitoring alert, including any false alarm, without filtering
against the known simulated onset. Preparation independently reproduces the
saved detector artifacts from the correct seed's complete baseline stream.
Missing alerts fail preparation; another seed's trigger is never substituted.
Training starts strictly after the alert and uses the remaining stream as an
offline collected buffer. For seed 29822, this is positions 2001-2800 (800 rows)
after the MMD alert at position 2000. The three detector families remain in
the detector benchmark; this does not add three separate LoRA methods.

Only explicit `--run` loads/downloads model weights or trains. CPU preparation
and checks do not import PyTorch/Transformers. All three prepared runs were
executed on RunPod, and the outputs were imported locally on 2026-11-20.
Local checks confirmed the frozen plans and complete scored held-out results.
Additional billed experiments require authorization; none are scheduled.

## Existing seed-29822 held-out evidence

The configured `historical_results` entries pin the team's baseline and LoRA
summaries to the fetched commit. Every report checks those summaries' confusion
counts against the current held-out class totals and recomputes F1, accuracy,
precision, recall, and FPR. It looks for each declared prediction CSV locally
and in the pinned Git tree, and applies the benchmark's message-level checks if
found. Summaries alone never enter audited aggregates. A missing prediction
artifact changes otherwise-missing coverage to `needs_verification`, not a
scheduled repeat. A complete current-protocol run retains its complete status.

All four summary arithmetic checks pass. Reported historical spam F1 is:

| Model, seed 29822 | UCI | Super SMS |
|---|---:|---:|
| 15-shot baseline | 49.21% | 87.48% |
| Drift-triggered LoRA | 93.98% | 98.63% |

These imply +11.15 pp Super SMS F1 and 190.98% UCI F1 retention, **pending
artifact/protocol verification**. They are not accepted benchmark improvements.
LoRA reports 725.37 s training, 7,057.69 MB peak memory, 24,320,000 trainable
parameters, and a 46.51 MB adapter; these measurements remain unverified.

Full held-out prediction CSVs were absent from the inspected repository and
D-drive harness locations. The historical baseline used different 15-shot
example IDs; the historical LoRA trigger uses the legacy calibration and window
timing. Its recorded run commit lacks `train_lora.py`, and the summary lacks
exact training/validation IDs and source hashes. Those older summaries remain
unverified and excluded. The current benchmark uses the validated three-seed
predictions generated under the frozen protocol. Recovering old files is not a
remaining benchmark run. Any later recovery still requires the normal
run-config and prediction checks before it can count as benchmark evidence.

## Version control workflow

Changes are local and unstaged on `feature/concept-drift-benchmark`. The commands below
stage the implementation and selected benchmark summaries, excluding the
unrelated raw manifest edit, harness outputs, processed data, prediction files,
adapters, and ZIP archives. Review the staged diff before committing.

```powershell
git switch feature/concept-drift-benchmark
git status --short
git add .gitignore README.md BENCHMARK_README.md LORA_README.md requirements-lora.txt configs/benchmark.json configs/step7_lora.json scripts/benchmark_common.py scripts/benchmark_detectors.py scripts/benchmark_history.py scripts/benchmark_reporting.py scripts/run_benchmark.py scripts/run_harness.py scripts/detect_drift.py scripts/run_step7_lora.py scripts/train_lora.py scripts/train_adapted.py tests/test_benchmark.py tests/test_step7_lora.py
git add -f results/benchmark/BENCHMARK_REPORT.md results/benchmark/VALIDATION.md results/benchmark/benchmark_f1.png results/benchmark/benchmark_table.csv results/benchmark/per_seed_metrics.csv results/benchmark/adaptation.csv results/benchmark/adaptation_summary.csv results/benchmark/detector_metrics.csv results/benchmark/detector_summary.csv results/benchmark/resources.csv results/benchmark/resource_summary.csv results/benchmark/lora_preflight/VALIDATION.md
git -c core.whitespace=trailing-space,space-before-tab,cr-at-eol diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "Add concept drift benchmark with baseline and LoRA results"
git push -u origin feature/concept-drift-benchmark
```

Generated reports are ignored by Git, so the explicit summary-file list uses
`git add -f`. The selected reports contain no raw SMS text. Retain processed
data, predictions, adapters, and archives in local artifact storage. These
commands publish the feature branch only; they do not open or merge a pull
request. Staging, commits, and publication remain manual operations.
