# Benchmark validation

## Completed benchmark validation: 2026-11-20

**PASS: all configured baseline and LoRA runs are complete.**

- `python scripts/run_benchmark.py --require-complete`: **exit 0**. All 15
  requested seed/population combinations are complete: nine baseline and six
  LoRA. LoRA stream evaluation is intentionally not requested for this offline
  post-alert classifier.
- `python -m unittest discover -s tests -v`: **43 tests passed again today**.
- Downloaded archive: `step7-lora-results-checked.zip`, **122,938,634 bytes**,
  SHA-256 `b8d9fce82c6e5c802ff9238f24898ca8567a7149c52b2d0e3f76792b32d25144`.
  ZIP integrity and all 36 imported file hashes passed. The original ZIP, logs,
  environment capture, and detailed import audit are preserved under
  [`imports/runpod_lora_20261120/`](imports/runpod_lora_20261120/).
- The returned plans exactly match the prepared local plans for seeds 29822,
  29823, and 29824. Model revision, executed code identity, dataset hashes,
  recorded MMD alerts, training/validation membership, full held-out coverage,
  score/threshold consistency, adapter hashes/sizes, and recomputed metrics all
  passed. No adapter was deserialized during CPU validation.
- Each LoRA seed used 800 rows strictly after its own recorded alert at position
  2000, 200 validation rows, and separate 1,031-message UCI / 2,000-message Super
  SMS test sets. All six LoRA prediction files retain continuous spam scores.
- The combined report, all tables, detector aggregates, and error-bar plot were
  regenerated. The plot was visually checked. No inference or training was
  launched during this local import/validation step.

| Seed | Baseline Super SMS F1 (%) | LoRA Super SMS F1 (%) | Baseline UCI F1 (%) | LoRA UCI F1 (%) |
|---|---:|---:|---:|---:|
| 29822 | 74.29 | 98.70 | 27.59 | 94.21 |
| 29823 | 78.45 | 98.55 | 31.20 | 92.25 |
| 29824 | 92.84 | 98.38 | 64.53 | 93.50 |
| Mean +/- sample SD, n=3 | **81.86 +/- 9.73** | **98.54 +/- 0.16** | **41.11 +/- 20.37** | **93.32 +/- 0.99** |

SD is in percentage points. Paired Super SMS F1 change is **+16.68 +/- 9.88
points**, and mean per-seed UCI F1 retention is **260.67 +/- 102.87%**. Retention
is the mean of the three paired ratios, not a ratio of the aggregate means;
values above 100% mean improvement over that seed's baseline. The baseline's
wide seed variation must remain visible.

The report contains accuracy, precision, spam recall, FPR (FP / all actual ham),
and trapezoidal continuous-score PR-AUC. LoRA PR-AUC is **99.89 +/- 0.03%** on
Super SMS and **95.83 +/- 0.79%** on UCI. Baseline PR-AUC is N/A because the
existing runs are label-only, not because a baseline run is missing.

LoRA ran on NVIDIA RTX PRO 4500 Blackwell, PyTorch 2.8.0+cu128, Transformers
4.57.6, Accelerate 1.16.0, Python 3.12.3, bfloat16. Mean training wall time was
**105.54 +/- 2.09 seconds per seed**; the two held-out evaluations together took
**39.23 +/- 1.09 seconds per seed**. Peak allocated CUDA memory was
**6,522.01 +/- 1.45 MiB**, with **24,320,000 trainable parameters** and a
**46.51 MiB adapter weight file** per seed. The report's legacy `_mb` fields
represent MiB for these LoRA runs. Timers exclude pod setup, weight downloads,
and idle billing; actual cost and measured accelerator utilization remain N/A.
Baseline inference ran on a different GPU, so timings are not a controlled
hardware speed comparison.

ADWIN, statistical KS/PSI, and embedding MMD are complete for all three baseline
seeds. ADWIN detected no shift; KS, PSI, and MMD first alerted at position 2000
with 999-message delay in each seed. Zero false alerts were observed within the
limited, disjoint 400-message pre-drift monitoring exposure per seed.

**Remaining core runs: none.** The saved stream already contains a stepwise
gradual transition: five UCI-only windows, four mixed windows with Super SMS
shares of 10%, 25%, 50%, and 75%, then five Super-SMS-only windows. Actual source
counts in all 2,800 rows match the manifest and configuration. All completed
baseline stream and detector runs used this schedule. Earlier documentation
describing gradual drift as unrun was incorrect.

The generator uses an explicit window schedule rather than a scalar width
parameter `w`. A separate abrupt-versus-gradual comparison or transition-width
study has not been evaluated. No additional GPU run is needed for the existing
gradual schedule. Ablation studies, failure analysis, and independent
reproduction are outside the current benchmark scope. Historical summary-only results stay
unverified and excluded; current complete runs do not depend on recovering them.
The exact raw manifest edits and read-only D-drive stream were preserved. All
changes remain local and unstaged; no commit, push, or PR was made.

## Earlier baseline-only checkpoint (2026-11-20)

The section below records the state before the user ran LoRA on RunPod. Its
pending-run notes are superseded by the final validation above.

**All three baseline seeds are complete and validated.** Running
`python scripts/run_benchmark.py` succeeded and regenerated the combined report,
tables, detector results, and sample-SD error-bar plot from saved predictions.
For each seed, coverage is 2,800 stream / 1,031 held-out UCI / 2,000 held-out
Super SMS messages. Seed 29822's stream is reused from the complete D-drive
run; the partial Desktop stream remains untouched.

| Baseline seed | Held-out Super SMS spam F1 (%) | Held-out UCI spam F1 (%) |
|---|---:|---:|
| 29822 | 74.29 | 27.59 |
| 29823 | 78.45 | 31.20 |
| 29824 | 92.84 | 64.53 |
| Mean +/- sample SD, n=3 | **81.86 +/- 9.73** | **41.11 +/- 20.37** |

SD is in percentage points. All six metric columns, separate stream diagnostics,
and recorded inference timings are in [`BENCHMARK_REPORT.md`](BENCHMARK_REPORT.md).
Baseline PR-AUC remains N/A because these runs contain labels without continuous
spam scores. The held-out baseline runs record the same model digest and RTX
3050 Laptop GPU. The wide variation across seeds remains visible in the report.

CPU LoRA preflight passed for **29822, 29823, and 29824**. Each seed's own
recorded MMD alert independently resolves to window 10, position 2000, selecting
800 strictly post-alert training rows and 200 validation rows. The team trainer
and helper are integrated from pinned commit
`8ac3bedfe207dfbe21fc76ba2afbf4c32e540aec`; no trainer search remains pending.
The latest test-suite result remains **43 tests passed on 2026-11-19**; this
report-only update did not change executable code or rerun the suite. Actual
processed-data and frozen-plan CPU checks passed today for all three seeds.
See [`lora_preflight/VALIDATION.md`](lora_preflight/VALIDATION.md).

| Method | Seeds | Remaining work |
|---|---|---|
| 15-shot baseline | 29822, 29823, 29824 | None: all stream and held-out outputs validated; detectors evaluated |
| Drift-triggered LoRA | 29822 | CPU plan ready; historical results still lack verified prediction/provenance artifacts; current-protocol GPU run not started |
| Drift-triggered LoRA | 29823, 29824 | CPU plans ready; training and both held-out evaluations not started |

Next GPU command for seed 29822, only after approved GPU environment setup:

```bash
python scripts/run_step7_lora.py --run --plan results/benchmark/lora_preflight/seed_29822/plan.json --device cuda
```

GPU setup and artifact-transfer requirements are in `LORA_README.md`. This
validation launched no model inference, downloads, training, or paid compute.
All changes remain local and unstaged.

## Earlier validation snapshot (2026-11-19)

The following records the earlier state; its pending-baseline and trainer-search
notes are superseded by the current readiness section above.

Validated locally on 2026-11-19 in the benchmark checkout, before new full inference
or training runs. Scope: 15-shot Llama 3.2 3B versus drift-triggered LoRA; spam F1
primary; ADWIN, statistical KS/PSI, and embedding MMD detector families.

- `python -m unittest discover -s tests -v`: **32 tests passed**. Covers invalid
  responses, AUC availability, sample SD, retention, prediction coverage/order,
  leakage and future-data rejection, detector observation time, disjoint
  calibration, no synthetic-error fallback, paired artifact ingestion, offline
  populations, inference resume, default CLI behavior without inference,
  read-only per-population external reuse, no fallback to partial local data,
  explicit missing-seed/error-bar states, and the team's probability column.
  New checks cover all-actual-ham FPR, continuous-score trapezoidal PR-AUC and
  unavailability reasons, complete-model-seed aggregation, paired adaptation
  outcomes, population-scoped resource measurements, historical count arithmetic,
  rejection of invalid historical predictions, and verification-pending states.
- Python compilation passed for the new benchmark modules and modified scripts.
- The preceding bounded real Ollama check completed with **six predictions** (two per
  population). Command:
  `python scripts/run_benchmark.py --smoke --smoke-limit 2 --output-dir results/benchmark/step7_revision --require-complete`.
  Its preserved artifacts are in `step7_revision/smoke/`. External reuse entries are
  cleared for smoke checks. These are integration checks, not accuracy estimates.
  Earlier smoke artifacts in `smoke/` and `verification/smoke/` are preserved.
- Re-running a smoke check with identical provenance reused its completed
  prediction files without additional inference.
- This metric/report revision ran tests and CPU analysis of cached predictions;
  it launched no new model inference, downloads, training, or paid compute.
- `python scripts/run_benchmark.py`: validated processed inputs (2,800 stream
  messages; 1,031 UCI / 2,000 Super SMS held-out examples), audited the complete
  D-drive predictions, recomputed detectors locally, and regenerated the report,
  CSV table, and `benchmark_f1.png`. The plot was visually checked.
- `--require-complete`: confirmed exit code **2** with current missing experiments.
- Whitespace checks passed for tracked files changed by this task. Existing
  unrelated edits in `data/raw/manifest.json` were left untouched.

## Reused seed 29822

The complete stream comes from `D:/DATA298A_Harness/seed_29822`, not the 2,637-row
Desktop copy. Both the prediction file and run configuration passed the
benchmark's own checks. All 2,800 predictions match the current stream's IDs,
labels, ordering, windows, model tag, seed, and fixed prompt example IDs.

- Predictions SHA-256: `5ef9ef9b01a4531f56508a5c55919dc9de62422ac4e6e45e5677a0a560281e03`.
- Recomputed stream spam F1: **0.4037626628 (40.38%)**.
- Recomputed stream accuracy: **0.4096428571 (40.96%)**; six invalid outputs.
- Precision **25.32%**, spam recall **99.64%**, FPR **73.48%** = 1,646 / 2,240
  actual ham. PR-AUC **N/A: label-only run has no continuous spam scores**.
- Recorded inference wall time: **6,887.54 seconds**. Peak memory/cost were not
  recorded and remain N/A. These stream values are not held-out model scores.
- ADWIN: no detected shift. KS, PSI, MMD: first alert at message **2000**, delay
  **999 messages / 4 windows**. All four variants had zero false alerts over the
  disjoint **400-message / 2-window** pre-drift monitoring exposure.
- One complete seed means **SD is unavailable**; no fabricated zero error bars.

External inputs and the partial Desktop copy were not modified. D-drive
prediction SHA-256 was unchanged after report generation. The unrelated raw
manifest remained SHA-256
`55db00bb341d89a46ff9917258525a23976915f419ba6d4ff624b9d7d7665d00`.

## Pending runs and trainer

| Model | Seed(s) | Still needed |
|---|---|---|
| 15-shot baseline | 29822 | Verify existing held-out UCI/Super SMS predictions and protocol before deciding on repeats |
| 15-shot baseline | 29823, 29824 | Stream, both held-out sets, and per-seed detectors |
| Drift-triggered LoRA | 29822 | Verify existing held-out results, original trainer source, trigger and training provenance |
| Drift-triggered LoRA | 29823, 29824 | Trainer integration, training, both held-out sets, audited provenance |

`git fetch origin` succeeded. The trainer was located and inspected at
`origin/main` commit `8ac3bedfe207dfbe21fc76ba2afbf4c32e540aec`:
`scripts/train_lora.py` imports `scripts/train_adapted.py`. Both files parse
successfully; neither was imported into the working tree nor executed. See
`BENCHMARK_README.md` for the provenance and detector-trigger integration gaps.
Historical team baseline/LoRA metrics lack full predictions in the inspected
repository and D-drive locations. The benchmark now audits pinned summaries
itself. All four held-out arithmetic checks pass, but message membership,
continuous scores, and execution provenance remain unverified. Historical
baseline prompt IDs and LoRA trigger calibration/timing also differ from the
current protocol. These results are marked `needs_verification`, excluded from
audited aggregates, and are not automatically scheduled for repetition.

| Historical seed-29822 summary (unverified) | UCI spam F1 | Super SMS spam F1 |
|---|---:|---:|
| Baseline | 49.21% | 87.48% |
| Drift-triggered LoRA | 93.98% | 98.63% |

Summary arithmetic implies +11.15 pp Super SMS F1 and 190.98% UCI F1 retention;
these are not accepted benchmark improvements. Reported LoRA resources are
725.37 seconds training, 46.39/184.09 seconds UCI/Super SMS inference,
7,057.69 MB peak memory, 24,320,000 trainable parameters, and a 46.51 MB adapter.
The report labels all these historical measurements as unverified.

The corrected table columns are spam F1 (primary), accuracy, precision, spam
recall, FPR, and PR-AUC. Each completed population is shown per seed. Held-out
means require both test sets for that model seed; paired changes require a
complete baseline seed too. Sample SD needs at least two eligible seeds.
No LoRA scheduling or paid compute has occurred.

Next command (readiness only):

```bash
python scripts/run_benchmark.py --require-complete
```

Expected exit code is 2 until missing/verification-pending artifacts are resolved.
Recover the existing held-out files and align protocols before renting a GPU.
Git handoff commands are in `BENCHMARK_README.md`; generated reports are ignored
by Git unless explicitly added.

All task changes remain local and unstaged. No commits, pushes, or PRs were made.

**No new full inference runs or adaptation training were launched.** This
revision used CPU analysis of cached predictions and the 32-test suite. The
preceding bounded Ollama smoke results are retained separately.
