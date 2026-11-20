# Drift-triggered LoRA integration validation

## Completed GPU execution and local import: 2026-11-20

The user executed all three frozen plans on RunPod's NVIDIA RTX PRO 4500
Blackwell. All three final folders and six full held-out prediction CSVs were
downloaded and locally validated. The executed plans exactly match their
prepared counterparts; input/code/model identity, usage IDs, continuous scores,
decision thresholds, adapter hashes, and saved metrics passed validation.
The full benchmark `--require-complete` command exited **0**, and **all 43 tests
passed again** after import. No core LoRA run remains pending.

The checked archive has SHA-256
`b8d9fce82c6e5c802ff9238f24898ca8567a7149c52b2d0e3f76792b32d25144`.
The archive, extracted logs, full GPU environment, and import audit are saved in
`results/benchmark/imports/runpod_lora_20261120/`. Actual package versions are
also embedded in each final `run_config.json`. See
[`../VALIDATION.md`](../VALIDATION.md) for final metrics and resource measurements.

## Earlier CPU preparation record

The remaining sections document the pre-execution checks. Their statements
about GPU execution being pending or untested describe that earlier state.

Validated locally on 2026-11-20 in the benchmark checkout after all three
baseline seeds completed. No LoRA training, model download,
package installation, GPU rental, or paid compute was performed.

## CPU results

- `python -m unittest discover -s tests -v`: **43 tests passed on 2026-11-19**, preserving the
  existing 32 benchmark tests and adding 11 LoRA integration tests.
  No executable code changed in the 2026-11-20 report/preparation update, so
  the suite was not repeated; actual-data CPU checks below were run today.
- `python scripts/run_step7_lora.py --prepare --seed 29822`: **PASS** on the
  actual processed data and complete D-drive stream predictions. Saved detector
  thresholds, summaries, and alerts match CPU recomputation for seed 29822.
- `python scripts/run_step7_lora.py --check --plan results/benchmark/lora_preflight/seed_29822/plan.json`:
  **PASS again on 2026-11-20**, using the frozen portable inputs and current code/data hashes.
- `python scripts/run_step7_lora.py --prepare --seed 29823` and `--seed 29824`:
  **PASS on 2026-11-20**, including independent detector recomputation and
  frozen-plan checks for each seed. Seed 29823 also passed a separate `--check`.
- Python compilation passed for the imported trainer/helper, benchmark entry point,
  and its tests.
- Tests cover no-onset-fallback behavior, wrong-seed/hash rejection, observed
  window-end timing, false-alarm trigger selection without oracle filtering,
  strictly post-alert training, deterministic validation selection, tamper
  rejection, continuous scores and threshold consistency, complete held-out
  coverage, publication only after completion, and refusal to overwrite output.
- Synthetic trainer outputs passed the **actual benchmark collector** for both
  held-out populations, with valid PR-AUC and population-specific timing metadata.
  Synthetic artifacts lived only in test temporary directories.

## Prepared experiments

| Field | Seed 29822 |
|---|---|
| Team source | `8ac3bedfe207dfbe21fc76ba2afbf4c32e540aec` |
| Model | `unsloth/Llama-3.2-3B` |
| Model revision | `d4446454d87d51aa42e1fb174f25acc5f8762331` |
| Trigger | First recorded MMD monitoring alert; window 10, position 2000 |
| Training | Positions 2001-2800: 800 records, 640 ham / 160 spam |
| Validation | 200 seeded records from existing validation splits |
| Held-out tests | 1,031 UCI and 2,000 Super SMS |
| Evaluation mode | Offline post-alert buffer |
| Output | `results/benchmark_runs/drift_triggered_lora/seed_29822` |
| Prepared plan | `seed_29822/plan.json` |

| Seed | Recorded MMD trigger | Training rows | Validation rows | CPU status |
|---|---|---:|---:|---|
| 29822 | Window 10, position 2000 | 800 | 200 | PASS |
| 29823 | Window 10, position 2000 | 800 | 200 | PASS |
| 29824 | Window 10, position 2000 | 800 | 200 | PASS |

Each plan is in `seed_<seed>/plan.json` and uses that seed's own validated
baseline/detector snapshot. All three alerts happen to select positions
2001-2800, with 640 ham and 160 spam training rows. Each plan keeps both complete
held-out sets separate: 1,031 UCI and 2,000 Super SMS messages.

Plan contents include original and executed code hashes, processed-data hashes,
baseline/detector snapshot hashes, exact training/validation IDs, fixed model
revision, and frozen hyperparameters. All three plans are prepared, not executed.

The existing historical seed-29822 LoRA summaries remain unverified. This plan
uses the benchmark's corrected detector protocol and does not adopt the older
trigger position or insert historical numbers into audited results.

## GPU handoff

After approval and GPU environment setup/transfer described in `LORA_README.md`:

```bash
python scripts/run_step7_lora.py --run --plan results/benchmark/lora_preflight/seed_29822/plan.json --device cuda
```

The local environment lacks PyTorch, Transformers, and Accelerate. CPU checks
verify orchestration and artifact correctness; actual model training, VRAM fit,
and GPU/library compatibility remain untested. The command requires CUDA with
bfloat16 support and an absent final output directory. It has **not** been run.

## Preserved work

Baseline scripts and external stream predictions were unchanged. Their hashes:

- `scripts/run_harness.py`: `463d84a757dd68ce7aaa8e49ddc71a04b53e0fef399d0930c1a8d6744362fa62`
- `scripts/evaluate_baseline.py`: `2cafc712c9f605e964894b242600242f5c9641940bcd5d2898a118df11da23eb`
- D-drive predictions: `5ef9ef9b01a4531f56508a5c55919dc9de62422ac4e6e45e5677a0a560281e03`
- Unrelated `data/raw/manifest.json`: `55db00bb341d89a46ff9917258525a23976915f419ba6d4ff624b9d7d7665d00`

No baseline output was overwritten or baseline process stopped. The main
benchmark report was regenerated on 2026-11-20 after all baseline seeds finished.
It confirms 3/3 completed baseline model seeds and 0/3 verified LoRA model seeds.
Regenerate it again after LoRA artifacts are available. All task edits are local and
unstaged. No commits, pushes, or PRs were made.
