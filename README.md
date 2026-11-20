# AI-Model-Concept-Drift

## Concept drift benchmark

See [BENCHMARK_README.md](BENCHMARK_README.md) for the benchmark protocol,
adapted-model artifact format, and validation commands. Generate a readiness
report from existing artifacts without running model inference:

```bash
python scripts/run_benchmark.py
```

The default configuration compares the 15-shot baseline with drift-triggered
LoRA across seeds 29822, 29823, and 29824, with spam F1 primary. It separates
held-out tests from stream diagnostics and marks missing, incomplete, or
verification-pending runs. Columns include spam F1, accuracy, precision, spam
recall, all-actual-ham FPR, and continuous-score PR-AUC (N/A for label-only runs).
Seed 29822's complete D-drive stream is reused read-only; held-out results are
resolved separately. ADWIN, statistical KS/PSI, and embedding MMD are the three
detector families. Reports include mean/sample-SD tables and an F1 comparison plot.
Full inference requires the explicit `--run-baseline` flag.

The team's pinned LoRA trainer is integrated through `scripts/run_step7_lora.py`.
See [LORA_README.md](LORA_README.md) for CPU preparation, recorded per-seed MMD
triggers, portable plans, and the GPU command. Training requires explicit `--run`.

As of 2026-11-20, all three baseline and LoRA seeds are complete and locally
validated. LoRA runs were executed on RunPod; the resulting predictions and
adapters match the frozen plans. See the generated [benchmark report](results/benchmark/BENCHMARK_REPORT.md)
and [validation record](results/benchmark/VALIDATION.md). The evaluated stream
already contains a stepwise gradual transition: Super SMS represents 10%, 25%,
50%, and 75% of windows 6-9, followed by 100% in windows 10-14. An explicit
transition-width parameter or comparison across different widths has not been
evaluated. No additional run is required for the existing gradual schedule.
