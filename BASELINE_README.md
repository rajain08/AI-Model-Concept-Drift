
# Llama 3.2 3B Baseline Evaluation

## Overview

This experiment evaluates the unadapted Llama 3.2 3B
model on SMS spam classification using two datasets:

- UCI SMS Spam Collection (pre-drift proxy)
- Super SMS (post-drift proxy)

The experiment uses a fixed 15-shot prompt with
examples selected from the UCI training split.

No fine-tuning, RAG, or other adaptation is applied.

## Environment

- Python 3.10
- Ollama 0.40.6
- Model: llama3.2:3b
- Few-shot sampling seed: 29822
- Temperature: 0
- Maximum generated tokens: 128
- Local execution on macOS

## Dataset preparation

From the repository root:

python scripts/download_data.py

python scripts/clean_and_split.py

The current pipeline uses a 60/20/20 split.

## Install model

ollama pull llama3.2:3b

## Smoke test

python scripts/evaluate_baseline.py \
  --limit 10 \
  --output-dir results/baseline_smoke

## Full evaluation

python scripts/evaluate_baseline.py \
  --output-dir results/baseline_full

## Generate confusion matrices

python scripts/plot_baseline.py

## Results

UCI test set (1,031 messages):
- Accuracy: 75.46%
- Spam F1: 49.70%
- Invalid model responses: 2

Super SMS test set (2,000 messages):
- Accuracy: 86.25%
- Spam F1: 87.62%
- Invalid model responses: 0

## Limitations

This is a paper-inspired local baseline, not an
exact replication of Hur et al. (2026).

Differences in dataset preprocessing, sample
composition, model quantization, prompt wording,
and few-shot example selection may affect results.

The datasets do not provide a verified chronological
drift transition. Evaluating them separately
does not demonstrate drift detection or adaptation.

The two invalid UCI responses are counted as
incorrect in accuracy and are reported separately
from the 2x2 confusion matrix.

## Simulated Drift Window Evaluation

Generate the stream:

```bash
python scripts/build_stream.py
```

Run the unadapted baseline across all windows:

```bash
python scripts/evaluate_windows.py \
  --output-dir results/window_baseline
```

The experiment uses 14 windows of 200 messages each.
Windows 1-5 contain UCI messages only.
The simulated transition begins in window 6.
Windows 10-14 contain Super SMS messages only.

Of 2,800 stream messages, 12 were excluded because
they were also used in the 15-shot prompt.
The final evaluation scored 2,788 messages.

Results are saved in:

- results/window_baseline/window_metrics.csv
- results/window_baseline/window_performance.png
- results/window_baseline/run_info.json

These curves are training-stream diagnostics,
not held-out test-set performance.
They show no sustained accuracy decline during
the simulated transition.

No drift detector, adaptation, or model updating
was applied during this experiment.

## Reference

Hur et al. (2026), Are large language models
effective for detecting spam messages?

https://link.springer.com/article/10.1007/s10207-026-01283-5
