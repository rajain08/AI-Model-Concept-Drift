
# Llama 3.2 3B Baseline Comparison

## Experiment setup

- Model: Llama 3.2 3B (Ollama)
- Inference: Local MacBook, quantized model
- Adaptation: None
- Few-shot examples: 15 fixed UCI training messages
- Seed: 29822
- Temperature: 0
- Maximum generated tokens: 128
- Data split: 60/20/20
- UCI test size: 1,031
- Super SMS test size: 2,000

## Results

| Metric | UCI | Super SMS |
|---|---:|---:|
| Accuracy | 75.46% | 86.25% |
| Spam precision | 33.42% | 79.69% |
| Spam recall | 96.88% | 97.30% |
| Spam F1 | 49.70% | 87.62% |
| Invalid responses | 2 | 0 |

## Comparison with Hur et al. (2026), Table 16

| Dataset | Published accuracy | Local accuracy |
|---|---:|---:|
| UCI | 91.35% | 75.46% |
| Super SMS | 72.15% | 86.25% |

The local results differ substantially from the
published values.

Some possible explanations include differences in
data sampling, cleaning, deduplication, dataset
splits, few-shot example selection, prompting
and quantized model inference.

The Super SMS sample is balanced between spam
and ham, unlike the UCI dataset. Therefore,
the accuracy values are not directly comparable
without accounting for class distributions.

The local baseline is a paper-inspired evaluation,
not an exact reproduction of the published study.

## Interpretation

The model achieved higher accuracy on Super SMS
than on UCI. This does not establish that concept
drift was absent or that the model adapted.

No adaptation or drift-detection mechanism was
applied in this experiment.

This baseline provides a reference for evaluating
future adaptation methods under the same conditions.

## Reference

Hur et al. (2026).
Are large language models effective for detecting
spam messages?

https://link.springer.com/article/10.1007/s10207-026-01283-5
