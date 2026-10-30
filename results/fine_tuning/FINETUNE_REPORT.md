# Phase 5: Drift-Triggered Full Fine-Tuning Report

## 1. Drift Trigger Summary
- **Trigger Source**: `calibrated_detector_fallback` (Embedding MMD (RBF))
- **Trigger Window**: Window 7 (Stream Position 1201)
- **Detection Delay**: 200 messages (1 windows after Window 6 onset)
- **ADWIN Delta Tested**: 0.01 (ADWIN fired directly: `False`)
- **Fallback Note**: ADWIN had 0 alerts at delta=0.01 because unadapted error rate did not increase after drift onset; triggered via earliest calibrated drift alert (Embedding MMD (RBF) at window 7, pos 1201).

## 2. Adaptation Dataset & Full Fine-Tuning Setup
- **Model**: `unsloth/Llama-3.2-3B`
- **Optimizer**: `lomo` (100% Full Parameter Fine-Tuning)
- **Training Source**: `stream_post_trigger` (1600 messages: 320 spam, 1280 ham)
- **Validation Set**: 200 held-out validation messages
- **Zero Test Leakage Verified**: `True`
- **Total / Trainable Parameters**: 3,212,755,968 / 3,212,755,968 (100% Full Fine-Tuning)
- **Epochs / Batch Size / LR**: 2 / 1 / 0.0001

## 3. Held-Out Test Split Evaluation (New vs. Old Splits)

| Split | Dataset | Messages | Baseline Spam F1 | Adapted Spam F1 | F1 Delta | Adapted Accuracy | Adapted Precision | Adapted Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **New Test Split (Post-Drift)** | Super SMS (`test`) | 2000 | 87.48% | **73.80%** | -13.69% | 76.35% | 82.73% | 66.60% |
| **Old Test Split (Pre-Drift)** | UCI SMS (`test`) | 1031 | 49.21% | **43.84%** | -5.36% | 81.86% | 35.61% | 57.03% |

- **Post-Drift Spam F1**: **73.80%** (TP=666, TN=861, FP=139, FN=334)
- **Pre-Drift Spam F1**: **43.84%** (TP=73, TN=771, FP=132, FN=55)
- **Pre-Drift F1 Retention**: **89.10%** of unadapted baseline F1 (-5.36%)

## 4. Compute & GPU Time
- **Accelerator**: `mps` (Apple Silicon MPS (Metal Performance Shaders))
- **Precision**: `bfloat16`
- **Training Compute Time**: 6166.95 s (wall time: 6267.37 s)
- **Test Evaluation Time**: 217.23 s
- **Total GPU / Compute Time**: 6384.18 s (1.773384 hours)
- **Peak Accelerator Memory**: 6921.69 MB

## 5. Generated Artifacts
- `adapted_model/`: Fine-tuned Hugging Face model weights and tokenizer
- `training_logs.csv` & `training_history.json`: Step-by-step optimization and validation logs
- `super_sms_test_predictions.csv`: Predictions on post-drift Super SMS test split
- `uci_test_predictions.csv`: Predictions on pre-drift UCI test split
- `metrics.json`: Full reproducible metrics and metadata
- `adaptation_evaluation_plot.png`: Loss curve and F1 retention bar chart
