# Drift-Triggered LoRA Fine-Tuning & Complete 10-Metric Benchmark Report

## 1. Drift Trigger & Adaptation Dataset
- **Trigger Source**: `calibrated_detector_fallback` (Embedding MMD (RBF)) at **Window 7 (Stream Position 1201)**
- **Detection Delay**: 200 messages (1 window after Window 6 onset) | **Pre-Drift False Alarm Rate**: 0 per 1,000 messages
- **Training Source**: `stream_post_trigger` (1600 messages: 320 spam, 1280 ham)
- **Validation Set**: 200 held-out validation messages (Zero Test Leakage: `True`)

## 2. LoRA Configuration vs. Full Fine-Tuning
- **Base Model**: `unsloth/Llama-3.2-3B`
- **LoRA Rank ($r$) / Alpha ($\alpha$) / Dropout**: `16` / `32.0` / `0.05`
- **Target Modules**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (196 linear layers + classification head)
- **Trainable Parameters**: **24,320,000** / 3,237,069,824 (**0.751%** vs. 100% in Full Fine-Tuning)
- **Saved Adapter Size**: **46.40 MB** (`adapter_model.pt`) vs. **6,133.48 MB** (`model.safetensors` for Full Fine-Tuning)
- **Optimizer**: `AdamW` with class-weighted cross-entropy (`spam_weight = 4.0`)

## 3. Complete 10-Metric Benchmark Table (All Required + Method-Specific Metrics)

| # | Required Metric | 15-Shot Baseline *(Unadapted)* | Model 1: Full Fine-Tuning *(LOMO)* | Model 2: LoRA Fine-Tuning *(AdamW)* |
| :-: | :--- | :---: | :---: | :---: |
| **1** | **Accuracy (Post-Drift Super SMS / Pre-Drift UCI)** | 86.05% / 74.98% | 76.35% / 81.86% | **98.65% / 98.55%** |
| **2** | **Precision (Post-Drift Super SMS / Pre-Drift UCI)** | 79.33% / 32.98% | 82.73% / 35.61% | **99.90% / 96.69%** |
| **3** | **Spam Recall (Post-Drift Super SMS / Pre-Drift UCI)** | **97.50% / 96.88%** | 66.60% / 57.03% | **97.40% / 91.41%** |
| **4** | **F1-Score — Primary Metric (Post-Drift / Pre-Drift)** | 87.48% / 49.21% | 73.80% / 43.84% | **98.63% / 93.98%** |
| **5** | **False-Positive Rate — FPR (Post-Drift / Pre-Drift)** | 25.40% / 27.91% | 13.90% / 14.62% | **0.10% / 0.44%** |
| **6** | **PR-AUC (Post-Drift Super SMS / Pre-Drift UCI)** | N/A *(discrete text)* | 0.8493 / 0.3905 | **0.9991 / 0.9823** |
| **—** | **ROC-AUC (Post-Drift Super SMS / Pre-Drift UCI)** | N/A *(discrete text)* | 0.8378 / 0.8124 | **0.9985 / 0.9900** |
| **7** | **Post-Drift F1 Recovery ($\Delta$ vs. Baseline)** | 87.48% *(ref)* | 73.80% (-13.69%) | **98.63% (+11.15%)** |
| **8** | **Old-Data Retention / Forgetting ($\Delta\text{F1}_{\text{UCI}}$)** | 100.00% (0.00%) | 89.10% (-5.36% drop) | **190.98% (+44.77% gain, 0% forgetting)** |
| **9** | **Training Time / Inference Time (Test Latency)** | 0 s / 1,059.9 s (349.7 ms/msg) | 6,167.0 s / 217.2 s (71.7 ms/msg) | **715.2 s / 230.5 s (76.0 ms/msg)** |
| **10** | **GPU Memory / Total Compute Cost** | Local Ollama / 0.294 hrs | 6,921.7 MB / 1.773 MPS hrs | **7,057.7 MB / 0.265 MPS hrs** |
| **LLM** | **Output-Format Compliance (Valid Predictions)** | 99.81% (2 invalid on UCI) | **100.00% (0 invalid)** | **100.00% (0 invalid)** |
| **LoRA** | **Trainable Parameters & Saved Checkpoint Size** | 0 params (0 MB) | 3,212,755,968 (100% / 6,133.5 MB) | **24,320,000 (0.751% / 46.4 MB)** |
