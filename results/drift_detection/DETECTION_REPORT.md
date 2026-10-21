# Drift Detection Evaluation (Phase 3 Report)

## Calibration Setup
- **Pre-Drift Reference Split**: First 1000 messages (Windows 1 to 5, 100% UCI SMS)
- **Drift Onset**: Message 1001 (Window 6, post-drift fraction = 0.10)
- **Total Windows Evaluated**: 14 (2800 total messages)

## Detector Performance Summary

| Detector | False Alarms (per 1,000 msgs) | First Alert Window | Delay (Windows) | Delay (Messages) | Total Alerts |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ADWIN (Error Stream)** | 0 | None | N/A | N/A | 0 |
| **Statistical KS Test** | 0 | 8 | 2 | 400 | 7 |
| **Statistical PSI** | 0 | 9 | 3 | 600 | 6 |
| **Embedding MMD (RBF)** | 0 | 7 | 1 | 200 | 8 |

## Thresholds Calibrated
- **ADWIN Delta**: 0.01
- **KS Statistic Threshold**: 0.0886
- **PSI Threshold**: 0.1000
- **MMD Threshold**: 0.0376

## Artifacts Generated
- `alerts.csv`: Detailed timestamped log of each triggered alert.
- `drift_metrics.csv`: Window-by-window values for error rate, KS, PSI, and MMD.
- `drift_detection_plot.png`: Multi-panel plot visualizing detector signals and alert triggers across the stream.
