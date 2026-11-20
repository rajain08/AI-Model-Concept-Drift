"""Causal detector benchmark using a disjoint calibration/monitoring split.

Windows 1-2 fit the reference, window 3 calibrates thresholds by bootstrap.
Only subsequent windows count toward false alarms and detection delay.
The ADWIN-style implementation is the project's existing approximation.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from benchmark_common import read_csv, read_json, sha256, validate_predictions, write_csv, write_json
from detect_drift import ADWIN, EmbeddingMMDDetector, calculate_ks_test, calculate_psi, extract_text_features

DETECTORS = ("adwin", "ks", "psi", "mmd")
DETECTOR_FAMILIES = {"adwin": "adwin", "ks": "statistical", "psi": "statistical", "mmd": "embedding"}


def summarize_alerts(alerts, stream, onset, monitoring_start):
    eligible = [r for r in stream if int(r["stream_position"]) >= monitoring_start]
    pre = [r for r in eligible if int(r["stream_position"]) < onset]
    pre_windows = {int(r["window"]) for r in pre}
    onset_window = next(int(r["window"]) for r in stream if int(r["stream_position"]) == onset)
    result = {}
    for detector in DETECTORS:
        events = sorted((a for a in alerts if a["detector"] == detector and a["stream_position"] >= monitoring_start), key=lambda a: a["stream_position"])
        false = [a for a in events if a["stream_position"] < onset]
        post = [a for a in events if a["stream_position"] >= onset]
        first = post[0] if post else None
        result[detector] = {
            "false_alert_count": len(false), "pre_drift_messages_evaluated": len(pre),
            "pre_drift_windows_evaluated": len(pre_windows),
            "false_alarms_per_1000": 1000 * len(false) / len(pre) if pre else None,
            "false_alarm_window_pct": 100 * len({a["window"] for a in false}) / len(pre_windows) if pre_windows else None,
            "detected": first is not None,
            "detection_delay_messages": first["stream_position"] - onset if first else None,
            "detection_delay_windows": first["window"] - onset_window if first else None,
            "first_alert_position": first["stream_position"] if first else None,
            "total_alerts": len(events),
        }
    return result


def evaluate_detectors(stream, predictions, manifest, seed, *, calibration_windows=3, bootstrap_samples=100):
    validate_predictions(predictions, stream, stream=True)
    windows = sorted({int(r["window"]) for r in stream})
    onset_window = manifest["drift_onset_window"]
    if not 2 <= calibration_windows < onset_window - 1:
        raise ValueError("Need reference, calibration, and at least one independent pre-drift monitoring window")
    if bootstrap_samples < 10:
        raise ValueError("At least 10 bootstrap samples are required")
    start = time.perf_counter()
    grouped = {w: [r for r in stream if int(r["window"]) == w] for w in windows}
    ref_rows = [r for r in stream if int(r["window"]) < calibration_windows]
    cal_rows = grouped[calibration_windows]
    ref_feats = extract_text_features([r["text"] for r in ref_rows])
    cal_feats = extract_text_features([r["text"] for r in cal_rows])
    embedder = EmbeddingMMDDetector(n_components=min(40, len(ref_rows) - 1), seed=seed)
    ref_embeds = embedder.fit_reference([r["text"] for r in ref_rows])
    cal_embeds = embedder.transform([r["text"] for r in cal_rows])
    rng = np.random.default_rng(seed)
    samples = {key: [] for key in ("ks", "psi", "mmd")}
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(cal_rows), size=len(cal_rows))
        samples["ks"].append(calculate_ks_test(ref_feats, cal_feats[indices])[0])
        samples["psi"].append(calculate_psi(ref_feats, cal_feats[indices])[0])
        samples["mmd"].append(embedder.compute_mmd(ref_embeds, cal_embeds[indices]))
    thresholds = {key: float(np.quantile(values, 0.99)) for key, values in samples.items()}
    adwin = ADWIN(delta=0.01)
    alerts, signals = [], []
    pred_by_id = {r["message_id"]: r for r in predictions}
    for window in windows:
        rows = grouped[window]
        monitoring = window > calibration_windows
        # Labels are used only after that message's prediction has been recorded.
        for row in rows:
            error = float(pred_by_id[row["message_id"]]["prediction"] != row["label"])
            detected = adwin.add_element(error)
            if detected and monitoring:
                alerts.append({"detector": "adwin", "window": window,
                               "stream_position": int(row["stream_position"]),
                               "metric_value": adwin.mean, "threshold": 0.01})
        signal = {"window": window, "phase": rows[0]["phase"], "monitoring": monitoring,
                  "adwin_mean_error": adwin.mean, "ks": None, "psi": None, "mmd": None}
        if monitoring:
            feats = extract_text_features([r["text"] for r in rows])
            embeds = embedder.transform([r["text"] for r in rows])
            signal.update(ks=calculate_ks_test(ref_feats, feats)[0],
                          psi=calculate_psi(ref_feats, feats)[0],
                          mmd=embedder.compute_mmd(ref_embeds, embeds))
            for detector, threshold in thresholds.items():
                if signal[detector] > threshold:
                    # A window statistic becomes available only at the window end.
                    alerts.append({"detector": detector, "window": window,
                                   "stream_position": int(rows[-1]["stream_position"]),
                                   "metric_value": signal[detector], "threshold": threshold})
        signals.append(signal)
    monitoring_start = int(grouped[calibration_windows + 1][0]["stream_position"])
    return {
        "seed": seed, "stream_sha256": manifest["stream_csv_sha256"],
        "calibration_windows": calibration_windows, "bootstrap_samples": bootstrap_samples,
        "monitoring_start_position": monitoring_start, "thresholds": thresholds,
        "detectors": summarize_alerts(alerts, stream, manifest["drift_onset_position"], monitoring_start),
        "runtime_seconds": time.perf_counter() - start,
        "notes": ["Calibration windows excluded from false-alarm exposure.",
                  "Window alerts timestamped at window end; ADWIN alerts at the observed message.",
                  "ADWIN is the project's approximate implementation, not the canonical River implementation.",
                  "MMD uses TF-IDF/SVD features, not pretrained semantic embeddings.",
                  "Bootstrap calibration is fixed in advance; no tuning on post-drift data."],
    }, alerts, signals


def save_detector_results(output, summary, alerts, signals):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    write_csv(output / "alerts.csv", alerts, ["detector", "window", "stream_position", "metric_value", "threshold"])
    write_csv(output / "drift_metrics.csv", signals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream-path", type=Path, default=Path("data/processed/stream/stream.csv"))
    parser.add_argument("--manifest-path", type=Path, default=Path("data/processed/stream/stream_manifest.json"))
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=29822)
    args = parser.parse_args()
    manifest = read_json(args.manifest_path)
    if sha256(args.stream_path) != manifest["stream_csv_sha256"]:
        raise ValueError("Stream hash differs from manifest")
    result = evaluate_detectors(read_csv(args.stream_path), read_csv(args.predictions_path), manifest, args.seed)
    save_detector_results(args.output_dir, *result)


if __name__ == "__main__":
    main()
