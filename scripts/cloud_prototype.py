"""Cloud-ready streaming prototype for prediction, drift, adaptation, and metrics.

The service is intentionally dependency-light so it can run on a container platform
without coupling the API to a particular cloud provider. Ollama is the default
prediction backend; ``MODEL_BACKEND=heuristic`` is an explicit local smoke-test
backend.

Run:
    python scripts/cloud_prototype.py
    curl http://localhost:8080/health
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import urllib.request
from collections import Counter
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("PROTOTYPE_DB", ROOT / "results" / "cloud_prototype.db"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("MODEL", "llama3.2:3b")
BACKEND = os.getenv("MODEL_BACKEND", "ollama").lower()
WINDOW_SIZE = max(10, int(os.getenv("DRIFT_WINDOW_SIZE", "50")))
REFERENCE_SIZE = max(WINDOW_SIZE, int(os.getenv("DRIFT_REFERENCE_SIZE", "100")))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "2.5"))

# Text features for drift detection
def text_features(text: str) -> tuple[float, ...]:
    length = max(len(text), 1)
    return (
        float(len(text)),
        float(len(text.split())),
        sum(char.isupper() for char in text) / length,
        sum(char.isdigit() for char in text) / length,
        sum(not char.isalnum() and not char.isspace() for char in text) / length,
    )

# Drift detection and metrics calculation
def drift_score(reference: list[tuple[float, ...]], current: list[tuple[float, ...]]) -> float:
    if not reference or not current:
        return 0.0
    scores = []
    for column in range(len(reference[0])):
        ref = [row[column] for row in reference]
        cur = [row[column] for row in current]
        mean = sum(ref) / len(ref)
        variance = sum((value - mean) ** 2 for value in ref) / max(len(ref), 1)
        scores.append(abs(sum(cur) / len(cur) - mean) / math.sqrt(variance + 1e-8))
    return sum(scores) / len(scores)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = tn = fp = fn = invalid = 0
    for row in rows:
        actual, predicted = row.get("label"), row["prediction"]
        if predicted not in {"ham", "spam"}:
            invalid += 1
        elif actual == "spam" and predicted == "spam":
            tp += 1
        elif actual == "ham" and predicted == "ham":
            tn += 1
        elif actual == "ham" and predicted == "spam":
            fp += 1
        elif actual == "spam" and predicted == "ham":
            fn += 1
    total = tp + tn + fp + fn + invalid
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "messages": total,
        "labeled_messages": tp + tn + fp + fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "invalid": invalid,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }

# Data storage and retrieval for events and adaptations, SQLite database management
class Store:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.db:
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY, received_at REAL NOT NULL, text TEXT NOT NULL,
                    label TEXT, prediction TEXT NOT NULL, raw_response TEXT,
                    drift_score REAL NOT NULL, drift_detected INTEGER NOT NULL,
                    adapted INTEGER NOT NULL, latency_ms REAL NOT NULL
                )"""
            )
            self.db.execute(
                """CREATE TABLE IF NOT EXISTS adaptations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL,
                    trigger_event_id TEXT NOT NULL, drift_score REAL NOT NULL,
                    examples_used INTEGER NOT NULL
                )"""
            )

    def event(self, row: dict[str, Any]) -> None:
        with self.lock, self.db:
            self.db.execute(
                """INSERT OR REPLACE INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["id"], row["received_at"], row["text"], row.get("label"),
                    row["prediction"], row["raw_response"], row["drift_score"],
                    int(row["drift_detected"]), int(row["adapted"]), row["latency_ms"],
                ),
            )

    def adaptation(self, event_id: str, score: float, examples: int) -> None:
        with self.lock, self.db:
            self.db.execute(
                "INSERT INTO adaptations(created_at, trigger_event_id, drift_score, examples_used) "
                "VALUES (?, ?, ?, ?)",
                (time.time(), event_id, score, examples),
            )

    def summary(self) -> dict[str, Any]:
        with self.lock:
            rows = [dict(row) for row in self.db.execute(
                "SELECT label, prediction FROM events WHERE label IS NOT NULL ORDER BY received_at"
            )]
            count = self.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            adaptations = self.db.execute("SELECT COUNT(*) FROM adaptations").fetchone()[0]
        return {"stored_events": count, "adaptations": adaptations, **metrics(rows)}

# Prediction and classification logic, including heuristic and Ollama backends, 
# adaptation based on recent events
class Predictor:
    def __init__(self) -> None:
        if BACKEND not in {"ollama", "heuristic"}:
            raise ValueError("MODEL_BACKEND must be 'ollama' or 'heuristic'")
        self.examples: list[tuple[str, str]] = []
        self.lock = threading.Lock()

    def adapt(self, events: list[dict[str, Any]]) -> int:
        with self.lock:
            for event in events:
                if event.get("label") in {"ham", "spam"}:
                    self.examples.append((event["text"], event["label"]))
            self.examples = self.examples[-100:]
            return len(self.examples)

    def predict(self, text: str) -> tuple[str, str]:
        if BACKEND == "heuristic":
            return self._heuristic(text)
        prompt = (
            "Classify the SMS as spam or ham. Reply with exactly one word.\n"
            + "".join(f"SMS: {example}\nLabel: {label}\n" for example, label in self.examples[-15:])
            + f"SMS: {json.dumps(text)}\nLabel:"
        )
        payload = json.dumps({
            "model": MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "num_predict": 8},
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{OLLAMA_URL.rstrip('/')}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = json.load(response).get("response", "").strip()
        match = re.match(r"(?i)^(spam|ham)\b", raw)
        return (match.group(1).lower() if match else "invalid"), raw

    def _heuristic(self, text: str) -> tuple[str, str]:
        spam_words = {"free", "winner", "win", "prize", "urgent", "claim", "offer", "cash"}
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        score = len(tokens & spam_words)
        with self.lock:
            for example, label in self.examples:
                overlap = len(tokens & set(re.findall(r"[a-z]+", example.lower())))
                score += overlap * (1 if label == "spam" else -1) / 20
        prediction = "spam" if score >= 1 else "ham"
        return prediction, f"heuristic_score={score:.3f}"

# Main pipeline for processing events and detecting drift
class Pipeline:
    def __init__(self) -> None:
        self.store = Store(DB_PATH)
        self.predictor = Predictor()
        self.reference: list[tuple[float, ...]] = []
        self.window: list[tuple[float, ...]] = []
        self.recent: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def process(self, event: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(event.get("text"), str) or not event["text"].strip():
            raise ValueError("Each event requires a non-empty text string")
        event_id = str(event.get("id") or hashlib.sha256(
            f"{time.time_ns()}:{event['text']}".encode()
        ).hexdigest()[:16])
        started = time.perf_counter()
        with self.lock:
            prediction, raw = self.predictor.predict(event["text"])
            features = text_features(event["text"])
            self.window.append(features)
            if len(self.reference) < REFERENCE_SIZE:
                self.reference.append(features)
            if len(self.window) >= WINDOW_SIZE:
                score = drift_score(self.reference, self.window)
                detected = score >= DRIFT_THRESHOLD and len(self.reference) >= REFERENCE_SIZE
                current = self.recent[-WINDOW_SIZE:]
                adapted = False
                if detected:
                    examples = self.predictor.adapt(current)
                    self.reference = self.window[-REFERENCE_SIZE:]
                    self.store.adaptation(event_id, score, examples)
                    adapted = True
                self.window.clear()
            else:
                score, detected, adapted = 0.0, False, False
            result = {
                **event, "id": event_id, "prediction": prediction, "raw_response": raw,
                "drift_score": round(score, 6), "drift_detected": detected,
                "adapted": adapted, "received_at": time.time(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
            self.recent.append(result)
            self.store.event(result)
            return result


PIPELINE = Pipeline()

# HTTP server handler for cloud ready streaming prototype, handling health checks, 
# metrics, and event processing
class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(HTTPStatus.OK, {"status": "ok", "backend": BACKEND, "model": MODEL})
        elif self.path == "/metrics":
            self._send(HTTPStatus.OK, PIPELINE.store.summary())
        else:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path not in {"/events", "/stream"}:
            self._send(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            events = payload.get("events", payload) if isinstance(payload, dict) else payload
            if isinstance(events, dict):
                events = [events]
            if not isinstance(events, list) or not events:
                raise ValueError("Body must be an event object or a non-empty events array")
            results = [PIPELINE.process(event) for event in events]
            self._send(HTTPStatus.OK, {"processed": len(results), "results": results})
        except (ValueError, KeyError, json.JSONDecodeError) as error:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self._send(HTTPStatus.BAD_GATEWAY, {"error": f"prediction backend failed: {error}"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def run() -> None:
    port = int(os.getenv("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"cloud prototype listening on :{port} using {BACKEND}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run()
