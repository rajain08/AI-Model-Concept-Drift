"""Run the scheduled ingest -> transform -> warehouse pipeline.

Run from any directory:
    python scripts/run_data_pipeline.py

The existing download, cleaning, and stream stages remain the source of truth
for transformations. This script orchestrates them and loads their outputs into
one SQLite warehouse transaction, making repeated scheduled runs idempotent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_WAREHOUSE = ROOT / "warehouse" / "concept_drift.sqlite"

MESSAGE_FIELDS = [
    "message_id", "dataset", "source_row", "label", "original_text", "text",
    "changed", "encoding_fixed", "ftfy_changed", "masked", "split",
]
STREAM_FIELDS = [
    "stream_position", "window", "phase", "post_drift_fraction",
    "message_id", "dataset", "is_post_drift", "label", "text",
]
WINDOW_FIELDS = [
    "window", "phase", "post_drift_fraction", "messages",
    "pre_drift_messages", "post_drift_messages", "spam", "ham",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path, fields: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required pipeline output not found: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in fields if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        return list(reader)


def run_stage(command: list[str], log_handle) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    log_handle.write(f"\n$ {' '.join(command)}\n")
    log_handle.flush()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_handle.write(line)
    process.wait()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


def create_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            status TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            stream_count INTEGER NOT NULL DEFAULT 0,
            window_count INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            dataset TEXT NOT NULL,
            source_row INTEGER NOT NULL,
            label TEXT NOT NULL,
            original_text TEXT NOT NULL,
            text TEXT NOT NULL,
            changed INTEGER NOT NULL,
            encoding_fixed INTEGER NOT NULL,
            ftfy_changed INTEGER NOT NULL,
            masked INTEGER NOT NULL,
            split TEXT NOT NULL,
            loaded_run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id)
        );
        CREATE TABLE IF NOT EXISTS stream_messages (
            stream_position INTEGER PRIMARY KEY,
            window INTEGER NOT NULL,
            phase TEXT NOT NULL,
            post_drift_fraction REAL NOT NULL,
            message_id TEXT NOT NULL REFERENCES messages(message_id),
            dataset TEXT NOT NULL,
            is_post_drift INTEGER NOT NULL,
            label TEXT NOT NULL,
            text TEXT NOT NULL,
            loaded_run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id)
        );
        CREATE TABLE IF NOT EXISTS stream_windows (
            window INTEGER PRIMARY KEY,
            phase TEXT NOT NULL,
            post_drift_fraction REAL NOT NULL,
            messages INTEGER NOT NULL,
            pre_drift_messages INTEGER NOT NULL,
            post_drift_messages INTEGER NOT NULL,
            spam INTEGER NOT NULL,
            ham INTEGER NOT NULL,
            loaded_run_id TEXT NOT NULL REFERENCES pipeline_runs(run_id)
        );
        """
    )


def value(row: dict[str, str], field: str) -> object:
    raw = row[field]
    if field in {"source_row", "stream_position", "window", "messages",
                 "pre_drift_messages", "post_drift_messages", "spam", "ham"}:
        return int(raw)
    if field in {"changed", "encoding_fixed", "ftfy_changed", "masked",
                 "is_post_drift"}:
        if raw.lower() in {"true", "1"}:
            return 1
        if raw.lower() in {"false", "0"}:
            return 0
        raise ValueError(f"Expected a boolean value for {field}, got {raw!r}")
    if field == "post_drift_fraction":
        return float(raw)
    return raw


def insert_rows(
    db: sqlite3.Connection,
    table: str,
    fields: list[str],
    rows: Iterable[dict[str, str]],
    run_id: str,
) -> int:
    columns = fields + ["loaded_run_id"]
    placeholders = ", ".join("?" for _ in columns)
    query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    values = [tuple(value(row, field) for field in fields) + (run_id,) for row in rows]
    db.executemany(query, values)
    return len(values)


def table_manifest(db: sqlite3.Connection) -> list[dict[str, object]]:
    tables = ["pipeline_runs", "messages", "stream_messages", "stream_windows"]
    result = []
    for table in tables:
        columns = [
            {"name": row[1], "type": row[2], "not_null": bool(row[3])}
            for row in db.execute(f"PRAGMA table_info({table})")
        ]
        count = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        result.append({"table": table, "row_count": count, "columns": columns})
    return result


def load_warehouse(
    warehouse_path: Path,
    processed_dir: Path,
    run_id: str,
    started_at: str,
) -> dict[str, int]:
    messages = []
    for dataset in ("uci_messages.csv", "super_messages.csv"):
        messages.extend(read_csv(processed_dir / dataset, MESSAGE_FIELDS))
    stream_dir = processed_dir / "stream"
    stream = read_csv(stream_dir / "stream.csv", STREAM_FIELDS)
    windows = read_csv(stream_dir / "windows.csv", WINDOW_FIELDS)

    message_ids = {row["message_id"] for row in messages}
    stream_ids = [row["message_id"] for row in stream]
    if len(message_ids) != len(messages):
        raise RuntimeError("Cleaned message IDs are not unique")
    missing = sorted(set(stream_ids) - message_ids)
    if missing:
        raise RuntimeError(f"Stream references {len(missing)} messages absent from the warehouse load")

    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(warehouse_path)
    db.execute("PRAGMA foreign_keys = ON")
    try:
        create_schema(db)
        db.execute(
            "INSERT INTO pipeline_runs(run_id, started_at, status) VALUES (?, ?, 'running')",
            (run_id, started_at),
        )
        with db:
            db.execute("DELETE FROM stream_messages")
            db.execute("DELETE FROM stream_windows")
            db.execute("DELETE FROM messages")
            message_count = insert_rows(db, "messages", MESSAGE_FIELDS, messages, run_id)
            stream_count = insert_rows(db, "stream_messages", STREAM_FIELDS, stream, run_id)
            window_count = insert_rows(db, "stream_windows", WINDOW_FIELDS, windows, run_id)
            completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            db.execute(
                """UPDATE pipeline_runs
                   SET completed_at = ?, status = 'succeeded',
                       message_count = ?, stream_count = ?, window_count = ?
                   WHERE run_id = ?""",
                (completed_at, message_count, stream_count, window_count, run_id),
            )
        counts = {
            "messages": message_count,
            "stream_messages": stream_count,
            "stream_windows": window_count,
        }
        manifest = {
            "warehouse": str(warehouse_path),
            "run_id": run_id,
            "loaded_at": completed_at,
            "source_files": {
                "messages": [
                    {"path": str(processed_dir / name), "sha256": sha256_file(processed_dir / name)}
                    for name in ("uci_messages.csv", "super_messages.csv")
                ],
                "stream": {
                    "path": str(stream_dir / "stream.csv"),
                    "sha256": sha256_file(stream_dir / "stream.csv"),
                },
                "windows": {
                    "path": str(stream_dir / "windows.csv"),
                    "sha256": sha256_file(stream_dir / "windows.csv"),
                },
            },
            "tables": table_manifest(db),
        }
        manifest_path = warehouse_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Warehouse written to {warehouse_path}")
        print(f"Warehouse manifest written to {manifest_path}")
        return counts
    except Exception as error:
        db.rollback()
        db.execute(
            "UPDATE pipeline_runs SET completed_at = ?, status = 'failed', error = ? WHERE run_id = ?",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), str(error), run_id),
        )
        db.commit()
        raise
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "drift_stream.yaml")
    parser.add_argument("--warehouse", type=Path, default=DEFAULT_WAREHOUSE)
    parser.add_argument(
        "--log-file", type=Path,
        help="pipeline log file (default: <warehouse directory>/pipeline.log)",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--mask-uci", action="store_true")
    parser.add_argument("--allow-unexpected-drops", action="store_true")
    args = parser.parse_args()
    args.raw_dir = args.raw_dir.resolve()
    args.processed_dir = args.processed_dir.resolve()
    args.config = args.config.resolve()
    args.warehouse = args.warehouse.resolve()
    args.log_file = (args.log_file or args.warehouse.parent / "pipeline.log").resolve()

    run_id = uuid.uuid4().hex
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    download = [sys.executable, str(SCRIPTS / "download_data.py"), "--data-dir", str(args.raw_dir)]
    if args.force_download:
        download.append("--force")
    clean = [
        sys.executable, str(SCRIPTS / "clean_and_split.py"),
        "--raw-dir", str(args.raw_dir), "--output-dir", str(args.processed_dir),
    ]
    if args.mask_uci:
        clean.append("--mask-uci")
    if args.allow_unexpected_drops:
        clean.append("--allow-unexpected-drops")
    build = [
        sys.executable, str(SCRIPTS / "build_stream.py"),
        "--config", str(args.config), "--processed-dir", str(args.processed_dir),
    ]
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"\n[{started_at}] START run_id={run_id}\n"
            f"warehouse={args.warehouse}\n"
        )
        try:
            run_stage(download, log_handle)
            run_stage(clean, log_handle)
            run_stage(build, log_handle)
            counts = load_warehouse(args.warehouse, args.processed_dir, run_id, started_at)
            message = f"Pipeline succeeded: {counts}"
            print(message)
            log_handle.write(message + "\n")
            return 0
        except Exception as error:
            message = f"Pipeline failed: {error}"
            print(message, file=sys.stderr)
            log_handle.write(message + "\n")
            return 1
        finally:
            log_handle.write(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] END\n")


if __name__ == "__main__":
    sys.exit(main())
