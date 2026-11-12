# AI-Model-Concept-Drift

## Clone and run

From a fresh clone, run these commands in PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\run_data_pipeline.py
```

The pipeline runs:

```text
download -> clean/split -> build stream -> load SQLite warehouse
```

It writes the local warehouse and operational evidence to `warehouse/`:

- `concept_drift.sqlite`: SQLite warehouse
- `concept_drift.manifest.json`: source hashes, table schemas, and row counts
- `pipeline.log`: timestamped stage output, failures, and retry messages

Generated datasets and warehouse files are intentionally ignored by Git. The
raw-data manifest and source code are the reproducible project inputs.