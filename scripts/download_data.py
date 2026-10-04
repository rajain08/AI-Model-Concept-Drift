"""Download the raw datasets for the Self-Adaptive LLM concept drift project.
Usage:
    python scripts/download_data.py              # download anything missing, verify everything
    python scripts/download_data.py --force      # re-download and overwrite the manifest
"""

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

DATASETS = {
    "uci_sms_spam": {
        "url": "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
        "filename": "SMSSpamCollection",
        "zip_member": "SMSSpamCollection",
        # Not pinned yet: the first run records the hash in the manifest.
        # Paste it here once the team agrees on the file, so later runs verify against it.
        "sha256": "7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d",
        "role": "pre-drift",
        "licence": "CC BY 4.0 (attribution required)",
        "citation": "Almeida, T., & Hidalgo, J. M. G. (2011). SMS Spam Collection [Dataset]. "
                    "UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84",
    },
    "super_sms": {
        # Pinned to a specific commit so the file cannot change 
        "url": "https://raw.githubusercontent.com/smspamresearch/spstudy/"
               "a0889f36fb59e23a7f853a6932485d93d18b8afc/Data/super_sms_dataset.csv",
        "filename": "super_sms_dataset.csv",
        "zip_member": None,
        "sha256": "ba825b468b59fb229cc56c15a6c2d14d593a88051d54ea2f402166bf2ab894d3",
        "role": "post-drift",
        "licence": "No licence file; README requests citation of Salman et al. (2024). "
                   "Academic, non-commercial use only; do not redistribute.",
        "citation": "Salman, M., Ikram, M., & Kaafar, M. A. (2024). Super SMS Dataset [Dataset]. "
                    "GitHub. https://github.com/smspamresearch/spstudy",
    },
}


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "data298a-team1-downloader"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except Exception as err: 
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {err}") from err
            print(f"  attempt {attempt} failed ({err}); retrying...")
            time.sleep(2 * attempt)


def summarize_uci(data: bytes) -> dict:
    """Tab-separated lines: '<label>\t<message>'."""
    labels = {}
    rows = 0
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        label, _, _ = line.partition("\t")
        labels[label] = labels.get(label, 0) + 1
        rows += 1
    return {"rows": rows, "label_counts": labels, "encoding": "utf-8"}


def summarize_super_sms(data: bytes) -> dict:
    """CSV with columns SMSes, Labels (0 = ham, 1 = spam).

    The file mixes encodings (cp1252 punctuation plus bytes cp1252 cannot map),
    so it is read as latin-1, which never fails. Fix text encoding during cleaning,
    not here.
    """
    reader = csv.DictReader(io.StringIO(data.decode("latin-1")))
    labels = {}
    rows = 0
    for row in reader:
        label = (row.get("Labels") or "").strip() or "missing"
        labels[label] = labels.get(label, 0) + 1
        rows += 1
    return {"rows": rows, "label_counts": labels, "columns": reader.fieldnames,
            "encoding": "mixed; read as latin-1"}


SUMMARIZERS = {"uci_sms_spam": summarize_uci, "super_sms": summarize_super_sms}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="data/raw", help="where raw files are stored")
    parser.add_argument("--force", action="store_true", help="re-download and overwrite the manifest")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = {} if args.force or not manifest_path.exists() else json.loads(manifest_path.read_text())

    for name, spec in DATASETS.items():
        print(f"[{name}]")
        dest = data_dir / spec["filename"]

        if dest.exists() and not args.force:
            data = dest.read_bytes()
            print(f"  found {dest}")
        else:
            print(f"  downloading {spec['url']}")
            payload = fetch(spec["url"])
            if spec["zip_member"]:
                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    data = zf.read(spec["zip_member"])
            else:
                data = payload
            dest.write_bytes(data)
            print(f"  saved {dest} ({len(data):,} bytes)")

        digest = sha256_of(data)
        expected = spec["sha256"] or manifest.get(name, {}).get("sha256")
        if expected and digest != expected:
            print(f"  ERROR: hash mismatch.\n    expected {expected}\n    got      {digest}\n"
                  "  The file changed. Investigate before using it, or rerun with --force "
                  "if the team agrees to adopt the new version.")
            return 1
        if not spec["sha256"] and name not in manifest:
            print(f"  first download: recorded sha256 {digest}")

        summary = SUMMARIZERS[name](data)
        print(f"  rows: {summary['rows']:,}  labels: {summary['label_counts']}")

        if name not in manifest:
            manifest[name] = {
                "url": spec["url"],
                "file": str(dest),
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sha256": digest,
                "bytes": len(data),
                "role": spec["role"],
                "licence": spec["licence"],
                "citation": spec["citation"],
                **summary,
            }

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nManifest written to {manifest_path}. Commit the manifest, not the data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())