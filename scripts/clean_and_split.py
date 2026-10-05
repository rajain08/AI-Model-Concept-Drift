"""Clean the SMS datasets, remove leakage, and create deterministic splits.

Run from the repository root:
    python scripts/clean_and_split.py

Generated files are written to data/processed/, never data/raw/.
"""

import argparse
import csv
import hashlib
import io
import json
import random
import re
from collections import Counter
from pathlib import Path

from ftfy import fix_text
from ftfy import fix_encoding


DEFAULT_SEED = 29822
HUR_UCI_ROWS = 5572
SUPER_PER_LABEL = 5_000
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
EXPECTED_SUPER_DROPS = {"missing_label": 2, "empty_message": 1}
SUPER_LABELS = {"0": "ham", "1": "spam"}
VALID_LABELS = {"spam", "ham"}
UTF8_BOM = b"\xef\xbb\xbf"
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s()./-]{6,}\d)(?!\w)")
AMOUNT_RE = re.compile(
    r"(?<!\w)(?:[$£€]\s?\d+(?:[.,]\d+)?|\d+(?:[.,]\d+)?\s?(?:[$£€]|(?:p|bucks|dollars|pounds)))(?!\w)",
    re.I,
)
CSV_FIELDS = [
    "message_id", "dataset", "source_row", "label", "original_text", "text",
    "changed", "encoding_fixed", "ftfy_changed", "masked", "split",
]
REMOVED_FIELDS = [
    "message_id", "dataset", "source_row", "label", "original_text",
    "reason", "duplicate_of", "label_conflict",
]


# ---------------------------------------------------------------- reading ----

def decode_lines(raw: bytes) -> tuple[list[str], dict]:
    """Decode each line as UTF-8, falling back to cp1252.

    Splits on b"\\n" only (never on str.splitlines(), which also splits on
    \\x0b, \\x0c, \\x85, \\u2028, ...). Trailing "\\r" is left on each line.
    """
    if raw.startswith(UTF8_BOM):
        raw = raw[len(UTF8_BOM):]
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"":
        lines.pop()
    decoded = []
    stats = {"cp1252_fallback_lines": 0, "undecodable_lines": 0}
    for line in lines:
        try:
            decoded.append(line.decode("utf-8"))
            continue
        except UnicodeDecodeError:
            pass
        try:
            decoded.append(line.decode("cp1252"))
            stats["cp1252_fallback_lines"] += 1
        except UnicodeDecodeError:  # bytes undefined in cp1252 (0x81, 0x8d, ...)
            decoded.append(line.decode("cp1252", errors="replace"))
            stats["undecodable_lines"] += 1
    return decoded, stats


def clean_message(text: str, mask_uci: bool = False) -> tuple[str, dict]:
    """Return cleaned text plus flags saying which step changed it."""
    encoding_fixed = fix_encoding(text) != text
    cleaned = fix_text(text)
    ftfy_changed = cleaned != text
    masked = False
    if mask_uci:
        masked_text = AMOUNT_RE.sub("XXX", PHONE_RE.sub("XXX", cleaned))
        masked = masked_text != cleaned
        cleaned = masked_text
    return cleaned, {"encoding_fixed": encoding_fixed, "ftfy_changed": ftfy_changed, "masked": masked}


def make_row(prefix, dataset, source_row, label, original, mask=False) -> dict:
    cleaned, flags = clean_message(original, mask)
    return {
        "message_id": f"{prefix}-{source_row:06d}",
        "dataset": dataset,
        "source_row": source_row,
        "label": label,
        "original_text": original,
        "text": cleaned,
        "changed": original != cleaned,
        **flags,
    }


def read_uci(path: Path, mask_uci: bool) -> tuple[list[dict], dict]:
    lines, stats = decode_lines(path.read_bytes())
    rows, malformed, blank = [], [], 0
    for source_row, line in enumerate(lines, start=1):  # physical line number
        line = line.removesuffix("\r")
        if not line.strip():
            blank += 1
            continue
        label, separator, message = line.partition("\t")
        if not separator:
            malformed.append(source_row)
            continue
        rows.append(make_row("uci", "uci_sms_spam", source_row, label.strip().lower(), message, mask_uci))
    stats.update(blank_lines=blank, malformed_rows=len(malformed), malformed_source_rows=malformed[:20])
    return rows, stats


def read_super(path: Path) -> tuple[list[dict], dict]:
    lines, stats = decode_lines(path.read_bytes())
    reader = csv.DictReader(io.StringIO("\n".join(lines) + "\n", newline=""))
    if not {"SMSes", "Labels"} <= set(reader.fieldnames or []):
        raise ValueError(f"Unexpected Super SMS header: {reader.fieldnames}")
    rows, unrecognized = [], Counter()
    for source_row, record in enumerate(reader, start=2):  # CSV record number, header = 1
        raw_label = (record.get("Labels") or "").strip()
        label = SUPER_LABELS.get(raw_label, "")
        if raw_label and not label:
            unrecognized[raw_label] += 1
        rows.append(make_row("super", "super_sms", source_row, label, record.get("SMSes") or ""))
    if unrecognized:
        raise ValueError(f"Unrecognized Super SMS label values: {dict(unrecognized)}")
    return rows, stats


# --------------------------------------------------------------- cleaning ----

def dedup_key(text: str) -> str:
    return " ".join(text.casefold().split())


def drop_invalid(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop rows with a missing label or an empty message (original or cleaned).

    A row can have both problems; its "reason" is then "missing_label|empty_message".
    """
    kept, dropped = [], []
    for row in rows:
        if row["label"] and row["label"] not in VALID_LABELS:
            raise ValueError(f"Unrecognized label {row['label']!r} in {row['message_id']}")
        reasons = []
        if not row["label"]:
            reasons.append("missing_label")
        if not row["original_text"].strip() or not row["text"].strip():
            reasons.append("empty_message")
        if reasons:
            dropped.append({**row, "reason": "|".join(reasons)})
        else:
            kept.append(row)
    return kept, dropped


def count_reasons(dropped: list[dict]) -> dict:
    """Count each reason independently (a row with both counts under both)."""
    counts = Counter(reason for row in dropped for reason in row["reason"].split("|"))
    return dict(counts)


def deduplicate(rows: list[dict], seen: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Drop rows whose normalised text was already seen (within or across datasets)."""
    kept, removed = [], []
    for row in rows:
        key = dedup_key(row["text"])
        first = seen.get(key)
        if first is None:
            seen[key] = row
            kept.append(row)
            continue
        same_dataset = first["dataset"] == row["dataset"]
        removed.append({
            **row,
            "reason": "duplicate_within_dataset" if same_dataset else "duplicate_across_datasets",
            "duplicate_of": first["message_id"],
            "label_conflict": first["label"] != row["label"],
        })
    return kept, removed


def uci_ham_candidates(rows: list[dict]) -> list[dict]:
    """List UCI ham rows that could explain the gap to Hur et al. (for human review)."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(dedup_key(row["text"]), []).append(row)
    candidates = []
    for row in rows:
        if row["label"] != "ham":
            continue
        reasons, group = [], groups[dedup_key(row["text"])]
        if not row["text"].strip():
            reasons.append("empty_message")
        if "\ufffd" in row["original_text"] or CONTROL_RE.search(row["original_text"]):  # ftfy strips these from "text"
            reasons.append("bad_characters")
        if len({r["label"] for r in group}) > 1:
            reasons.append("label_conflict")
        if group[0] is not row:
            reasons.append("duplicate_of_earlier_row")
        if reasons:
            candidates.append({
                "message_id": row["message_id"], "source_row": row["source_row"],
                "reasons": "|".join(reasons), "duplicate_of": group[0]["message_id"] if group[0] is not row else "",
                "text": row["text"],
            })
    return candidates


# ------------------------------------------------------ sampling/splitting ----

def balanced_sample(rows: list[dict], seed: int, per_label: int) -> list[dict]:
    rng = random.Random(seed)
    sampled = []
    for label in ("spam", "ham"):
        candidates = [row for row in rows if row["label"] == label]
        if len(candidates) < per_label:
            raise ValueError(f"Not enough {label} rows for a {per_label}-row sample")
        rng.shuffle(candidates)
        sampled.extend(candidates[:per_label])
    rng.shuffle(sampled)
    return sampled


def stratified_split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    result = {name: [] for name in SPLIT_FRACTIONS}
    for label in ("spam", "ham"):
        group = [row for row in rows if row["label"] == label]
        rng.shuffle(group)
        test_count = round(len(group) * SPLIT_FRACTIONS["test"])
        val_count = round(len(group) * SPLIT_FRACTIONS["val"])
        result["test"].extend(group[:test_count])
        result["val"].extend(group[test_count:test_count + val_count])
        result["train"].extend(group[test_count + val_count:])
    for split, split_rows in result.items():
        rng.shuffle(split_rows)
        for row in split_rows:
            row["split"] = split
    return result


# ---------------------------------------------------------------- leakage ----

def leakage_check(split_rows, replay_ids: set[str] | None, rag_ids: set[str] | None) -> dict:
    ids = {name: {r["message_id"] for r in rows} for name, rows in split_rows.items()}
    keys = {name: {dedup_key(r["text"]) for r in rows} for name, rows in split_rows.items()}
    overlaps = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlaps[f"{a}_{b}_id"] = len(ids[a] & ids[b])
        overlaps[f"{a}_{b}_text"] = len(keys[a] & keys[b])
    replay_overlap = sorted(ids["test"] & replay_ids) if replay_ids is not None else []
    rag_overlap = sorted(ids["test"] & rag_ids) if rag_ids is not None else []
    return {
        "split_overlaps": overlaps,
        "test_replay_id_overlap": replay_overlap,
        "test_rag_id_overlap": rag_overlap,
        "replay_checked": replay_ids is not None,
        "rag_checked": rag_ids is not None,
        "passed": not any(overlaps.values()) and not replay_overlap and not rag_overlap,
    }


def read_id_file(path: Path | None) -> set[str] | None:
    return set(path.read_text(encoding="utf-8").split()) if path else None


# ----------------------------------------------------------------- output ----

def write_csv(path: Path, rows: list[dict], fields: list[str] = CSV_FIELDS) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def write_split_ids(directory: Path, split_rows: dict[str, list[dict]]) -> dict[str, str]:
    hashes = {}
    for split, rows in split_rows.items():
        path = directory / f"{split}.txt"
        path.write_text("".join(f"{row['message_id']}\n" for row in rows), encoding="utf-8")
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def changed_counts(rows: list[dict]) -> dict:
    return {key: sum(bool(r[key]) for r in rows) for key in ("changed", "encoding_fixed", "ftfy_changed", "masked")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--mask-uci", action="store_true", help="mask UCI phone numbers and amounts as XXX (off by default, pending team decision)")
    parser.add_argument("--uci-exclude-file", type=Path, help="one UCI message_id per line, after team review of uci_ham_candidates.csv")
    parser.add_argument("--replay-ids", type=Path, help="file of replay message IDs to check against test")
    parser.add_argument("--rag-ids", type=Path, help="file of RAG message IDs to check against test")
    parser.add_argument("--require-full-leakage-check", action="store_true", help="fail unless both --replay-ids and --rag-ids are given")
    parser.add_argument("--allow-unexpected-drops", action="store_true", help="do not fail if Super SMS drops differ from 2 missing labels + 1 empty message")
    args = parser.parse_args()
    if args.require_full_leakage_check and not (args.replay_ids and args.rag_ids):
        parser.error("--require-full-leakage-check needs both --replay-ids and --rag-ids")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits_dir = args.output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    # 1-2. Read, fix encoding, keep originals, count changes.
    uci_all, uci_read = read_uci(args.raw_dir / "SMSSpamCollection", args.mask_uci)
    super_all, super_read = read_super(args.raw_dir / "super_sms_dataset.csv")
    raw_rows = {"uci_sms_spam": len(uci_all), "super_sms": len(super_all)}
    changed = {"uci_sms_spam": changed_counts(uci_all), "super_sms": changed_counts(super_all)}

    # 4. Diagnostic for the UCI -> Hur gap (before any exclusion).
    candidates = uci_ham_candidates(uci_all)
    write_csv(args.output_dir / "uci_ham_candidates.csv", candidates,
              ["message_id", "source_row", "reasons", "duplicate_of", "text"])
    candidate_reasons = Counter(r for c in candidates for r in c["reasons"].split("|"))

    excluded = set(args.uci_exclude_file.read_text(encoding="utf-8").split()) if args.uci_exclude_file else set()
    unknown_excluded = excluded - {r["message_id"] for r in uci_all}
    if unknown_excluded:
        raise ValueError(f"--uci-exclude-file has unknown IDs: {sorted(unknown_excluded)[:10]}")
    uci_all = [r for r in uci_all if r["message_id"] not in excluded]

    # 3. Drop missing labels / empty messages (both datasets).
    uci_rows, uci_invalid = drop_invalid(uci_all)
    super_rows, super_invalid = drop_invalid(super_all)
    super_drop_counts = count_reasons(super_invalid)
    drops_match = super_drop_counts == EXPECTED_SUPER_DROPS
    if not drops_match and not args.allow_unexpected_drops:
        raise RuntimeError(f"Super SMS drops {super_drop_counts} != expected {EXPECTED_SUPER_DROPS}; "
                           "inspect the raw file or pass --allow-unexpected-drops")
    uci_before_dedup = len(uci_rows)

    # 5. Dedup within and across datasets (UCI first, so Super loses cross-dataset ties).
    seen: dict[str, dict] = {}
    uci_rows, uci_dups = deduplicate(uci_rows, seen)
    super_rows, super_dups = deduplicate(super_rows, seen)
    removed = uci_invalid + super_invalid + uci_dups + super_dups

    # 6-7. Sample and split.
    sampled_super = balanced_sample(super_rows, args.seed, SUPER_PER_LABEL)
    uci_splits = stratified_split(uci_rows, args.seed)
    super_splits = stratified_split(sampled_super, args.seed)

    # 8. Leakage check BEFORE anything is written.
    replay_ids, rag_ids = read_id_file(args.replay_ids), read_id_file(args.rag_ids)
    known_ids = {r["message_id"] for r in uci_rows} | {r["message_id"] for r in super_rows}
    cross_overlap = len({dedup_key(r["text"]) for r in uci_rows} & {dedup_key(r["text"]) for r in sampled_super})
    leakage = {
        "uci_sms_spam": leakage_check(uci_splits, replay_ids, rag_ids),
        "super_sms": leakage_check(super_splits, replay_ids, rag_ids),
        "cross_dataset_text_overlap": cross_overlap,
        "replay_ids_unknown": len(replay_ids - known_ids) if replay_ids is not None else None,
        "rag_ids_unknown": len(rag_ids - known_ids) if rag_ids is not None else None,
    }
    leakage["passed"] = all(leakage[d]["passed"] for d in ("uci_sms_spam", "super_sms")) and cross_overlap == 0
    leakage["fully_checked"] = replay_ids is not None and rag_ids is not None
    if not leakage["passed"]:
        raise RuntimeError(f"Leakage check failed (nothing written): {json.dumps(leakage, indent=2)}")
    if not leakage["fully_checked"]:
        print("WARNING: replay and/or RAG ID files not supplied; those leakage checks were skipped.")

    # Write outputs.
    write_csv(args.output_dir / "uci_messages.csv", uci_rows)
    write_csv(args.output_dir / "super_cleaned_full.csv", super_rows)  # population before sampling, for EDA
    write_csv(args.output_dir / "super_messages.csv", sampled_super)
    write_csv(args.output_dir / "removed_rows.csv", removed, REMOVED_FIELDS)
    split_hashes = {}
    for dataset, split_rows in (("uci_sms_spam", uci_splits), ("super_sms", super_splits)):
        dataset_dir = splits_dir / dataset
        dataset_dir.mkdir(exist_ok=True)
        split_hashes[dataset] = write_split_ids(dataset_dir, split_rows)

    label_counts = lambda rows: dict(Counter(r["label"] for r in rows))
    report = {
        "seed": args.seed,
        "mask_uci": args.mask_uci,
        "read": {"uci_sms_spam": uci_read, "super_sms": super_read},
        "raw_rows": raw_rows,
        "changed_text_rows": changed,
        "dropped": {
            "uci_excluded_by_review_file": len(excluded),
            "uci_invalid": count_reasons(uci_invalid),
            "super_invalid": super_drop_counts,
            "super_rows_dropped": len(super_invalid),
            "uci_duplicates": len(uci_dups),
            "super_duplicates_within_dataset": sum(r["reason"] == "duplicate_within_dataset" for r in super_dups),
            "super_duplicates_vs_uci": sum(r["reason"] == "duplicate_across_datasets" for r in super_dups),
            "duplicates_with_conflicting_labels": sum(bool(r["label_conflict"]) for r in uci_dups + super_dups),
        },
        "super_drops_match_expected": drops_match,
        "uci_hur_reference_rows": HUR_UCI_ROWS,
        "uci_rows_before_dedup": uci_before_dedup,
        "uci_hur_difference_before_dedup": uci_before_dedup - HUR_UCI_ROWS,
        "uci_ham_candidate_reasons": dict(candidate_reasons),
        "final_rows": {
            "uci_sms_spam": len(uci_rows),
            "super_sms_population_before_sampling": len(super_rows),
            "super_sms_sample": len(sampled_super),
        },
        "label_counts": {
            "uci_sms_spam": label_counts(uci_rows),
            "super_sms_population_before_sampling": label_counts(super_rows),
            "super_sms_sample": label_counts(sampled_super),
        },
        "splits": {
            dataset: {split: len(rows) for split, rows in split_rows.items()}
            for dataset, split_rows in (("uci_sms_spam", uci_splits), ("super_sms", super_splits))
        },
        "split_file_sha256": split_hashes,
        "leakage": leakage,
    }
    (args.output_dir / "cleaning_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())