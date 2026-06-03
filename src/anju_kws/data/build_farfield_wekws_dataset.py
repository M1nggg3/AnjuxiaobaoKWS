from __future__ import annotations

import argparse
import json
import random
import wave
from collections import Counter
from pathlib import Path


KEYWORD_CHARS = ["\u5b89", "\u5c45", "\u5c0f", "\u5b9d"]
KEYWORD = "".join(KEYWORD_CHARS)
KEYWORD_TOKENS = " ".join(KEYWORD_CHARS)
TOKENS = ["<blk>", *KEYWORD_CHARS, "<filler>"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")


def jsonable(row: dict) -> dict:
    return {
        key: value.as_posix() if isinstance(value, Path) else value
        for key, value in row.items()
    }


def duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def normalize_path(path: str, dataset_root: Path) -> Path:
    raw = Path(path)
    if raw.exists():
        return raw
    candidate = dataset_root / raw.name
    if candidate.exists():
        return candidate
    for subdir in ["clean_positive", "clean_negative", "farfield_positive", "farfield_negative"]:
        candidate = dataset_root / subdir / raw.name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(path)


def load_rows(dataset_root: Path, include_clean: bool) -> list[dict]:
    manifests = [
        ("farfield_positive", dataset_root / "manifests" / "farfield_positive.jsonl"),
        ("farfield_negative", dataset_root / "manifests" / "farfield_negative.jsonl"),
    ]
    if include_clean:
        manifests.extend([
            ("clean_positive", dataset_root / "manifests" / "clean_positive_filtered.jsonl"),
            ("clean_negative", dataset_root / "manifests" / "clean_negative_filtered.jsonl"),
        ])

    rows: list[dict] = []
    for group, manifest in manifests:
        for row in read_jsonl(manifest):
            wav_path = normalize_path(row["path"], dataset_root)
            label_type = row["label_type"]
            rows.append({
                "key": row.get("sample_id") or wav_path.stem,
                "wav": wav_path,
                "label_type": label_type,
                "txt": KEYWORD_TOKENS if label_type == "positive" else "<filler>",
                "transcript": KEYWORD if label_type == "positive" else row.get("text", "<filler>"),
                "duration": float(row.get("duration_sec") or duration_sec(wav_path)),
                "category": group,
                "source_group": row.get("source_group", group),
                "farfield_profile": row.get("farfield_profile"),
                "snr_db": row.get("snr_db"),
            })
    return rows


def load_mined_hard_negatives(
    data_list: Path,
    score_file: Path,
    min_score: float,
    repeat: int,
) -> list[dict]:
    if repeat <= 0:
        return []

    detected: dict[str, float] = {}
    with score_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[1] == "detected":
                score = float(parts[3])
                if score >= min_score:
                    detected[parts[0]] = score

    base_rows: dict[str, dict] = {}
    with data_list.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["key"] in detected:
                wav_path = Path(row["wav"])
                base_rows[row["key"]] = {
                    "key": row["key"],
                    "wav": wav_path,
                    "label_type": "negative",
                    "txt": "<filler>",
                    "transcript": "<filler>",
                    "duration": float(row.get("duration") or duration_sec(wav_path)),
                    "category": "mined_continuous_false_alarm",
                    "source_group": "rk3566_continuous_false_alarm_mined",
                    "farfield_profile": None,
                    "snr_db": None,
                    "mined_score": detected[row["key"]],
                }

    rows: list[dict] = []
    for idx in range(repeat):
        for row in base_rows.values():
            dup = dict(row)
            dup["key"] = f"{row['key']}_mine{idx + 1:03d}"
            dup["is_mined_hard_negative"] = True
            dup["mined_repeat_index"] = idx + 1
            rows.append(dup)
    return rows


def balance_rows(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    positives = [row for row in rows if row["label_type"] == "positive"]
    negatives = [row for row in rows if row["label_type"] == "negative"]
    if not positives or not negatives:
        raise ValueError(f"Need both positive and negative rows, got pos={len(positives)} neg={len(negatives)}")
    if len(negatives) >= len(positives):
        return rows

    balanced = list(rows)
    needed = len(positives) - len(negatives)
    for idx in range(needed):
        base = rng.choice(negatives)
        dup = dict(base)
        dup["key"] = f"{base['key']}_negdup{idx + 1:05d}"
        dup["category"] = f"{base['category']}_balanced_repeat"
        dup["is_balanced_repeat"] = True
        balanced.append(dup)
    return balanced


def split_rows(rows: list[dict], seed: int, train_ratio: float, dev_ratio: float) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for label_type in ["positive", "negative"]:
        items = [row for row in rows if row["label_type"] == label_type]
        rng.shuffle(items)
        n_train = int(len(items) * train_ratio)
        n_dev = int(len(items) * dev_ratio)
        splits["train"].extend(items[:n_train])
        splits["dev"].extend(items[n_train:n_train + n_dev])
        splits["test"].extend(items[n_train + n_dev:])
    for items in splits.values():
        rng.shuffle(items)
    return splits


def write_split(split_dir: Path, rows: list[dict]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "wav.scp").open("w", encoding="utf-8") as wav_f, \
            (split_dir / "text").open("w", encoding="utf-8") as text_f, \
            (split_dir / "wav.dur").open("w", encoding="utf-8") as dur_f, \
            (split_dir / "data.list").open("w", encoding="utf-8") as list_f, \
            (split_dir / "metadata.jsonl").open("w", encoding="utf-8") as meta_f:
        for row in rows:
            key = row["key"]
            wav = row["wav"].as_posix()
            wav_f.write(f"{key} {wav}\n")
            text_f.write(f"{key} {row['transcript']}\n")
            dur_f.write(f"{key} {row['duration']:.3f}\n")
            list_f.write(json.dumps({
                "key": key,
                "wav": wav,
                "txt": row["txt"],
                "duration": round(row["duration"], 3),
            }, ensure_ascii=True) + "\n")
            meta_f.write(json.dumps({
                "key": key,
                "label_type": row["label_type"],
                "transcript": row["transcript"],
                "category": row["category"],
                "source_group": row["source_group"],
                "farfield_profile": row.get("farfield_profile"),
                "snr_db": row.get("snr_db"),
                "is_balanced_repeat": bool(row.get("is_balanced_repeat", False)),
                "duration": round(row["duration"], 3),
            }, ensure_ascii=False) + "\n")


def write_dict(dict_dir: Path) -> None:
    dict_dir.mkdir(parents=True, exist_ok=True)
    (dict_dir / "dict.txt").write_text(
        "\n".join(f"{token} {idx}" for idx, token in enumerate(TOKENS)) + "\n",
        encoding="utf-8",
    )
    (dict_dir / "words.txt").write_text("<blk>\n<filler>\n", encoding="utf-8")


def summarize(rows: list[dict]) -> dict:
    return {
        "total": len(rows),
        "positive": sum(row["label_type"] == "positive" for row in rows),
        "negative": sum(row["label_type"] == "negative" for row in rows),
        "by_category": dict(Counter(row["category"] for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dict_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--dev_ratio", type=float, default=0.10)
    parser.add_argument("--include_clean", action="store_true")
    parser.add_argument("--balance_negatives", action="store_true")
    parser.add_argument("--hard_negative_data_list")
    parser.add_argument("--hard_negative_score_file")
    parser.add_argument("--hard_negative_min_score", type=float, default=0.0)
    parser.add_argument("--hard_negative_repeat", type=int, default=0)
    parser.add_argument("--hard_negative_train_only", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    dict_dir = Path(args.dict_dir)
    rows = load_rows(dataset_root, include_clean=args.include_clean)
    original_summary = summarize(rows)
    if args.balance_negatives:
        rows = balance_rows(rows, args.seed)
    balanced_summary = summarize(rows)
    splits = split_rows(rows, args.seed, args.train_ratio, args.dev_ratio)
    mined_rows: list[dict] = []
    if args.hard_negative_data_list and args.hard_negative_score_file:
        mined_rows = load_mined_hard_negatives(
            Path(args.hard_negative_data_list),
            Path(args.hard_negative_score_file),
            args.hard_negative_min_score,
            args.hard_negative_repeat,
        )
        if args.hard_negative_train_only:
            splits["train"].extend(mined_rows)
        else:
            rows.extend(mined_rows)
            splits = split_rows(rows, args.seed, args.train_ratio, args.dev_ratio)

    for split, split_rows_ in splits.items():
        write_split(output_dir / split, split_rows_)
    write_jsonl(output_dir / "all_rows.jsonl", rows)
    write_dict(dict_dir)

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "dict_dir": str(dict_dir),
        "include_clean": args.include_clean,
        "balance_negatives": args.balance_negatives,
        "original": original_summary,
        "balanced": balanced_summary,
        "mined_hard_negative": summarize(mined_rows),
        "hard_negative_data_list": args.hard_negative_data_list,
        "hard_negative_score_file": args.hard_negative_score_file,
        "hard_negative_min_score": args.hard_negative_min_score,
        "hard_negative_repeat": args.hard_negative_repeat,
        "hard_negative_train_only": args.hard_negative_train_only,
        "splits": {split: summarize(split_rows_) for split, split_rows_ in splits.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
