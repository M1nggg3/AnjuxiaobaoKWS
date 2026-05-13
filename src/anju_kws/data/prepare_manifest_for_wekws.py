from __future__ import annotations

import argparse
import json
import random
import wave
from pathlib import Path


KEYWORD_CHARS = ["\u5b89", "\u5c45", "\u5c0f", "\u5b9d"]
KEYWORD = "".join(KEYWORD_CHARS)
KEYWORD_TOKENS = " ".join(KEYWORD_CHARS)
TOKENS = ["<blk>", *KEYWORD_CHARS, "<filler>"]


def duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def read_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            wav = Path(obj["path"])
            if not wav.exists():
                raise FileNotFoundError(wav)
            rows.append({
                "key": obj.get("sample_id") or wav.stem,
                "wav": wav,
                "label_type": obj["label_type"],
                "text": obj.get("text", KEYWORD if obj["label_type"] == "positive" else "<filler>"),
                "duration": duration_sec(wav),
                "category": obj.get("category", obj["label_type"]),
            })
    return rows


def split_rows(rows: list[dict], seed: int, train_ratio: float, dev_ratio: float) -> dict[str, list[dict]]:
    pos = [row for row in rows if row["label_type"] == "positive"]
    neg = [row for row in rows if row["label_type"] == "negative"]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split_one(items: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        n = len(items)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        return items[:n_train], items[n_train:n_train + n_dev], items[n_train + n_dev:]

    p_train, p_dev, p_test = split_one(pos)
    n_train, n_dev, n_test = split_one(neg)
    splits = {
        "train": p_train + n_train,
        "dev": p_dev + n_dev,
        "test": p_test + n_test,
    }
    for split_rows_ in splits.values():
        rng.shuffle(split_rows_)
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
            if row["label_type"] == "positive":
                transcript = KEYWORD
                txt = KEYWORD_TOKENS
            else:
                transcript = row["text"]
                txt = "<filler>"
            wav_f.write(f"{key} {wav}\n")
            text_f.write(f"{key} {transcript}\n")
            dur_f.write(f"{key} {row['duration']:.3f}\n")
            list_f.write(json.dumps({
                "key": key,
                "wav": wav,
                "txt": txt,
                "duration": round(row["duration"], 3),
            }, ensure_ascii=True) + "\n")
            meta_f.write(json.dumps({
                "key": key,
                "label_type": row["label_type"],
                "transcript": transcript,
                "category": row["category"],
                "duration": round(row["duration"], 3),
            }, ensure_ascii=False) + "\n")


def write_dict(dict_dir: Path) -> None:
    dict_dir.mkdir(parents=True, exist_ok=True)
    (dict_dir / "dict.txt").write_text(
        "\n".join(f"{token} {idx}" for idx, token in enumerate(TOKENS)) + "\n",
        encoding="utf-8",
    )
    (dict_dir / "words.txt").write_text("<blk>\n<filler>\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dict_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--dev_ratio", type=float, default=0.10)
    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest))
    splits = split_rows(rows, args.seed, args.train_ratio, args.dev_ratio)
    output_dir = Path(args.output_dir)
    for split, split_rows_ in splits.items():
        write_split(output_dir / split, split_rows_)
    write_dict(Path(args.dict_dir))
    summary = {
        "manifest": args.manifest,
        "output_dir": args.output_dir,
        "dict_dir": args.dict_dir,
        "total": len(rows),
        "positive": sum(row["label_type"] == "positive" for row in rows),
        "negative": sum(row["label_type"] == "negative" for row in rows),
        "splits": {
            split: {
                "total": len(split_rows_),
                "positive": sum(row["label_type"] == "positive" for row in split_rows_),
                "negative": sum(row["label_type"] == "negative" for row in split_rows_),
            }
            for split, split_rows_ in splits.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
