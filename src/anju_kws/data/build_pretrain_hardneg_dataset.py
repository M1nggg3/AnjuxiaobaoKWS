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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_row(row: dict, force_label: str | None = None, category: str | None = None) -> dict:
    wav = Path(row.get("path") or row.get("wav") or row.get("audio_filepath"))
    if not wav.exists():
        raise FileNotFoundError(wav)
    label_type = force_label or row.get("label_type")
    if label_type not in {"positive", "negative"}:
        raise ValueError(f"invalid label_type={label_type} for {wav}")
    return {
        "key": row.get("sample_id") or row.get("key") or wav.stem,
        "wav": wav,
        "label_type": label_type,
        "text": KEYWORD if label_type == "positive" else row.get("text", "<filler>"),
        "duration": float(row.get("duration_sec") or row.get("duration") or duration_sec(wav)),
        "category": category or row.get("category", label_type),
        "source_group": row.get("source_group", ""),
    }


def collect_rows(base_manifest: Path, hard_manifest: Path, real_noise_manifest: Path | None, max_real_noise_train: int, seed: int) -> list[dict]:
    rows = []
    for row in read_jsonl(base_manifest):
        rows.append(normalize_row(row))
    for row in read_jsonl(hard_manifest):
        rows.append(normalize_row(row, force_label="negative"))
    if real_noise_manifest:
        real_rows = [normalize_row(row, force_label="negative", category="real_noise_train") for row in read_jsonl(real_noise_manifest)]
        rng = random.Random(seed)
        rng.shuffle(real_rows)
        rows.extend(real_rows[:max_real_noise_train])
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
    splits = {"train": p_train + n_train, "dev": p_dev + n_dev, "test": p_test + n_test}
    for items in splits.values():
        rng.shuffle(items)
    return splits


def write_split(split_dir: Path, rows: list[dict]) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    with (split_dir / "data.list").open("w", encoding="utf-8") as list_f, \
            (split_dir / "metadata.jsonl").open("w", encoding="utf-8") as meta_f, \
            (split_dir / "text").open("w", encoding="utf-8") as text_f, \
            (split_dir / "wav.scp").open("w", encoding="utf-8") as wav_f, \
            (split_dir / "wav.dur").open("w", encoding="utf-8") as dur_f:
        for row in rows:
            txt = KEYWORD_TOKENS if row["label_type"] == "positive" else "<filler>"
            transcript = KEYWORD if row["label_type"] == "positive" else row["text"]
            wav = row["wav"].as_posix()
            list_f.write(json.dumps({
                "key": row["key"],
                "wav": wav,
                "txt": txt,
                "duration": round(row["duration"], 3),
            }, ensure_ascii=True) + "\n")
            meta_f.write(json.dumps({
                "key": row["key"],
                "label_type": row["label_type"],
                "transcript": transcript,
                "category": row["category"],
                "source_group": row["source_group"],
                "duration": round(row["duration"], 3),
            }, ensure_ascii=False) + "\n")
            text_f.write(f"{row['key']} {transcript}\n")
            wav_f.write(f"{row['key']} {wav}\n")
            dur_f.write(f"{row['key']} {row['duration']:.3f}\n")


def write_dict(dict_dir: Path) -> None:
    dict_dir.mkdir(parents=True, exist_ok=True)
    (dict_dir / "dict.txt").write_text(
        "\n".join(f"{token} {idx}" for idx, token in enumerate(TOKENS)) + "\n",
        encoding="utf-8",
    )
    (dict_dir / "words.txt").write_text("<blk>\n<filler>\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_manifest", default=r"E:\CodeWorking\Dataset\anju_xiaobao_aishell218_posneg_rknoise_20260509\manifest.jsonl")
    parser.add_argument("--hard_manifest", default=r"E:\CodeWorking\Dataset\anju_xiaobao_hard_negative_aishell218_rknoise_20260509\manifest.jsonl")
    parser.add_argument("--real_noise_manifest", default=r"E:\CodeWorking\Dataset\anju_xiaobao_kws_dataset_20260508\fixed_eval\negative_noise\manifest.jsonl")
    parser.add_argument("--output_dir", default=r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\data\prepared_pretrain_hardneg_20260509")
    parser.add_argument("--dict_dir", default=r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\dict\pretrain_hardneg_20260509")
    parser.add_argument("--max_real_noise_train", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--dev_ratio", type=float, default=0.10)
    args = parser.parse_args()

    rows = collect_rows(
        Path(args.base_manifest),
        Path(args.hard_manifest),
        Path(args.real_noise_manifest) if args.real_noise_manifest else None,
        args.max_real_noise_train,
        args.seed,
    )
    splits = split_rows(rows, args.seed, args.train_ratio, args.dev_ratio)
    output_dir = Path(args.output_dir)
    for split, items in splits.items():
        write_split(output_dir / split, items)
    write_dict(Path(args.dict_dir))

    summary = {
        "total": len(rows),
        "positive": sum(row["label_type"] == "positive" for row in rows),
        "negative": sum(row["label_type"] == "negative" for row in rows),
        "by_category": {},
        "splits": {},
        "base_manifest": args.base_manifest,
        "hard_manifest": args.hard_manifest,
        "real_noise_manifest": args.real_noise_manifest,
        "max_real_noise_train": args.max_real_noise_train,
    }
    for row in rows:
        summary["by_category"][row["category"]] = summary["by_category"].get(row["category"], 0) + 1
    for split, items in splits.items():
        summary["splits"][split] = {
            "total": len(items),
            "positive": sum(row["label_type"] == "positive" for row in items),
            "negative": sum(row["label_type"] == "negative" for row in items),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
