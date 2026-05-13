import argparse
import json
import random
import shutil
import wave
from pathlib import Path


KEYWORD = "\u5b89\u5c45\u5c0f\u5b9d"
TOKENS = ["<blk>", "\u5b89", "\u5c45", "\u5c0f", "\u5b9d", "<filler>"]


def duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def split_rows(rows, seed=777, train_ratio=0.8, dev_ratio=0.1):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n = len(rows)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    return {
        "train": rows[:n_train],
        "dev": rows[n_train:n_train + n_dev],
        "test": rows[n_train + n_dev:],
    }


def write_split(split_dir: Path, rows):
    split_dir.mkdir(parents=True, exist_ok=True)
    wav_scp = split_dir / "wav.scp"
    text = split_dir / "text"
    wav_dur = split_dir / "wav.dur"
    data_list = split_dir / "data.list"

    with wav_scp.open("w", encoding="utf-8") as wav_f, \
            text.open("w", encoding="utf-8") as text_f, \
            wav_dur.open("w", encoding="utf-8") as dur_f, \
            data_list.open("w", encoding="utf-8") as list_f:
        for row in rows:
            key = row["key"]
            wav = row["wav"].as_posix()
            txt = " ".join(KEYWORD)
            dur = row["duration"]
            wav_f.write(f"{key} {wav}\n")
            text_f.write(f"{key} {KEYWORD}\n")
            dur_f.write(f"{key} {dur:.3f}\n")
            list_f.write(json.dumps({
                "key": key,
                "wav": wav,
                "txt": txt,
                "duration": round(dur, 3),
            }, ensure_ascii=True) + "\n")


def write_dict(dict_dir: Path):
    dict_dir.mkdir(parents=True, exist_ok=True)
    with (dict_dir / "dict.txt").open("w", encoding="utf-8") as f:
        for idx, token in enumerate(TOKENS):
            f.write(f"{token} {idx}\n")
    with (dict_dir / "words.txt").open("w", encoding="utf-8") as f:
        f.write("<blk>\n")
        f.write("<filler>\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--output_dir", default="data/prepared_tts_smoke")
    parser.add_argument("--dict_dir", default="dict/tts_smoke")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--copy_raw", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    wav_dir = source_dir / "wav"
    manifest = source_dir / "manifest.jsonl"
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"missing wav dir: {wav_dir}")
    if not manifest.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest}")

    rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            wav = Path(obj["path"]).resolve()
            if not wav.exists():
                wav = wav_dir / Path(obj["path"]).name
            if not wav.exists():
                raise FileNotFoundError(f"missing wav: {wav}")
            rows.append({
                "key": wav.stem,
                "wav": wav,
                "duration": duration_sec(wav),
            })

    if len(rows) == 0:
        raise RuntimeError("no wav rows found")

    output_dir = Path(args.output_dir).resolve()
    if args.copy_raw:
        raw_dir = Path("data/raw/tts_positive").resolve()
        raw_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for row in rows:
            dst = raw_dir / row["wav"].name
            if not dst.exists():
                shutil.copy2(row["wav"], dst)
            row["wav"] = dst
            copied.append(row)
        rows = copied

    splits = split_rows(rows, seed=args.seed)
    for split, split_rows_ in splits.items():
        write_split(output_dir / split, split_rows_)

    write_dict(Path(args.dict_dir).resolve())

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "dict_dir": str(Path(args.dict_dir).resolve()),
        "keyword": KEYWORD,
        "total": len(rows),
        "splits": {k: len(v) for k, v in splits.items()},
        "duration_total_sec": round(sum(row["duration"] for row in rows), 3),
    }
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
