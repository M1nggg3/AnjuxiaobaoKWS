import argparse
import json
import random
import wave
from pathlib import Path

KEYWORD = "\u5b89\u5c45\u5c0f\u5b9d"
TOKENS = ["<blk>", "\u5b89", "\u5c45", "\u5c0f", "\u5b9d", "<filler>"]


def read_simple_yaml(path: Path):
    data = {}
    current = None
    list_key = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        stripped = line.strip()
        if list_key and stripped.startswith("- "):
            data[list_key].append(parse_value(stripped[2:].strip()))
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                if key.endswith("manifests"):
                    data[key] = []
                    list_key = key
                    current = None
                else:
                    data[key] = {}
                    current = key
                    list_key = None
            else:
                data[key] = parse_value(value)
                current = None
                list_key = None
        elif current and ":" in line:
            key, value = line.strip().split(":", 1)
            data[current][key.strip()] = parse_value(value.strip())
    return data


def parse_value(value):
    value = value.strip()
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        value = value[1:-1]
    try:
        return json.loads(f'"{value}"')
    except Exception:
        pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def read_manifest(path: Path, fallback_label_type: str):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            wav = Path(obj["path"]).resolve()
            if not wav.exists():
                raise FileNotFoundError(wav)
            text = obj.get("text", "")
            original_label_type = obj.get("label_type", fallback_label_type)
            if original_label_type == "positive":
                text = KEYWORD
            rows.append({
                "key": wav.stem,
                "wav": wav,
                "duration": duration_sec(wav),
                "label_type": original_label_type,
                "text": text,
                "category": obj.get("category", original_label_type),
                "prompt_id": obj.get("prompt_id", ""),
                "gender": obj.get("gender") or obj.get("prompt_gender", ""),
                "speed": obj.get("speed", ""),
            })
    return rows


def split_balanced(pos_rows, neg_rows, seed, train_ratio, dev_ratio):
    rng = random.Random(seed)
    pos_rows = list(pos_rows)
    neg_rows = list(neg_rows)
    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)

    def split_one(rows):
        n = len(rows)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        return {
            "train": rows[:n_train],
            "dev": rows[n_train:n_train + n_dev],
            "test": rows[n_train + n_dev:],
        }

    pos = split_one(pos_rows)
    neg = split_one(neg_rows)
    result = {}
    for split in ("train", "dev", "test"):
        mixed = pos[split] + neg[split]
        rng.shuffle(mixed)
        result[split] = mixed
    return result


def apply_train_repeats(splits, repeat_conf):
    if not isinstance(repeat_conf, dict) or not repeat_conf:
        return splits
    train_rows = list(splits["train"])
    extra = []
    for row in train_rows:
        repeat = int(repeat_conf.get(row["category"], 1))
        for rep_idx in range(2, repeat + 1):
            copied = dict(row)
            copied["key"] = f"{row['key']}__train_rep{rep_idx}"
            copied["oversample_from_key"] = row["key"]
            extra.append(copied)
    if extra:
        rng = random.Random(991)
        splits["train"] = train_rows + extra
        rng.shuffle(splits["train"])
    return splits


def write_dict(dict_dir: Path):
    dict_dir.mkdir(parents=True, exist_ok=True)
    (dict_dir / "dict.txt").write_text(
        "\n".join(f"{token} {idx}" for idx, token in enumerate(TOKENS)) + "\n",
        encoding="utf-8",
    )
    (dict_dir / "words.txt").write_text("<blk>\n<filler>\n", encoding="utf-8")


def write_split(split_dir: Path, rows):
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
                txt = " ".join(KEYWORD)
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
                "prompt_id": row["prompt_id"],
                "gender": row["gender"],
                "speed": row["speed"],
                "duration": round(row["duration"], 3),
            }, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/tts_pos500_neg1000.yaml")
    args = parser.parse_args()

    conf = read_simple_yaml(Path(args.config))
    positive_manifests = conf.get("positive_manifests") or [
        conf["positive_manifest"]
    ]
    negative_manifests = conf.get("negative_manifests") or [
        conf["negative_manifest"]
    ]
    pos_rows = []
    for manifest in positive_manifests:
        rows = read_manifest(Path(manifest), "positive")
        pos_rows.extend([
            row for row in rows
            if row["label_type"] == "positive"
            or row["text"] == KEYWORD
            or row["category"].startswith("positive")
        ])
    neg_rows = []
    for manifest in negative_manifests:
        rows = read_manifest(Path(manifest), "negative")
        neg_rows.extend([
            row for row in rows
            if row["label_type"] == "negative"
            or row["text"] == "<filler>"
            or row["category"].startswith("negative")
            or row["category"].endswith("noise")
        ])

    bad = [row for row in neg_rows if KEYWORD in row["text"]]
    if bad:
        raise RuntimeError(f"negative rows contain keyword: {bad[:5]}")

    splits = split_balanced(
        pos_rows,
        neg_rows,
        int(conf.get("seed", 777)),
        float(conf["split"]["train"]),
        float(conf["split"]["dev"]),
    )
    splits = apply_train_repeats(splits, conf.get("train_positive_category_repeat"))

    output_dir = Path(conf["output_dir"]).resolve()
    dict_dir = Path(conf["dict_dir"]).resolve()
    for split, rows in splits.items():
        write_split(output_dir / split, rows)
    write_dict(dict_dir)

    summary = {
        "name": conf["name"],
        "keyword": KEYWORD,
        "positive_count": len(pos_rows),
        "negative_count": len(neg_rows),
        "output_dir": str(output_dir),
        "dict_dir": str(dict_dir),
        "splits": {},
    }
    for split, rows in splits.items():
        summary["splits"][split] = {
            "total": len(rows),
            "positive": sum(row["label_type"] == "positive" for row in rows),
            "negative": sum(row["label_type"] == "negative" for row in rows),
            "duration_sec": round(sum(row["duration"] for row in rows), 3),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
