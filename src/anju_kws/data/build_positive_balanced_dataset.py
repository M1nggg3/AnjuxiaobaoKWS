from __future__ import annotations

import argparse
import json
import math
import random
import wave
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF


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


def load_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path), backend="soundfile")
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = AF.resample(wav, sr, sample_rate)
    return wav


def crop_or_tile_noise(noise: torch.Tensor, target_len: int, rng: random.Random) -> torch.Tensor:
    if noise.shape[1] >= target_len:
        start = rng.randint(0, noise.shape[1] - target_len)
        return noise[:, start:start + target_len]
    repeat = math.ceil(target_len / noise.shape[1])
    return noise.repeat(1, repeat)[:, :target_len]


def mix_with_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    clean_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-12)
    noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-12)
    mixed = clean + noise * (clean_rms / (10 ** (snr_db / 20.0)) / noise_rms)
    peak = mixed.abs().max().item()
    if peak > 0.98:
        mixed = mixed / peak * 0.98
    return mixed.clamp(-1.0, 1.0)


def pad_silence(wav: torch.Tensor, sample_rate: int, pre_ms: int, post_ms: int) -> torch.Tensor:
    pre = torch.zeros((1, int(sample_rate * pre_ms / 1000)))
    post = torch.zeros((1, int(sample_rate * post_ms / 1000)))
    return torch.cat([pre, wav, post], dim=1)


def speed_perturb(wav: torch.Tensor, factor: float, sample_rate: int = 16000) -> torch.Tensor:
    if abs(factor - 1.0) < 1e-4:
        return wav
    new_sr = int(sample_rate * factor)
    return AF.resample(wav, new_sr, sample_rate)


def save_audio(path: Path, wav: torch.Tensor, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = wav.abs().max().item()
    if peak > 0.98:
        wav = wav / peak * 0.98
    torchaudio.save(str(path), wav.clamp(-1.0, 1.0), sample_rate, encoding="PCM_S", bits_per_sample=16)


def make_real_positive_augments(real_rows: list[dict], noise_rows: list[dict], output_dir: Path, seed: int) -> list[dict]:
    rng = random.Random(seed)
    aug_dir = output_dir / "wav" / "real_positive_augmented"
    noise_cache: dict[Path, torch.Tensor] = {}
    variants = [
        {"snr": 30, "gain": 0.95, "speed": 1.00, "pre": 80, "post": 120},
        {"snr": 25, "gain": 1.05, "speed": 1.00, "pre": 160, "post": 80},
        {"snr": 20, "gain": 0.90, "speed": 0.96, "pre": 60, "post": 180},
        {"snr": 20, "gain": 1.10, "speed": 1.04, "pre": 180, "post": 60},
        {"snr": 15, "gain": 0.85, "speed": 0.98, "pre": 120, "post": 120},
        {"snr": 15, "gain": 1.00, "speed": 1.02, "pre": 240, "post": 160},
        {"snr": 12, "gain": 0.95, "speed": 1.00, "pre": 40, "post": 260},
        {"snr": 18, "gain": 1.08, "speed": 1.00, "pre": 260, "post": 40},
    ]
    aug_rows = []
    for row in real_rows:
        clean = load_audio(row["wav"])
        for idx, variant in enumerate(variants, 1):
            noise_row = noise_rows[(len(aug_rows) + idx) % len(noise_rows)]
            noise_path = noise_row["wav"]
            if noise_path not in noise_cache:
                noise_cache[noise_path] = load_audio(noise_path)
            wav = speed_perturb(clean, variant["speed"]) * variant["gain"]
            wav = pad_silence(wav, 16000, variant["pre"], variant["post"])
            noise = crop_or_tile_noise(noise_cache[noise_path], wav.shape[1], rng)
            wav = mix_with_snr(wav, noise, variant["snr"])
            key = f"{row['key']}_realpos_aug{idx:02d}"
            dst = aug_dir / f"{key}.wav"
            save_audio(dst, wav)
            aug_rows.append({
                "key": key,
                "wav": dst,
                "label_type": "positive",
                "text": KEYWORD,
                "duration": duration_sec(dst),
                "category": "rk3566_real_positive_augmented",
                "source_group": "rk3566_real_positive",
            })
    return aug_rows


def read_detected_rows(score_path: Path, continuous_metadata: Path) -> list[dict]:
    detected = []
    with score_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1] == "detected":
                detected.append(parts[0])
    meta_by_key = {row["key"]: row for row in read_jsonl(continuous_metadata)}
    return [
        normalize_row(meta_by_key[key], force_label="negative", category="continuous_false_alarm_hard_negative")
        for key in detected
        if key in meta_by_key
    ]


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


def split_base(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(seed)
    pos = [row for row in rows if row["label_type"] == "positive"]
    neg = [row for row in rows if row["label_type"] == "negative"]
    rng.shuffle(pos)
    rng.shuffle(neg)

    def split(items: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        n_train = int(len(items) * 0.85)
        n_dev = int(len(items) * 0.10)
        return items[:n_train], items[n_train:n_train + n_dev], items[n_train + n_dev:]

    p_train, p_dev, p_test = split(pos)
    n_train, n_dev, n_test = split(neg)
    return p_train + n_train, p_dev + n_dev, p_test + n_test


def duplicate_rows(rows: list[dict], repeat: int, prefix: str) -> list[dict]:
    out = []
    for copy_idx in range(repeat):
        for row in rows:
            cloned = dict(row)
            cloned["key"] = f"{row['key']}_{prefix}{copy_idx + 1:02d}"
            cloned["category"] = f"{row['category']}_oversampled"
            out.append(cloned)
    return out


def oversample_fractional(rows: list[dict], full_repeat: int, extra_ratio: float, rng: random.Random, prefix: str) -> list[dict]:
    out = duplicate_rows(rows, full_repeat, prefix)
    if extra_ratio <= 0:
        return out
    extra_count = int(round(len(rows) * extra_ratio))
    sampled = rows[:]
    rng.shuffle(sampled)
    for idx, row in enumerate(sampled[:extra_count], 1):
        cloned = dict(row)
        cloned["key"] = f"{row['key']}_{prefix}_extra{idx:04d}"
        cloned["category"] = f"{row['category']}_oversampled"
        out.append(cloned)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_manifest", default=r"E:\CodeWorking\Dataset\anju_xiaobao_aishell218_posneg_rknoise_20260509\manifest.jsonl")
    parser.add_argument("--hard_manifest", default=r"E:\CodeWorking\Dataset\anju_xiaobao_hard_negative_merged_aishell218_20260509\manifest.jsonl")
    parser.add_argument("--real_positive_train_list", default=r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\data\prepared_pretrain_hardneg_realpos_mined_20260509\eval_real_positive_train\data.list")
    parser.add_argument("--real_positive_holdout_list", default=r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\data\prepared_pretrain_hardneg_realpos_mined_20260509\eval_real_positive_holdout\data.list")
    parser.add_argument("--real_noise_manifest", default=r"E:\CodeWorking\Dataset\anju_xiaobao_kws_dataset_20260508\fixed_eval\negative_noise\manifest.jsonl")
    parser.add_argument("--continuous_score", default=r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\experiments\pretrain_hardneg_20260509_002\score_continuous_5s_best14.txt")
    parser.add_argument("--continuous_metadata", default=r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\data\eval_continuous_false_alarm_5s_20260509\metadata.jsonl")
    parser.add_argument("--output_dir", default=r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\data\prepared_pretrain_posbalanced_20260509")
    parser.add_argument("--dict_dir", default=r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\dict\pretrain_posbalanced_20260509")
    parser.add_argument("--tts_positive_full_repeat", type=int, default=1)
    parser.add_argument("--tts_positive_extra_ratio", type=float, default=0.5)
    parser.add_argument("--mined_train_count", type=int, default=90)
    parser.add_argument("--hard_pseudocut_train_count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    base_rows = [normalize_row(row) for row in read_jsonl(Path(args.base_manifest))]
    base_train, base_dev, base_test = split_base(base_rows, args.seed)
    base_train_pos = [row for row in base_train if row["label_type"] == "positive"]
    base_train_neg = [row for row in base_train if row["label_type"] == "negative"]

    real_train = [normalize_row(row, force_label="positive", category="rk3566_real_positive_train")
                  for row in read_jsonl(Path(args.real_positive_train_list))]
    real_holdout = [normalize_row(row, force_label="positive", category="rk3566_real_positive_holdout")
                    for row in read_jsonl(Path(args.real_positive_holdout_list))]
    real_noise = [normalize_row(row, force_label="negative", category="real_noise_train")
                  for row in read_jsonl(Path(args.real_noise_manifest))]
    rng.shuffle(real_noise)
    real_aug = make_real_positive_augments(real_train, real_noise, output_dir, args.seed)

    hard_rows = [normalize_row(row, force_label="negative") for row in read_jsonl(Path(args.hard_manifest))]
    hard_near = [row for row in hard_rows if row["category"] in {"hard_negative_near", "hard_negative_office_speech"}]
    hard_pseudocut = [row for row in hard_rows if row["category"] == "hard_negative_pseudocut_partial_keyword"]
    rng.shuffle(hard_pseudocut)

    mined = read_detected_rows(Path(args.continuous_score), Path(args.continuous_metadata))
    rng.shuffle(mined)
    mined_train = mined[:args.mined_train_count]
    mined_holdout = mined[args.mined_train_count:]

    train_rows = []
    train_rows.extend(oversample_fractional(
        base_train_pos,
        args.tts_positive_full_repeat,
        args.tts_positive_extra_ratio,
        rng,
        "posdup",
    ))
    train_rows.extend(base_train_neg)
    train_rows.extend(duplicate_rows(real_train, 4, "realdup"))
    train_rows.extend(real_aug)
    train_rows.extend(hard_near)
    train_rows.extend(hard_pseudocut[:args.hard_pseudocut_train_count])
    train_rows.extend(real_noise[:100])
    train_rows.extend(mined_train)
    rng.shuffle(train_rows)

    dev_start = args.hard_pseudocut_train_count
    dev_rows = base_dev + hard_pseudocut[dev_start:dev_start + 20] + real_noise[100:120]
    test_rows = base_test + hard_pseudocut[dev_start + 20:dev_start + 50] + real_noise[120:140]
    rng.shuffle(dev_rows)
    rng.shuffle(test_rows)

    write_split(output_dir / "train", train_rows)
    write_split(output_dir / "dev", dev_rows)
    write_split(output_dir / "test", test_rows)
    write_split(output_dir / "eval_real_positive_holdout", real_holdout)
    write_split(output_dir / "eval_real_positive_train", real_train)
    write_split(output_dir / "eval_false_alarm_holdout", mined_holdout)
    write_dict(Path(args.dict_dir))

    all_rows = train_rows + dev_rows + test_rows
    summary = {
        "total": len(all_rows),
        "positive": sum(row["label_type"] == "positive" for row in all_rows),
        "negative": sum(row["label_type"] == "negative" for row in all_rows),
        "splits": {},
        "by_category": {},
        "real_positive_train_count": len(real_train),
        "real_positive_augmented_count": len(real_aug),
        "real_positive_holdout_count": len(real_holdout),
        "mined_false_alarm_train_count": len(mined_train),
        "mined_false_alarm_holdout_count": len(mined_holdout),
        "tts_positive_full_repeat": args.tts_positive_full_repeat,
        "tts_positive_extra_ratio": args.tts_positive_extra_ratio,
        "hard_pseudocut_train_count": args.hard_pseudocut_train_count,
        "note": "Positive-balanced round: TTS positives use fractional oversampling, real positives are augmented, pseudocut hard negatives and mined continuous false alarms are included.",
    }
    for split, rows in {"train": train_rows, "dev": dev_rows, "test": test_rows}.items():
        summary["splits"][split] = {
            "total": len(rows),
            "positive": sum(row["label_type"] == "positive" for row in rows),
            "negative": sum(row["label_type"] == "negative" for row in rows),
        }
    for row in all_rows:
        summary["by_category"][row["category"]] = summary["by_category"].get(row["category"], 0) + 1
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
