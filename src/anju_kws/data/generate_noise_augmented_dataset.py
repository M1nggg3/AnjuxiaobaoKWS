import argparse
import json
import math
import random
import shutil
import wave
from pathlib import Path

import numpy as np


KEYWORD = "\u5b89\u5c45\u5c0f\u5b9d"


def read_manifest(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.getnframes()
        data = w.readframes(frames)
    if ch != 1 or sw != 2:
        raise ValueError(f"expected mono 16-bit wav: {path}")
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return audio, sr


def write_wav(path: Path, audio: np.ndarray, sr: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(clipped.tobytes())


def rms(audio):
    return math.sqrt(float(np.mean(np.square(audio.astype(np.float64)))) + 1e-12)


def random_noise_segment(noise_audio, target_len, rng):
    if len(noise_audio) >= target_len:
        start = rng.randint(0, len(noise_audio) - target_len)
        return noise_audio[start:start + target_len].copy()
    repeats = int(math.ceil(target_len / len(noise_audio)))
    return np.tile(noise_audio, repeats)[:target_len].copy()


def mix_with_snr(clean, noise, snr_db):
    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if noise_rms < 1e-6:
        return clean.copy()
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    mixed = clean + noise * (target_noise_rms / noise_rms)
    peak = np.max(np.abs(mixed)) if mixed.size else 0
    if peak > 32767:
        mixed = mixed * (32767 / peak)
    return mixed


def wav_duration_ms(path: Path):
    with wave.open(str(path), "rb") as w:
        return int(round(w.getnframes() / w.getframerate() * 1000))


def normalize_row_path(row):
    path = Path(row["path"]).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def copy_noise_segments(noise_manifest_path: Path, out_dir: Path):
    rows = read_manifest(noise_manifest_path)
    copied = []
    target_dir = out_dir / "wav" / "pure_noise"
    target_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows, 1):
        src = Path(row["path"]).resolve()
        dst = target_dir / src.name
        shutil.copy2(src, dst)
        copied.append({
            "index": idx,
            "path": str(dst),
            "text": "<filler>",
            "label_type": "negative",
            "category": "rk3566_environment_noise",
            "source_path": str(src),
            "source_remote_path": row.get("source_remote_path", ""),
            "sample_rate": 16000,
            "channels": 1,
            "sample_width_bytes": 2,
            "duration_ms": wav_duration_ms(dst),
        })
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive_manifest", required=True)
    parser.add_argument("--negative_manifest", required=True)
    parser.add_argument("--noise_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_positive", type=int, default=500)
    parser.add_argument("--num_negative", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--snr_db", default="5,10,15,20")
    parser.add_argument("--positive_snr_db", default=None)
    parser.add_argument("--negative_snr_db", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wav" / "positive_noisy").mkdir(parents=True, exist_ok=True)
    (out_dir / "wav" / "negative_noisy").mkdir(parents=True, exist_ok=True)

    pos_rows = read_manifest(Path(args.positive_manifest))
    neg_rows = read_manifest(Path(args.negative_manifest))
    noise_rows = read_manifest(Path(args.noise_manifest))
    default_snrs = [float(x.strip()) for x in args.snr_db.split(",") if x.strip()]
    positive_snrs = [float(x.strip()) for x in (args.positive_snr_db or args.snr_db).split(",") if x.strip()]
    negative_snrs = [float(x.strip()) for x in (args.negative_snr_db or args.snr_db).split(",") if x.strip()]

    noise_bank = []
    for row in noise_rows:
        audio, sr = read_wav(Path(row["path"]))
        if sr != 16000:
            raise ValueError(f"noise must be 16k: {row['path']}")
        noise_bank.append((row["sample_id"], Path(row["path"]), audio))

    pure_noise = copy_noise_segments(Path(args.noise_manifest), out_dir)

    selected_pos = pos_rows[:]
    selected_neg = neg_rows[:]
    rng.shuffle(selected_pos)
    rng.shuffle(selected_neg)
    selected_pos = selected_pos[:args.num_positive]
    selected_neg = selected_neg[:args.num_negative]

    positive_rows = []
    negative_rows = []
    idx = 1

    def augment(source_rows, label_type, target_subdir, prefix, snrs):
        nonlocal idx
        augmented = []
        for row in source_rows:
            src = normalize_row_path(row)
            clean, sr = read_wav(src)
            if sr != 16000:
                raise ValueError(f"source must be 16k: {src}")
            noise_id, noise_path, noise_audio = rng.choice(noise_bank)
            noise = random_noise_segment(noise_audio, len(clean), rng)
            snr = rng.choice(snrs)
            mixed = mix_with_snr(clean, noise, snr)
            dst = out_dir / "wav" / target_subdir / f"{prefix}_{idx:04d}.wav"
            write_wav(dst, mixed, sr=16000)
            text = KEYWORD if label_type == "positive" else row.get("text", "<filler>")
            augmented.append({
                "index": idx,
                "path": str(dst),
                "text": text,
                "label_type": label_type,
                "category": f"{label_type}_noise_augmented",
                "source_path": str(src),
                "source_text": row.get("text", text),
                "source_category": row.get("category", label_type),
                "noise_id": noise_id,
                "noise_path": str(noise_path),
                "snr_db": snr,
                "sample_rate": 16000,
                "channels": 1,
                "sample_width_bytes": 2,
                "duration_ms": wav_duration_ms(dst),
                "prompt_id": row.get("prompt_id", ""),
                "gender": row.get("gender") or row.get("prompt_gender", ""),
                "speed": row.get("speed", ""),
            })
            idx += 1
        return augmented

    positive_rows.extend(augment(selected_pos, "positive", "positive_noisy",
                                "anju_xiaobao_pos_noisy", positive_snrs))
    negative_rows.extend(augment(selected_neg, "negative", "negative_noisy",
                                "anju_xiaobao_neg_noisy", negative_snrs))

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in positive_rows + negative_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    positive_manifest_path = out_dir / "positive_noisy_manifest.jsonl"
    with positive_manifest_path.open("w", encoding="utf-8") as f:
        for row in positive_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    negative_manifest_path = out_dir / "negative_noisy_manifest.jsonl"
    with negative_manifest_path.open("w", encoding="utf-8") as f:
        for row in negative_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pure_noise_path = out_dir / "pure_noise_manifest.jsonl"
    with pure_noise_path.open("w", encoding="utf-8") as f:
        for row in pure_noise:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    combined_path = out_dir / "combined_negative_manifest.jsonl"
    with combined_path.open("w", encoding="utf-8") as f:
        for row in pure_noise:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in negative_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output_dir": str(out_dir),
        "positive_noisy": args.num_positive,
        "negative_noisy": args.num_negative,
        "pure_noise": len(pure_noise),
        "snr_db": default_snrs,
        "positive_snr_db": positive_snrs,
        "negative_snr_db": negative_snrs,
        "manifest": str(manifest_path),
        "positive_noisy_manifest": str(positive_manifest_path),
        "negative_noisy_manifest": str(negative_manifest_path),
        "pure_noise_manifest": str(pure_noise_path),
        "combined_negative_manifest": str(combined_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
