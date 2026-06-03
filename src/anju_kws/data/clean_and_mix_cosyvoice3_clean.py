from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import wave
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


FARFIELD_PROFILES = {
    "near_0p5m": {"distance_m": 0.5, "attenuation": 0.80, "reverb": 0.04, "hf_loss": 0.04, "snr_db": [20, 25, 30]},
    "mid_1m": {"distance_m": 1.0, "attenuation": 0.48, "reverb": 0.10, "hf_loss": 0.12, "snr_db": [10, 15, 20]},
    "far_2m": {"distance_m": 2.0, "attenuation": 0.28, "reverb": 0.18, "hf_loss": 0.22, "snr_db": [5, 10, 15]},
}


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        return {
            "sample_rate": sr,
            "channels": wf.getnchannels(),
            "sample_width_bytes": wf.getsampwidth(),
            "duration_sec": round(frames / sr, 3),
            "duration_ms": round(frames / sr * 1000),
        }


def read_audio(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.float()
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    return wav.clamp(-1.0, 1.0)


def write_audio(path: Path, wav: torch.Tensor, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    peak = wav.abs().max().item()
    if peak > 0.98:
        wav = wav / peak * 0.98
    torchaudio.save(str(path), wav.clamp(-1.0, 1.0), sample_rate, encoding="PCM_S", bits_per_sample=16)


def resolve_clean_path(row: dict, input_dir: Path) -> Path:
    subdir = "clean_positive" if row.get("label_type") == "positive" else "clean_negative"
    candidate = input_dir / subdir / Path(row["path"]).name
    if candidate.exists():
        return candidate
    raw = Path(row["path"])
    if raw.exists():
        return raw
    raise FileNotFoundError(f"Cannot resolve clean wav for {row.get('sample_id')}: {row.get('path')}")


def resolve_noise_paths(noise_manifest: Path, noise_dataset_root: Path) -> list[Path]:
    paths = []
    for row in read_jsonl(noise_manifest):
        candidates = []
        if row.get("relative_path"):
            candidates.append(noise_dataset_root / row["relative_path"])
        if row.get("path"):
            candidates.append(Path(row["path"]))
        for candidate in candidates:
            if candidate.exists():
                paths.append(candidate)
                break
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No usable noise wav found from {noise_manifest}")
    return paths


def lowpass_one_pole(wav: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return wav
    strength = min(max(strength, 0.0), 0.9)
    if wav.shape[1] < 3:
        return wav

    kernel_size = max(3, int(round(3 + strength * 80)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, wav.shape[1] if wav.shape[1] % 2 == 1 else wav.shape[1] - 1)
    if kernel_size < 3:
        return wav

    pad = kernel_size // 2
    window = torch.hann_window(kernel_size, periodic=False, dtype=wav.dtype, device=wav.device)
    window = window / window.sum().clamp_min(1e-12)
    kernel = window.view(1, 1, kernel_size).repeat(wav.shape[0], 1, 1)
    padded = F.pad(wav.unsqueeze(0), (pad, pad), mode="reflect")
    smoothed = F.conv1d(padded, kernel, groups=wav.shape[0]).squeeze(0)
    return ((1.0 - strength) * wav + strength * smoothed).clamp(-1.0, 1.0)


def add_simple_room_response(wav: torch.Tensor, reverb: float, sample_rate: int = 16000) -> torch.Tensor:
    if reverb <= 0:
        return wav
    out = wav.clone()
    for delay_ms, gain in zip([35, 68, 112, 177], [0.45, 0.30, 0.20, 0.12]):
        delay = int(sample_rate * delay_ms / 1000)
        if delay < wav.shape[1]:
            out[:, delay:] += wav[:, :-delay] * gain * reverb
    peak = out.abs().max().item()
    if peak > 0.98:
        out = out / peak * 0.98
    return out


def random_noise_segment(noise: torch.Tensor, target_len: int, rng: random.Random) -> torch.Tensor:
    if noise.shape[1] >= target_len:
        start = rng.randint(0, noise.shape[1] - target_len)
        return noise[:, start:start + target_len]
    repeat = math.ceil(target_len / noise.shape[1])
    return noise.repeat(1, repeat)[:, :target_len]


def mix_with_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    clean_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-12)
    noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-12)
    if noise_rms.item() < 1e-8:
        return clean
    target_noise_rms = clean_rms / (10 ** (snr_db / 20))
    mixed = clean + noise * (target_noise_rms / noise_rms)
    peak = mixed.abs().max().item()
    if peak > 0.98:
        mixed = mixed / peak * 0.98
    return mixed.clamp(-1.0, 1.0)


def farfield_transform(clean: torch.Tensor, noise: torch.Tensor, profile: dict, snr_db: float, rng: random.Random) -> torch.Tensor:
    speech = clean * float(profile["attenuation"])
    speech = lowpass_one_pole(speech, float(profile["hf_loss"]))
    speech = add_simple_room_response(speech, float(profile["reverb"]))
    noise_seg = random_noise_segment(noise, speech.shape[1], rng)
    return mix_with_snr(speech, noise_seg, snr_db)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_clean_dir", required=True)
    parser.add_argument("--noise_dataset_root", required=True)
    parser.add_argument("--noise_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_positive_sec", type=float, default=0.8)
    parser.add_argument("--min_negative_sec", type=float, default=0.8)
    parser.add_argument("--max_positive_sec", type=float, default=4.5)
    parser.add_argument("--max_negative_sec", type=float, default=8.0)
    parser.add_argument("--profiles", default="near_0p5m,mid_1m,far_2m")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--copy_clean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_clean_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_dir / "manifests" / "all_clean.jsonl")
    profiles = {name: FARFIELD_PROFILES[name] for name in args.profiles.split(",") if name}
    noise_paths = resolve_noise_paths(Path(args.noise_manifest), Path(args.noise_dataset_root))

    kept_clean_rows = []
    rejected_rows = []
    clean_dirs = {"positive": output_dir / "clean_positive", "negative": output_dir / "clean_negative"}

    for row in rows:
        src = resolve_clean_path(row, input_dir)
        info = wav_info(src)
        label = row["label_type"]
        min_sec = args.min_positive_sec if label == "positive" else args.min_negative_sec
        max_sec = args.max_positive_sec if label == "positive" else args.max_negative_sec
        if not (min_sec <= float(info["duration_sec"]) <= max_sec):
            rejected_rows.append({**row, "local_source_path": str(src), "reject_reason": "duration_out_of_range", **info})
            continue
        if info["sample_rate"] != 16000 or info["channels"] != 1 or info["sample_width_bytes"] != 2:
            rejected_rows.append({**row, "local_source_path": str(src), "reject_reason": "bad_wav_format", **info})
            continue

        if args.copy_clean:
            dst = clean_dirs[label] / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or args.overwrite:
                shutil.copy2(src, dst)
            clean_path = dst
        else:
            clean_path = src
        kept_clean_rows.append({**row, "path": str(clean_path), "local_source_path": str(src), "source_group": "cosyvoice3_clean_tts_filtered", **info})

    rng = random.Random(args.seed)
    noise_cache: dict[Path, torch.Tensor] = {}
    far_dirs = {"positive": output_dir / "farfield_positive", "negative": output_dir / "farfield_negative"}
    farfield_rows = []
    for idx, row in enumerate(kept_clean_rows, 1):
        clean_path = Path(row["path"])
        clean = read_audio(clean_path)
        label = row["label_type"]
        for profile_name, profile in profiles.items():
            for snr_db in profile["snr_db"]:
                noise_path = noise_paths[(idx + int(snr_db)) % len(noise_paths)]
                if noise_path not in noise_cache:
                    noise_cache[noise_path] = read_audio(noise_path)
                far_audio = farfield_transform(clean, noise_cache[noise_path], profile, float(snr_db), rng)
                far_id = f"{row['sample_id']}_{profile_name}_snr{int(snr_db):02d}"
                far_path = far_dirs[label] / f"{far_id}.wav"
                write_audio(far_path, far_audio, 16000)
                farfield_rows.append({
                    **row,
                    "sample_id": far_id,
                    "path": str(far_path),
                    "source_path": str(clean_path),
                    "source_group": "cosyvoice3_purewake_farfield_rk3566_noise",
                    "farfield_profile": profile_name,
                    "distance_m": profile["distance_m"],
                    "snr_db": snr_db,
                    "noise_path": str(noise_path),
                    **wav_info(far_path),
                })
        if idx % 100 == 0:
            print(f"[mix] {idx}/{len(kept_clean_rows)} clean rows -> {len(farfield_rows)} farfield rows", flush=True)

    clean_pos = [r for r in kept_clean_rows if r["label_type"] == "positive"]
    clean_neg = [r for r in kept_clean_rows if r["label_type"] == "negative"]
    far_pos = [r for r in farfield_rows if r["label_type"] == "positive"]
    far_neg = [r for r in farfield_rows if r["label_type"] == "negative"]

    write_jsonl(output_dir / "manifests" / "clean_filtered.jsonl", kept_clean_rows)
    write_jsonl(output_dir / "manifests" / "clean_positive_filtered.jsonl", clean_pos)
    write_jsonl(output_dir / "manifests" / "clean_negative_filtered.jsonl", clean_neg)
    write_jsonl(output_dir / "manifests" / "rejected_clean.jsonl", rejected_rows)
    write_jsonl(output_dir / "manifests" / "farfield_positive.jsonl", far_pos)
    write_jsonl(output_dir / "manifests" / "farfield_negative.jsonl", far_neg)
    write_jsonl(output_dir / "manifests" / "all_farfield.jsonl", farfield_rows)
    write_jsonl(output_dir / "manifest.jsonl", farfield_rows)

    summary = {
        "input_clean_dir": str(input_dir),
        "output_dir": str(output_dir),
        "noise_manifest": str(Path(args.noise_manifest)),
        "noise_count": len(noise_paths),
        "input_clean_count": len(rows),
        "kept_clean_count": len(kept_clean_rows),
        "rejected_clean_count": len(rejected_rows),
        "clean_positive_count": len(clean_pos),
        "clean_negative_count": len(clean_neg),
        "farfield_positive_count": len(far_pos),
        "farfield_negative_count": len(far_neg),
        "farfield_total_count": len(farfield_rows),
        "min_positive_sec": args.min_positive_sec,
        "min_negative_sec": args.min_negative_sec,
        "profiles": profiles,
        "manifest": str(output_dir / "manifest.jsonl"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
