from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
import wave
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F_audio


KEYWORD = "安居小宝"

DEFAULT_POSITIVE_TEXTS = [
    "安居小宝。",
    "安居小宝。",
    "安居小宝。",
    "安居小宝！",
    "安居小宝。",
    "安居小宝。",
]

DEFAULT_POSITIVE_SPEEDS = [
    0.90,
    0.95,
    1.00,
    1.04,
    1.08,
    1.12,
]

DEFAULT_HARD_NEGATIVE_TEXTS = [
    "安静小宝。",
    "安居宝宝。",
    "你好小宝。",
    "小宝小宝。",
    "安居小贝。",
    "安居小白。",
    "安居小播。",
    "安居小帮。",
    "办公室今天有会议。",
    "请把资料发给我。",
]

HOMOPHONE_NEGATIVES = [
    "安居小包。",
    "安居小保。",
    "安居小饱。",
]

FARFIELD_PROFILES = {
    "near_0p5m": {"distance_m": 0.5, "attenuation": 0.80, "reverb": 0.04, "hf_loss": 0.04, "snr_db": [20, 25, 30]},
    "mid_1m": {"distance_m": 1.0, "attenuation": 0.48, "reverb": 0.10, "hf_loss": 0.12, "snr_db": [10, 15, 20]},
    "far_2m": {"distance_m": 2.0, "attenuation": 0.28, "reverb": 0.18, "hf_loss": 0.22, "snr_db": [5, 10, 15]},
    "far_3m": {"distance_m": 3.0, "attenuation": 0.18, "reverb": 0.28, "hf_loss": 0.30, "snr_db": [0, 5, 10]},
}


def add_cosyvoice_repo(cosyvoice_repo: Path) -> None:
    if not (cosyvoice_repo / "cosyvoice").exists():
        raise FileNotFoundError(f"CosyVoice repo not found: {cosyvoice_repo}")
    sys.path.insert(0, str(cosyvoice_repo))
    matcha = cosyvoice_repo / "third_party" / "Matcha-TTS"
    if matcha.exists():
        sys.path.insert(0, str(matcha))


def read_wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        return {
            "sample_rate": sample_rate,
            "channels": wf.getnchannels(),
            "sample_width_bytes": wf.getsampwidth(),
            "duration_sec": round(frames / sample_rate, 3),
            "duration_ms": round(frames / sample_rate * 1000),
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("<filler>", ""))


def parse_aishell3_text(text_with_phone: str) -> str:
    parts = text_with_phone.strip().split()
    return "".join(parts[0::2])


def parse_spk_info(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 4:
                rows[parts[0]] = {
                    "age_group": parts[1],
                    "gender": normalize_gender(parts[2]),
                    "gender_source": "aishell3_spk_info",
                    "accent": parts[3],
                }
    return rows


def normalize_gender(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {"m", "male", "man", "男", "男性"}:
        return "male"
    if text in {"f", "female", "woman", "女", "女性"}:
        return "female"
    return "unknown"


def duration_ok(path: Path, min_sec: float, max_sec: float) -> bool:
    try:
        duration = read_wav_info(path)["duration_sec"]
    except Exception:
        return False
    return min_sec <= duration <= max_sec


def prompt_quality(path: Path, min_sec: float, max_sec: float, estimate_gender: bool) -> dict | None:
    try:
        info = read_wav_info(path)
    except Exception:
        return None
    duration = float(info["duration_sec"])
    if not min_sec <= duration <= max_sec:
        return None
    try:
        wav = read_audio(path, 16000)
    except Exception:
        return None
    mono = wav.squeeze(0)
    rms = float(torch.sqrt(torch.mean(mono ** 2) + 1e-12).item())
    peak = float(mono.abs().max().item())
    silence_ratio = float((mono.abs() < 0.008).float().mean().item())
    if rms < 0.006 or peak < 0.03 or silence_ratio > 0.82:
        return None

    duration_penalty = abs(duration - 5.5) * 4.0
    silence_penalty = max(0.0, silence_ratio - 0.45) * 80.0
    low_rms_penalty = max(0.0, 0.025 - rms) * 500.0
    clipping_penalty = 15.0 if peak > 0.98 else 0.0
    score = max(0.0, 100.0 - duration_penalty - silence_penalty - low_rms_penalty - clipping_penalty)

    gender = "unknown"
    gender_source = "unknown"
    f0_median_hz = 0.0
    if estimate_gender:
        f0_median_hz = estimate_f0_median(mono.unsqueeze(0), 16000)
        gender = gender_from_f0(f0_median_hz)
        gender_source = "f0_estimated" if gender != "unknown" else "f0_unknown"

    return {
        **info,
        "prompt_quality": {
            "score": round(score, 3),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "silence_ratio": round(silence_ratio, 4),
            "f0_median_hz": round(f0_median_hz, 2),
        },
        "gender": gender,
        "gender_source": gender_source,
    }


def estimate_f0_median(wav: torch.Tensor, sample_rate: int) -> float:
    try:
        pitch = F_audio.detect_pitch_frequency(
            wav,
            sample_rate,
            frame_time=0.02,
            win_length=30,
            freq_low=70,
            freq_high=360,
        )
    except Exception:
        return 0.0
    voiced = pitch[(pitch >= 70) & (pitch <= 360)]
    if voiced.numel() < 5:
        return 0.0
    return float(voiced.median().item())


def gender_from_f0(f0_median_hz: float) -> str:
    if f0_median_hz <= 0:
        return "unknown"
    if f0_median_hz < 155:
        return "male"
    if f0_median_hz > 175:
        return "female"
    return "unknown"


def collect_aishell3_prompts(root: Path, min_sec: float, max_sec: float) -> list[dict]:
    spk_info = parse_spk_info(root / "spk-info.txt")
    prompts: list[dict] = []
    for split in ("train", "test"):
        content = root / split / "content.txt"
        wav_root = root / split / "wav"
        if not content.exists() or not wav_root.exists():
            continue
        with content.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if "\t" not in line:
                    continue
                wav_name, text_pairs = line.split("\t", 1)
                speaker = wav_name[:7]
                wav_path = wav_root / speaker / wav_name
                if not wav_path.exists():
                    continue
                text = parse_aishell3_text(text_pairs)
                if len(normalize_text(text)) < 4:
                    continue
                prompts.append({
                    "source_dataset": "data_aishell3",
                    "source_split": split,
                    "speaker": speaker,
                    "prompt_wav": str(wav_path),
                    "prompt_text": text,
                    **spk_info.get(speaker, {}),
                })
    return prompts


def collect_aishell_prompts(root: Path, min_sec: float, max_sec: float) -> list[dict]:
    transcript = root / "transcript" / "aishell_transcript_v0.8.txt"
    audio_root = root / "wav" / "wav"
    if not transcript.exists() or not audio_root.exists():
        return []

    stem_to_path: dict[str, Path] = {}
    for wav in audio_root.rglob("*.wav"):
        stem_to_path[wav.stem] = wav

    prompts: list[dict] = []
    with transcript.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            parts = raw.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            wav_path = stem_to_path.get(utt_id)
            if not wav_path:
                continue
            match = re.search(r"S\d{4}", utt_id)
            speaker = match.group(0) if match else wav_path.parent.name
            prompt_text = text.replace(" ", "")
            if len(normalize_text(prompt_text)) < 4:
                continue
            split = ""
            for part in wav_path.parts:
                if part in {"train", "dev", "test"}:
                    split = part
                    break
            prompts.append({
                "source_dataset": "data_aishell",
                "source_split": split,
                "speaker": speaker,
                "prompt_wav": str(wav_path),
                "prompt_text": prompt_text,
            })
    return prompts


def one_per_speaker(
    rows: list[dict],
    limit: int,
    seed: int,
    min_sec: float,
    max_sec: float,
    max_candidates_per_speaker: int,
) -> list[dict]:
    estimate_gender = all(row.get("source_dataset") != "data_aishell3" for row in rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["speaker"]].append(row)
    speaker_keys = sorted(grouped)
    if limit > 0:
        rng = random.Random(seed)
        rng.shuffle(speaker_keys)
        speaker_keys = speaker_keys[:limit]
    selected = []
    for speaker in speaker_keys:
        candidates = sorted(grouped[speaker], key=lambda x: x["prompt_wav"])
        best = None
        for row in candidates[:max_candidates_per_speaker]:
            wav_path = Path(row["prompt_wav"])
            quality = prompt_quality(wav_path, min_sec, max_sec, estimate_gender=estimate_gender)
            if quality is None:
                continue
            gender = normalize_gender(row.get("gender")) if row.get("gender") else quality["gender"]
            gender_source = row.get("gender_source") or quality["gender_source"]
            item = {
                **row,
                **quality,
                "gender": gender,
                "gender_source": gender_source,
            }
            if best is None or item["prompt_quality"]["score"] > best["prompt_quality"]["score"]:
                best = item
        if best is not None:
            selected.append(best)
    random.Random(seed).shuffle(selected)
    if limit <= 0:
        return selected
    return selected[:limit]


def gender_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"male": 0, "female": 0, "unknown": 0}
    for row in rows:
        counts[normalize_gender(row.get("gender"))] = counts.get(normalize_gender(row.get("gender")), 0) + 1
    return counts


def balance_gender(rows: list[dict], seed: int, allow_unknown: bool = False, max_per_gender: int = 0) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {"male": [], "female": [], "unknown": []}
    for row in rows:
        buckets[normalize_gender(row.get("gender"))].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda x: (-float(x.get("prompt_quality", {}).get("score", 0.0)), x["source_dataset"], x["speaker"]))

    male_count = len(buckets["male"])
    female_count = len(buckets["female"])
    if male_count and female_count:
        keep = min(male_count, female_count)
        if max_per_gender > 0:
            keep = min(keep, max_per_gender)
        selected = buckets["male"][:keep] + buckets["female"][:keep]
    else:
        selected = buckets["male"] + buckets["female"]
        if max_per_gender > 0:
            selected = selected[:max_per_gender]
    if allow_unknown:
        selected += buckets["unknown"]
    rng.shuffle(selected)
    return sorted(selected, key=lambda x: (x["source_dataset"], normalize_gender(x.get("gender")), x["speaker"]))


def resolve_noise_paths(noise_manifest: Path, dataset_root: Path) -> list[Path]:
    paths: list[Path] = []
    with noise_manifest.open("r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            row = json.loads(raw)
            candidates = []
            if row.get("path"):
                candidates.append(Path(row["path"]))
            if row.get("relative_path"):
                candidates.append(dataset_root / row["relative_path"])
            for candidate in candidates:
                if candidate.exists():
                    paths.append(candidate)
                    break
    if not paths:
        raise FileNotFoundError(f"No usable noise wav found from {noise_manifest}")
    return sorted(set(paths))


def lowpass_one_pole(wav: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return wav
    alpha = min(max(strength, 0.0), 0.9)
    out = wav.clone()
    for i in range(1, out.shape[1]):
        out[:, i] = (1 - alpha) * out[:, i] + alpha * out[:, i - 1]
    return out


def add_simple_room_response(wav: torch.Tensor, reverb: float, sample_rate: int = 16000) -> torch.Tensor:
    if reverb <= 0:
        return wav
    delays_ms = [35, 68, 112, 177]
    gains = [0.45, 0.30, 0.20, 0.12]
    out = wav.clone()
    for delay_ms, gain in zip(delays_ms, gains):
        delay = int(sample_rate * delay_ms / 1000)
        if delay >= wav.shape[1]:
            continue
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


def farfield_transform(
    clean: torch.Tensor,
    noise: torch.Tensor,
    profile: dict,
    snr_db: float,
    rng: random.Random,
) -> torch.Tensor:
    speech = clean * float(profile["attenuation"])
    speech = lowpass_one_pole(speech, float(profile["hf_loss"]))
    speech = add_simple_room_response(speech, float(profile["reverb"]))
    noise_seg = random_noise_segment(noise, speech.shape[1], rng)
    return mix_with_snr(speech, noise_seg, snr_db)


def build_generation_plan(
    prompts: list[dict],
    samples_per_speaker: int,
    negatives_per_speaker: int,
    include_negatives: bool,
    include_homophone_negatives: bool,
) -> list[dict]:
    negative_texts = DEFAULT_HARD_NEGATIVE_TEXTS[:]
    if include_homophone_negatives:
        negative_texts = HOMOPHONE_NEGATIVES + negative_texts
    plan = []
    for prompt_index, prompt in enumerate(prompts, 1):
        for sample_i in range(samples_per_speaker):
            text = DEFAULT_POSITIVE_TEXTS[sample_i % len(DEFAULT_POSITIVE_TEXTS)]
            speed = DEFAULT_POSITIVE_SPEEDS[sample_i % len(DEFAULT_POSITIVE_SPEEDS)]
            sample_id = f"{prompt['source_dataset']}_{prompt['speaker']}_pos_{sample_i + 1:02d}"
            plan.append({**prompt, "sample_id": sample_id, "label_type": "positive", "text": text, "speed": speed})
        if include_negatives:
            for neg_i in range(negatives_per_speaker):
                text = negative_texts[(prompt_index + neg_i - 1) % len(negative_texts)]
                sample_id = f"{prompt['source_dataset']}_{prompt['speaker']}_neg_{neg_i + 1:02d}"
                plan.append({**prompt, "sample_id": sample_id, "label_type": "negative", "text": text, "speed": 1.0})
    return plan


def load_cosyvoice(args):
    if args.cosyvoice_loader == "auto":
        add_cosyvoice_repo(Path(args.cosyvoice_repo))
        from cosyvoice.cli.cosyvoice import AutoModel

        return AutoModel(model_dir=args.model_dir, fp16=args.fp16)

    from anju_kws.tts.cosyvoice3_slim import SlimCosyVoice3

    return SlimCosyVoice3(
        model_dir=Path(args.model_dir),
        cosyvoice_repo=Path(args.cosyvoice_repo),
        fp16=args.fp16,
    )


def generate_tts(cosyvoice, row: dict, dst: Path, text_frontend: bool, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    chunks = []
    start = time.time()
    for output in cosyvoice.inference_zero_shot(
        row["text"],
        row["prompt_text"],
        row["prompt_wav"],
        stream=True,
        speed=float(row["speed"]),
        text_frontend=text_frontend,
    ):
        chunks.append(output["tts_speech"].detach().cpu())
    if not chunks:
        raise RuntimeError(f"CosyVoice generated no audio for {row['sample_id']}")
    speech = torch.cat(chunks, dim=1)
    if cosyvoice.sample_rate != 16000:
        speech = torchaudio.transforms.Resample(cosyvoice.sample_rate, 16000)(speech)
    write_audio(dst, speech, 16000)
    print(f"[tts] {row['sample_id']} {row['label_type']} {read_wav_info(dst)['duration_sec']}s {time.time() - start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default=r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data\anju_xiaobao_kws_dataset_20260508")
    parser.add_argument("--external_root", default="")
    parser.add_argument("--output_dir", default=r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data\anju_xiaobao_farfield_cosyvoice3_20260518")
    parser.add_argument("--cosyvoice_repo", default=r"E:\CodeWorking\Project\CosyVoice")
    parser.add_argument("--model_dir", default=r"E:\CodeWorking\Project\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B")
    parser.add_argument("--noise_manifest", default="")
    parser.add_argument("--max_aishell3_speakers", type=int, default=50)
    parser.add_argument("--max_aishell_speakers", type=int, default=50)
    parser.add_argument("--max_prompt_candidates_per_speaker", type=int, default=12)
    parser.add_argument("--samples_per_speaker", type=int, default=6)
    parser.add_argument("--negatives_per_speaker", type=int, default=2)
    parser.add_argument("--min_prompt_sec", type=float, default=3.0)
    parser.add_argument("--max_prompt_sec", type=float, default=10.0)
    parser.add_argument("--profiles", default="near_0p5m,mid_1m,far_2m")
    parser.add_argument("--stage", choices=["clean", "all"], default="all", help="clean generates only TTS clean wavs; all also creates far-field noisy wavs.")
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cosyvoice_loader", choices=["slim", "auto"], default="slim", help="Use slim loader by default to avoid slow Matcha/librosa imports on Windows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_negatives", action="store_true")
    parser.add_argument("--no_gender_balance", action="store_true")
    parser.add_argument("--include_unknown_gender", action="store_true")
    parser.add_argument("--max_speakers_per_gender", type=int, default=0, help="Cap balanced male/female prompt speakers. 0 keeps the largest balanced set.")
    parser.add_argument("--include_homophone_negatives", action="store_true")
    parser.add_argument("--text_frontend", action="store_true", help="Enable CosyVoice text frontend. Default is off for stable Windows batch generation.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    external_root = Path(args.external_root) if args.external_root else dataset_root / "external_reference"
    output_dir = Path(args.output_dir)
    noise_manifest = Path(args.noise_manifest) if args.noise_manifest else dataset_root / "manifests" / "real_office_noise_segments.jsonl"

    aishell3 = collect_aishell3_prompts(external_root / "data_aishell3", args.min_prompt_sec, args.max_prompt_sec)
    aishell = collect_aishell_prompts(external_root / "data_aishell", args.min_prompt_sec, args.max_prompt_sec)
    selected_unbalanced = (
        one_per_speaker(
            aishell3,
            args.max_aishell3_speakers,
            args.seed,
            args.min_prompt_sec,
            args.max_prompt_sec,
            args.max_prompt_candidates_per_speaker,
        )
        + one_per_speaker(
            aishell,
            args.max_aishell_speakers,
            args.seed + 1,
            args.min_prompt_sec,
            args.max_prompt_sec,
            args.max_prompt_candidates_per_speaker,
        )
    )
    selected = selected_unbalanced
    if not args.no_gender_balance:
        selected = balance_gender(
            selected_unbalanced,
            args.seed + 2,
            allow_unknown=args.include_unknown_gender,
            max_per_gender=args.max_speakers_per_gender,
        )
    selected = sorted(selected, key=lambda x: (x["source_dataset"], x["speaker"]))
    plan = build_generation_plan(
        selected,
        args.samples_per_speaker,
        args.negatives_per_speaker,
        include_negatives=not args.no_negatives,
        include_homophone_negatives=args.include_homophone_negatives,
    )

    profile_names = [x.strip() for x in args.profiles.split(",") if x.strip()]
    profiles = {name: FARFIELD_PROFILES[name] for name in profile_names}
    noise_paths = [] if args.stage == "clean" else resolve_noise_paths(noise_manifest, dataset_root)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "manifests" / "selected_prompts.jsonl", selected)
    write_jsonl(output_dir / "manifests" / "selected_prompts_unbalanced.jsonl", selected_unbalanced)
    write_jsonl(output_dir / "manifests" / "generation_plan.jsonl", plan)
    summary = {
        "output_dir": str(output_dir),
        "stage": args.stage,
        "selected_prompt_count": len(selected),
        "selected_prompt_gender_counts": gender_counts(selected),
        "unbalanced_prompt_count": len(selected_unbalanced),
        "unbalanced_prompt_gender_counts": gender_counts(selected_unbalanced),
        "generation_plan_count": len(plan),
        "aishell3_prompt_candidates": len(aishell3),
        "aishell_prompt_candidates": len(aishell),
        "noise_count": len(noise_paths),
        "profiles": profiles,
        "dry_run": args.dry_run,
        "samples_per_speaker": args.samples_per_speaker,
        "negatives_per_speaker": 0 if args.no_negatives else args.negatives_per_speaker,
        "max_prompt_candidates_per_speaker": args.max_prompt_candidates_per_speaker,
        "gender_balance": not args.no_gender_balance,
        "max_speakers_per_gender": args.max_speakers_per_gender,
        "include_unknown_gender": args.include_unknown_gender,
        "include_homophone_negatives": args.include_homophone_negatives,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    prompt_dir = output_dir / "prompt_voices"
    clean_dirs = {"positive": output_dir / "clean_positive", "negative": output_dir / "clean_negative"}
    far_dirs = {"positive": output_dir / "farfield_positive", "negative": output_dir / "farfield_negative"}
    dirs_to_create = [prompt_dir, *clean_dirs.values()]
    if args.stage != "clean":
        dirs_to_create.extend(far_dirs.values())
    for path in dirs_to_create:
        path.mkdir(parents=True, exist_ok=True)

    for row in selected:
        src = Path(row["prompt_wav"])
        dst = prompt_dir / f"{row['source_dataset']}_{row['speaker']}_{src.name}"
        if not dst.exists() or args.overwrite:
            shutil.copy2(src, dst)
        row["prompt_wav_copied"] = str(dst)

    cosyvoice = load_cosyvoice(args)
    clean_rows = []
    farfield_rows = []
    rng = random.Random(args.seed)
    noise_cache: dict[Path, torch.Tensor] = {}

    for idx, row in enumerate(plan, 1):
        label = row["label_type"]
        clean_path = clean_dirs[label] / f"{row['sample_id']}.wav"
        generate_tts(cosyvoice, row, clean_path, text_frontend=args.text_frontend, overwrite=args.overwrite)
        clean_meta = {**row, "path": str(clean_path), "source_group": "cosyvoice3_clean_tts", **read_wav_info(clean_path)}
        clean_rows.append(clean_meta)

        if args.stage == "clean":
            continue

        clean_audio = read_audio(clean_path)
        for profile_name, profile in profiles.items():
            for snr_db in profile["snr_db"]:
                noise_path = noise_paths[(idx + int(snr_db)) % len(noise_paths)]
                if noise_path not in noise_cache:
                    noise_cache[noise_path] = read_audio(noise_path)
                far_audio = farfield_transform(clean_audio, noise_cache[noise_path], profile, float(snr_db), rng)
                far_id = f"{row['sample_id']}_{profile_name}_snr{int(snr_db):02d}"
                far_path = far_dirs[label] / f"{far_id}.wav"
                write_audio(far_path, far_audio, 16000)
                farfield_rows.append({
                    **row,
                    "sample_id": far_id,
                    "path": str(far_path),
                    "source_path": str(clean_path),
                    "source_group": "cosyvoice3_farfield_tts_rk3566_noise",
                    "farfield_profile": profile_name,
                    "distance_m": profile["distance_m"],
                    "snr_db": snr_db,
                    "noise_path": str(noise_path),
                    **read_wav_info(far_path),
                })
        print(f"[farfield] {idx:04d}/{len(plan)} {row['sample_id']} -> {len(profiles)} profiles")

    positive_clean = [r for r in clean_rows if r["label_type"] == "positive"]
    negative_clean = [r for r in clean_rows if r["label_type"] == "negative"]
    positive_far = [r for r in farfield_rows if r["label_type"] == "positive"]
    negative_far = [r for r in farfield_rows if r["label_type"] == "negative"]

    write_jsonl(output_dir / "manifests" / "clean_positive.jsonl", positive_clean)
    write_jsonl(output_dir / "manifests" / "clean_negative.jsonl", negative_clean)
    write_jsonl(output_dir / "manifests" / "all_clean.jsonl", clean_rows)
    if args.stage == "clean":
        write_jsonl(output_dir / "manifest.jsonl", clean_rows)
    else:
        write_jsonl(output_dir / "manifests" / "farfield_positive.jsonl", positive_far)
        write_jsonl(output_dir / "manifests" / "farfield_negative.jsonl", negative_far)
        write_jsonl(output_dir / "manifests" / "all_farfield.jsonl", farfield_rows)
        write_jsonl(output_dir / "manifest.jsonl", farfield_rows)

    summary.update({
        "dry_run": False,
        "clean_positive_count": len(positive_clean),
        "clean_negative_count": len(negative_clean),
        "farfield_positive_count": len(positive_far),
        "farfield_negative_count": len(negative_far),
        "manifest": str(output_dir / "manifest.jsonl"),
    })
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
