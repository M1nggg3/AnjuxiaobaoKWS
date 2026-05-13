from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import wave
from pathlib import Path

import torch
import torchaudio


SNR_VALUES = [5.0, 10.0, 15.0, 20.0]
NEAR_NEGATIVES = [
    "\u5b89\u5c45\u5c0f\u5305",
    "\u5b89\u5c45\u5c0f\u4fdd",
    "\u5b89\u5c45\u5c0f\u9971",
    "\u5b89\u9759\u5c0f\u5b9d",
    "\u5b89\u5c45\u5b9d\u5b9d",
    "\u5c0f\u5b9d",
    "\u4f60\u597d\u5c0f\u5b9d",
    "\u5b89\u5c45\u5c0f\u8d1d",
]
NEAR_NEGATIVES_4CHAR = [
    "\u5b89\u5c45\u5c0f\u5305",
    "\u5b89\u5c45\u5c0f\u4fdd",
    "\u5b89\u5c45\u5c0f\u9971",
    "\u5b89\u9759\u5c0f\u5b9d",
    "\u5b89\u5c45\u5b9d\u5b9d",
    "\u5b89\u5c45\u5c0f\u8d1d",
]
OFFICE_NEGATIVES = [
    "\u4eca\u5929\u4e0b\u5348\u8981\u5f00\u4f1a",
    "\u8fd9\u4efd\u6587\u6863\u5148\u653e\u684c\u9762",
    "\u6211\u5728\u529e\u516c\u5ba4\u5904\u7406\u4e8b\u60c5",
    "\u7a0d\u540e\u518d\u786e\u8ba4\u8fd9\u4e2a\u95ee\u9898",
    "\u660e\u5929\u4e0a\u5348\u518d\u8054\u7cfb",
    "\u7b49\u4e00\u4e0b\u6211\u5148\u770b\u4e00\u773c",
    "\u8fd9\u4e2a\u65b9\u6848\u8fd8\u8981\u518d\u8bc4\u4f30",
    "\u4f60\u628a\u8d44\u6599\u53d1\u7ed9\u6211",
]


def add_cosyvoice_repo(cosyvoice_repo: Path) -> None:
    if not (cosyvoice_repo / "cosyvoice").exists():
        raise FileNotFoundError(f"CosyVoice repo not found: {cosyvoice_repo}")
    sys.path.insert(0, str(cosyvoice_repo))
    matcha = cosyvoice_repo / "third_party" / "Matcha-TTS"
    if matcha.exists():
        sys.path.insert(0, str(matcha))


def wav_meta(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
        return {
            "duration_sec": frames / sr,
            "duration_ms": round(frames / sr * 1000),
            "sample_rate": sr,
            "channels": wf.getnchannels(),
            "sample_width_bytes": wf.getsampwidth(),
        }


def save_16k_mono(speech: torch.Tensor, sample_rate: int, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    speech = speech.detach().cpu()
    if speech.ndim == 1:
        speech = speech.unsqueeze(0)
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        speech = torchaudio.transforms.Resample(sample_rate, 16000)(speech)
    speech = speech.clamp(-1.0, 1.0)
    torchaudio.save(str(dst), speech, 16000, encoding="PCM_S", bits_per_sample=16)


def load_audio_mono(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path), backend="soundfile")
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
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


def read_prompts(path: Path) -> list[dict]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_manifest", default=r"E:\CodeWorking\Dataset\anju_xiaobao_aishell218_posneg_rknoise_20260509\selected_prompts.jsonl")
    parser.add_argument("--noise_dir", default=r"E:\CodeWorking\Dataset\anju_xiaobao_kws_dataset_20260508\real_recordings\office_noise\rk3566_noise_segments")
    parser.add_argument("--output_dir", default=r"E:\CodeWorking\Dataset\anju_xiaobao_hard_negative_aishell218_rknoise_20260509")
    parser.add_argument("--model_dir", default=r"D:\models\CosyVoice2-0.5B")
    parser.add_argument("--cosyvoice_repo", default=r"D:\codeWorking\TTS\CosyVoice")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--max_prompts", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--near_only", action="store_true", help="Use only short near-word hard negatives to avoid long-form TTS stalls.")
    parser.add_argument("--near_4char_only", action="store_true", help="Use only four-character near negatives; this avoids very short TTS stalls.")
    parser.add_argument("--samples_per_prompt", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    clean_dir = out_dir / "clean_wav"
    noisy_dir = out_dir / "noisy" / "wav"
    prompt_copy_dir = out_dir / "prompt_voices"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    prompt_copy_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(Path(args.prompt_manifest))[:args.max_prompts]
    noise_paths = sorted(Path(args.noise_dir).rglob("*.wav"))
    if not noise_paths:
        raise FileNotFoundError(args.noise_dir)

    add_cosyvoice_repo(Path(args.cosyvoice_repo))
    from cosyvoice.cli.cosyvoice import AutoModel

    cosyvoice = AutoModel(model_dir=args.model_dir, fp16=args.fp16)
    rng = random.Random(args.seed)
    clean_rows = []
    noisy_rows = []
    noise_cache: dict[Path, torch.Tensor] = {}
    text_pools = [NEAR_NEGATIVES, OFFICE_NEGATIVES]

    total = len(prompts) * args.samples_per_prompt
    index = 0
    for prompt_index, prompt in enumerate(prompts, 1):
        prompt_src = Path(prompt["prompt_wav"])
        prompt_dst = prompt_copy_dir / prompt_src.name
        if not prompt_dst.exists() or args.overwrite:
            shutil.copy2(prompt_src, prompt_dst)
        for kind, pool in enumerate(text_pools[:args.samples_per_prompt]):
            index += 1
            if args.near_only:
                pool = NEAR_NEGATIVES_4CHAR if args.near_4char_only else NEAR_NEGATIVES
                text = pool[(prompt_index - 1 + kind * (len(pool) // 2)) % len(pool)]
            else:
                text = pool[(prompt_index - 1) % len(pool)]
            sample_id = f"anju_xiaobao_hardneg_{index:04d}"
            clean_path = clean_dir / f"{sample_id}.wav"
            if not clean_path.exists() or args.overwrite:
                generated = False
                for output in cosyvoice.inference_zero_shot(
                    text,
                    prompt["prompt_text"],
                    str(prompt_dst),
                    stream=False,
                    speed=1.0 if kind == 0 else 1.05,
                ):
                    save_16k_mono(output["tts_speech"], cosyvoice.sample_rate, clean_path)
                    generated = True
                    break
                if not generated:
                    raise RuntimeError(f"no audio generated for {sample_id}")

            clean = load_audio_mono(clean_path)
            noise_path = noise_paths[(index - 1) % len(noise_paths)]
            if noise_path not in noise_cache:
                noise_cache[noise_path] = load_audio_mono(noise_path)
            snr = SNR_VALUES[(index - 1) % len(SNR_VALUES)]
            noise = crop_or_tile_noise(noise_cache[noise_path], clean.shape[1], rng)
            mixed = mix_with_snr(clean, noise, snr)
            noisy_path = noisy_dir / f"{sample_id}_snr{int(snr):02d}.wav"
            torchaudio.save(str(noisy_path), mixed, 16000, encoding="PCM_S", bits_per_sample=16)

            base = {
                "sample_id": sample_id,
                "text": text,
                "label_type": "negative",
                "category": "hard_negative_near" if args.near_only or kind == 0 else "hard_negative_office_speech",
                "source_group": "tts_hard_negative_aishell218",
                "prompt_text": prompt["prompt_text"],
                "prompt_wav": str(prompt_dst),
                "prompt_speaker": prompt["speaker"],
                "prompt_gender": prompt.get("gender", ""),
                "prompt_age_group": prompt.get("age_group", ""),
                "prompt_accent": prompt.get("accent", ""),
            }
            clean_rows.append({**base, "path": str(clean_path), **wav_meta(clean_path)})
            noisy_rows.append({
                **base,
                "sample_id": noisy_path.stem,
                "path": str(noisy_path),
                "source_path": str(clean_path),
                "noise_path": str(noise_path),
                "snr_db": snr,
                **wav_meta(noisy_path),
            })
            print(f"[{index:04d}/{total}] {text} -> {noisy_path.name}")

    write_jsonl(out_dir / "clean_manifest.jsonl", clean_rows)
    write_jsonl(out_dir / "manifest.jsonl", noisy_rows)
    summary = {
        "output_dir": str(out_dir),
        "prompt_count": len(prompts),
        "clean_count": len(clean_rows),
        "noisy_count": len(noisy_rows),
        "near_negative_texts": NEAR_NEGATIVES_4CHAR if args.near_4char_only else NEAR_NEGATIVES,
        "office_negative_texts": OFFICE_NEGATIVES,
        "noise_count": len(noise_paths),
        "snr_values": SNR_VALUES,
    }
    (out_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
