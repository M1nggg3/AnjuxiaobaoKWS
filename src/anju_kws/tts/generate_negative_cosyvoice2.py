from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path

import torch
import torchaudio

from anju_kws.tts.generate_negative_text_plan import (
    KEYWORD,
    KEYWORD_PINYIN,
    pinyin_signature,
    reject_reason,
)


def add_cosyvoice_repo(cosyvoice_repo: Path) -> None:
    if not (cosyvoice_repo / "cosyvoice").exists():
        raise FileNotFoundError(f"CosyVoice repo not found: {cosyvoice_repo}")
    sys.path.insert(0, str(cosyvoice_repo))
    matcha = cosyvoice_repo / "third_party" / "Matcha-TTS"
    if matcha.exists():
        sys.path.insert(0, str(matcha))


def audio_meta(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
        return {
            "duration_ms": round(frames / sr * 1000),
            "sample_rate": sr,
            "channels": wf.getnchannels(),
            "sample_width_bytes": wf.getsampwidth(),
        }


def silence_edges_ms(wav_path: Path, threshold: float = 0.01) -> tuple[int, int]:
    speech, sr = torchaudio.load(str(wav_path), backend="soundfile")
    speech = speech.mean(dim=0)
    if speech.numel() == 0:
        return 0, 0
    peak = speech.abs().max().item()
    if peak <= 1e-8:
        return round(speech.numel() / sr * 1000), round(speech.numel() / sr * 1000)
    voiced = speech.abs() >= peak * threshold
    idx = voiced.nonzero(as_tuple=False).flatten()
    if idx.numel() == 0:
        dur = round(speech.numel() / sr * 1000)
        return dur, dur
    leading = round(idx[0].item() / sr * 1000)
    trailing = round((speech.numel() - 1 - idx[-1].item()) / sr * 1000)
    return leading, trailing


def save_16k_mono(speech: torch.Tensor, sample_rate: int, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    speech = speech.detach().cpu()
    if speech.ndim == 1:
        speech = speech.unsqueeze(0)
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(speech)
    speech = speech.clamp(-1.0, 1.0)
    torchaudio.save(str(dst), speech, 16000, encoding="PCM_S", bits_per_sample=16)


def load_plan(plan_path: Path) -> list[dict]:
    rows = []
    with plan_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            reason = reject_reason(row["text"])
            if reason is not None:
                raise ValueError(f"unsafe text in plan: {row['text']} ({reason})")
            if tuple(row.get("pinyin", "").split()) == KEYWORD_PINYIN:
                raise ValueError(f"same pinyin as keyword in plan: {row['text']}")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_plan", default=r"E:\CodeWorking\Dataset\anju_xiaobao_negative_cosyvoice2_1000_gpu\text_plan.jsonl")
    parser.add_argument("--output_dir", default=r"E:\CodeWorking\Dataset\anju_xiaobao_negative_cosyvoice2_1000_gpu")
    parser.add_argument("--model_dir", default=r"D:\models\CosyVoice2-0.5B")
    parser.add_argument("--cosyvoice_repo", default=r"D:\codeWorking\TTS\CosyVoice")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wav"
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists, pass --overwrite to regenerate")
    out_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    add_cosyvoice_repo(Path(args.cosyvoice_repo))
    from cosyvoice.cli.cosyvoice import AutoModel

    rows = load_plan(Path(args.text_plan))
    cosyvoice = AutoModel(model_dir=args.model_dir, fp16=args.fp16)
    cuda_available = torch.cuda.is_available()
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else ""

    manifest_rows = []
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for i, row in enumerate(rows, 1):
            dst = wav_dir / f"anju_xiaobao_negative_{i:04d}.wav"
            generated = False
            for output in cosyvoice.inference_zero_shot(
                row["text"],
                row["prompt_text"],
                row["prompt_wav"],
                stream=False,
                speed=float(row["speed"]),
            ):
                save_16k_mono(output["tts_speech"], cosyvoice.sample_rate, dst)
                generated = True
                break
            if not generated:
                raise RuntimeError(f"CosyVoice generated no audio for row {i}: {row['text']}")
            meta = audio_meta(dst)
            leading_ms, trailing_ms = silence_edges_ms(dst)
            record = {
                "index": i,
                "path": str(dst),
                "text": row["text"],
                "label_type": "negative",
                "category": row["category"],
                "pinyin": " ".join(pinyin_signature(row["text"])),
                "keyword": KEYWORD,
                "keyword_pinyin": " ".join(KEYWORD_PINYIN),
                "homophone_filter_passed": True,
                "prompt_id": row["prompt_id"],
                "gender": row.get("prompt_gender", ""),
                "speed": float(row["speed"]),
                "duration_ms": meta["duration_ms"],
                "sample_rate": meta["sample_rate"],
                "channels": meta["channels"],
                "sample_width_bytes": meta["sample_width_bytes"],
                "prompt_text": row["prompt_text"],
                "prompt_wav": row["prompt_wav"],
                "prompt_speaker": row.get("prompt_speaker", ""),
                "prompt_age_group": row.get("prompt_age_group", ""),
                "prompt_accent": row.get("prompt_accent", ""),
                "model_dir": args.model_dir,
                "cosyvoice_repo": args.cosyvoice_repo,
                "cuda_available": cuda_available,
                "cuda_device": cuda_device,
                "fp16": bool(args.fp16),
                "leading_ms": leading_ms,
                "trailing_ms": trailing_ms,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            manifest_rows.append(record)
            print(f"[{i:04d}/{len(rows)}] {dst.name} category={row['category']} prompt={row['prompt_id']} speed={row['speed']}")

    category_counts = {}
    for row in manifest_rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    summary = {
        "output_dir": str(out_dir),
        "count": len(manifest_rows),
        "keyword": KEYWORD,
        "keyword_pinyin": " ".join(KEYWORD_PINYIN),
        "category_counts": category_counts,
        "model_dir": args.model_dir,
        "cosyvoice_repo": args.cosyvoice_repo,
        "cuda_available": cuda_available,
        "cuda_device": cuda_device,
        "fp16": bool(args.fp16),
        "manifest": str(manifest_path),
    }
    (out_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
