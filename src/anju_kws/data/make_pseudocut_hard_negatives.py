from __future__ import annotations

import argparse
import json
import random
import wave
from collections import OrderedDict
from pathlib import Path

import torch
import torchaudio


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def save_pseudocut(src: Path, dst: Path, keep_ratio: float) -> None:
    wav, sr = torchaudio.load(str(src), backend="soundfile")
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        sr = 16000
    keep = max(int(wav.shape[1] * keep_ratio), int(0.55 * sr))
    keep = min(keep, wav.shape[1])
    cut = wav[:, :keep].clone()
    fade_len = min(int(0.08 * sr), cut.shape[1] // 4)
    if fade_len > 8:
        fade = torch.linspace(1.0, 0.0, fade_len).unsqueeze(0)
        cut[:, -fade_len:] *= fade
    dst.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(dst), cut.clamp(-1.0, 1.0), sr, encoding="PCM_S", bits_per_sample=16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_manifest", default=r"E:\CodeWorking\Dataset\anju_xiaobao_aishell218_posneg_rknoise_20260509\manifest.jsonl")
    parser.add_argument("--output_dir", default=r"E:\CodeWorking\Dataset\anju_xiaobao_hard_negative_aishell218_pseudocut_rknoise_20260509")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--keep_ratio_min", type=float, default=0.58)
    parser.add_argument("--keep_ratio_max", type=float, default=0.72)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.base_manifest))
    positives = [row for row in rows if row.get("label_type") == "positive"]
    by_speaker: OrderedDict[str, list[dict]] = OrderedDict()
    for row in positives:
        speaker = row.get("prompt_speaker") or row.get("source_group") or row["sample_id"]
        by_speaker.setdefault(speaker, []).append(row)

    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wav"
    rng = random.Random(args.seed)
    out_rows = []
    for index, (speaker, speaker_rows) in enumerate(by_speaker.items(), 1):
        row = speaker_rows[0]
        src = Path(row["path"])
        keep_ratio = rng.uniform(args.keep_ratio_min, args.keep_ratio_max)
        sample_id = f"anju_xiaobao_hardneg_pseudocut_{index:04d}"
        dst = wav_dir / f"{sample_id}.wav"
        save_pseudocut(src, dst, keep_ratio)
        out_rows.append({
            "sample_id": sample_id,
            "path": str(dst),
            "source_path": str(src),
            "text": "partial_安居小宝",
            "label_type": "negative",
            "category": "hard_negative_pseudocut_partial_keyword",
            "source_group": "tts_positive_pseudocut_aishell218",
            "prompt_speaker": speaker,
            "keep_ratio": round(keep_ratio, 4),
            **wav_meta(dst),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "output_dir": str(out_dir),
        "source_manifest": args.base_manifest,
        "speaker_count": len(by_speaker),
        "hard_negative_count": len(out_rows),
        "category": "hard_negative_pseudocut_partial_keyword",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
