import argparse
import json
import wave
from pathlib import Path

import numpy as np


def read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        frames = w.getnframes()
        data = w.readframes(frames)
    if ch != 1 or sw != 2:
        raise ValueError(f"expected mono 16-bit wav: {path}")
    return np.frombuffer(data, dtype=np.int16), sr


def write_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.astype(np.int16).tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sample_id_prefix", required=True)
    parser.add_argument("--source_remote_path", default="")
    parser.add_argument("--segment_sec", type=float, default=5.0)
    parser.add_argument("--min_last_sec", type=float, default=2.0)
    args = parser.parse_args()

    input_wav = Path(args.input_wav).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    audio, sr = read_wav(input_wav)
    if sr != 16000:
        raise ValueError(f"noise wav must be 16k: {input_wav}")

    seg_len = int(args.segment_sec * sr)
    min_last = int(args.min_last_sec * sr)
    rows = []
    idx = 0
    for start in range(0, len(audio), seg_len):
        seg = audio[start:start + seg_len]
        if len(seg) < min_last:
            break
        sample_id = f"{args.sample_id_prefix}_seg{idx:04d}"
        out_path = output_dir / f"{sample_id}.wav"
        write_wav(out_path, seg, sr)
        rows.append({
            "sample_id": sample_id,
            "label_type": "negative",
            "category": "rk3566_environment_noise",
            "transcript": "<filler>",
            "relative_path": out_path.as_posix(),
            "path": str(out_path),
            "sample_rate": sr,
            "channels": 1,
            "sample_width_bytes": 2,
            "duration_sec": round(len(seg) / sr, 3),
            "source_path": str(input_wav),
            "source_remote_path": args.source_remote_path,
        })
        idx += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_wav": str(input_wav),
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "segments": len(rows),
        "segment_sec": args.segment_sec,
        "total_duration_sec": round(len(audio) / sr, 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
