from __future__ import annotations

import argparse
import csv
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np

from anju_kws.eval.replay_android_streaming import (
    DEFAULT_SILENCE_CHUNKS_BEFORE_RESET,
    DEFAULT_SOFT_RESET_INTERVAL_CHUNKS,
    DEFAULT_SPEECH_PEAK_THRESHOLD,
    DEFAULT_SPEECH_RMS_THRESHOLD,
    NativeEquivalentReplay,
)


SAMPLE_RATE = 16000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_wav_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or sample_width != 2 or sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"Expected 16kHz mono PCM16 WAV, got sr={sample_rate}, channels={channels}, sample_width={sample_width}: {path}"
        )
    return np.frombuffer(frames, dtype="<i2").copy()


def load_runtime_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        "threshold": float(config.get("threshold_initial", 0.7)),
        "speech_rms_threshold": float(config.get("speech_rms_threshold", DEFAULT_SPEECH_RMS_THRESHOLD)),
        "speech_peak_threshold": int(config.get("speech_peak_threshold", DEFAULT_SPEECH_PEAK_THRESHOLD)),
        "silence_chunks_before_reset": int(
            config.get("silence_chunks_before_reset", DEFAULT_SILENCE_CHUNKS_BEFORE_RESET)
        ),
        "soft_reset_interval_chunks": int(
            config.get("soft_reset_interval_chunks", DEFAULT_SOFT_RESET_INTERVAL_CHUNKS)
        ),
    }


def score_window(
    runner: NativeEquivalentReplay,
    pcm: np.ndarray,
    source: str,
) -> dict[str, Any]:
    result = runner.replay(pcm, source)
    max_timeline = max((float(row["score"]) for row in result["timeline"]), default=0.0)
    max_event = max((float(row["score"]) for row in result["events"]), default=0.0)
    return {
        "max_score": max(max_timeline, max_event),
        "event_count": len(result["events"]),
        "timeline_rows": len(result["timeline"]),
    }


def iter_windows(pcm: np.ndarray, window_sec: float, hop_sec: float) -> list[tuple[int, int]]:
    window = int(round(window_sec * SAMPLE_RATE))
    hop = int(round(hop_sec * SAMPLE_RATE))
    if pcm.size <= window:
        return [(0, pcm.size)]
    starts = list(range(0, pcm.size - window + 1, hop))
    last_start = pcm.size - window
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window) for start in starts]


def evaluate_file(
    wav_path: Path,
    key: str,
    model_path: Path,
    runtime: dict[str, Any],
    provider: str,
    window_sec: float,
    hop_sec: float,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pcm = read_wav_pcm16(wav_path)
    windows = iter_windows(pcm, window_sec, hop_sec)
    window_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for index, (start, end) in enumerate(windows, start=1):
        runner = NativeEquivalentReplay(
            model_path=model_path,
            threshold=threshold,
            provider=provider,
            speech_rms_threshold=runtime["speech_rms_threshold"],
            speech_peak_threshold=runtime["speech_peak_threshold"],
            silence_chunks_before_reset=runtime["silence_chunks_before_reset"],
            soft_reset_interval_chunks=runtime["soft_reset_interval_chunks"],
        )
        score = score_window(runner, pcm[start:end], f"{key}_win{index:03d}")
        row = {
            "key": key,
            "wav": str(wav_path),
            "window_index": index,
            "start_sec": round(start / SAMPLE_RATE, 3),
            "end_sec": round(end / SAMPLE_RATE, 3),
            "duration_sec": round((end - start) / SAMPLE_RATE, 3),
            "score": round(float(score["max_score"]), 6),
            "event_count": score["event_count"],
            "timeline_rows": score["timeline_rows"],
            "pass_threshold": float(score["max_score"]) >= threshold,
        }
        window_rows.append(row)
        if best is None or row["score"] > best["score"]:
            best = row
    assert best is not None
    file_row = {
        "key": key,
        "wav": str(wav_path),
        "duration_sec": round(pcm.size / SAMPLE_RATE, 3),
        "window_count": len(windows),
        "best_score": best["score"],
        "best_start_sec": best["start_sec"],
        "best_end_sec": best["end_sec"],
        "best_event_count": best["event_count"],
        "pass_threshold": best["score"] >= threshold,
    }
    return file_row, window_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sliding-window native-equivalent WeKWS scoring on real voice WAVs.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\eval\usb_board_realvoice_20260603\manual_filtered_direct_score_ctc\manual_filtered_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            r"E:\CodeWorking\model\AnJuXiaoBaoKWS\wekws\fsmn_ctc_m2_m1_neg2500_pretrained_finetune_20260602_001\deploy_android\kws.onnx"
        ),
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=Path(
            r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\third_party\wekws\runtime\android\app\src\main\assets\models\m2_m1_neg2500_pretrained_finetune\kws_runtime_config.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\eval\usb_board_realvoice_20260603\sliding_window_native_eval"
        ),
    )
    parser.add_argument("--window-sec", type=float, default=2.4)
    parser.add_argument("--hop-sec", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runtime = load_runtime_config(args.runtime_config)
    threshold = float(args.threshold) if args.threshold is not None else float(runtime["threshold"])
    rows = read_jsonl(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    file_rows: list[dict[str, Any]] = []
    all_window_rows: list[dict[str, Any]] = []
    for row in rows:
        wav_path = Path(row.get("wav") or row.get("wav_path") or row.get("path"))
        key = str(row.get("key") or wav_path.stem)
        file_row, window_rows = evaluate_file(
            wav_path=wav_path,
            key=key,
            model_path=args.model,
            runtime=runtime,
            provider=args.onnx_provider,
            window_sec=args.window_sec,
            hop_sec=args.hop_sec,
            threshold=threshold,
        )
        file_rows.append(file_row)
        all_window_rows.extend(window_rows)

    with (args.output_dir / "sliding_file_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(file_rows[0].keys()))
        writer.writeheader()
        writer.writerows(file_rows)
    with (args.output_dir / "sliding_window_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_window_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_window_rows)

    scores = [float(row["best_score"]) for row in file_rows]
    summary = {
        "manifest": str(args.manifest),
        "model": str(args.model),
        "runtime_config": str(args.runtime_config),
        "output_dir": str(args.output_dir),
        "file_count": len(file_rows),
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "threshold": threshold,
        "speech_rms_threshold": runtime["speech_rms_threshold"],
        "speech_peak_threshold": runtime["speech_peak_threshold"],
        "soft_reset_interval_chunks": runtime["soft_reset_interval_chunks"],
        "max_best_score": round(max(scores), 6) if scores else 0.0,
        "avg_best_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
        "pass_count": sum(1 for score in scores if score >= threshold),
        "files": file_rows,
    }
    (args.output_dir / "sliding_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Sliding Window Real Voice Evaluation",
        "",
        f"- model: `{args.model}`",
        f"- window/hop: `{args.window_sec}s / {args.hop_sec}s`",
        f"- threshold: `{threshold}`",
        f"- best max score: `{summary['max_best_score']}`",
        f"- pass count: `{summary['pass_count']}/{summary['file_count']}`",
        "",
        "| key | best score | best window | pass |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in file_rows:
        lines.append(
            f"| {row['key']} | {row['best_score']:.6f} | {row['best_start_sec']:.3f}-{row['best_end_sec']:.3f}s | {row['pass_threshold']} |"
        )
    (args.output_dir / "SLIDING_WINDOW_EVAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
