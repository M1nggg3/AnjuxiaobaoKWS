from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SAMPLE_RATE = 16000
KEYWORD = r"\u5b89\u5c45\u5c0f\u5b9d"


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


def write_wav_pcm16(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.astype("<i2", copy=False).tobytes())


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


def make_local_config(src_config: Path, cmvn: Path, out_config: Path) -> None:
    cfg = yaml.safe_load(src_config.read_text(encoding="utf-8"))
    cfg.setdefault("model", {}).setdefault("cmvn", {})["cmvn_file"] = str(cmvn).replace("\\", "/")
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def parse_score_file(path: Path) -> dict[str, dict[str, Any]]:
    parsed: dict[str, dict[str, Any]] = {}
    pattern = re.compile(r"^(?P<key>\S+) detected (?P<keyword>\S+) (?P<score>[\d.]+)")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if match:
            parsed[match.group("key")] = {
                "detected": True,
                "keyword": match.group("keyword"),
                "score": float(match.group("score")),
            }
            continue
        parts = line.strip().split()
        if parts:
            parsed[parts[0]] = {"detected": False, "keyword": None, "score": 0.0}
    return parsed


def build_windows(rows: list[dict[str, Any]], out_dir: Path, window_sec: float, hop_sec: float) -> list[dict[str, Any]]:
    window_rows: list[dict[str, Any]] = []
    wav_dir = out_dir / "windows_wav"
    for row in rows:
        wav_path = Path(row.get("wav") or row.get("wav_path") or row.get("path"))
        key = str(row.get("key") or wav_path.stem)
        pcm = read_wav_pcm16(wav_path)
        for index, (start, end) in enumerate(iter_windows(pcm, window_sec, hop_sec), start=1):
            win_key = f"{key}_win{index:03d}_{start / SAMPLE_RATE:.2f}_{end / SAMPLE_RATE:.2f}".replace(".", "p")
            win_path = wav_dir / f"{win_key}.wav"
            write_wav_pcm16(win_path, pcm[start:end])
            window_rows.append(
                {
                    "source_key": key,
                    "key": win_key,
                    "wav": str(win_path).replace("\\", "/"),
                    "txt": "? ? ? ?",
                    "source_wav": str(wav_path),
                    "start_sec": round(start / SAMPLE_RATE, 3),
                    "end_sec": round(end / SAMPLE_RATE, 3),
                    "duration": round((end - start) / SAMPLE_RATE, 3),
                }
            )
    return window_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_score_ctc(
    repo: Path,
    python: str,
    config: Path,
    data_list: Path,
    dict_dir: Path,
    checkpoint: Path,
    score_file: Path,
    gpu: int,
) -> tuple[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "third_party" / "wekws") + os.pathsep + str(repo / "src") + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    cmd = [
        python,
        str(repo / "third_party" / "wekws" / "wekws" / "bin" / "score_ctc.py"),
        "--config",
        str(config),
        "--test_data",
        str(data_list),
        "--dict",
        str(dict_dir),
        "--checkpoint",
        str(checkpoint),
        "--score_file",
        str(score_file),
        "--keywords",
        KEYWORD,
        "--gpu",
        str(gpu),
        "--batch_size",
        "16",
    ]
    result = subprocess.run(cmd, cwd=str(repo), env=env, text=True, capture_output=True, check=False)
    (score_file.parent / "score_ctc_stdout.log").write_text(result.stdout, encoding="utf-8", errors="replace")
    (score_file.parent / "score_ctc_stderr.log").write_text(result.stderr, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"score_ctc failed with code {result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
    return result.stdout, result.stderr


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sliding-window official WeKWS score_ctc evaluation.")
    parser.add_argument("--repo", type=Path, default=Path(r"E:\CodeWorking\Project\AnJuXiaoBaoKWS"))
    parser.add_argument("--python", default=r"D:\conda-envs\cosyvoice310\python.exe")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\eval\usb_board_realvoice_20260603\manual_filtered_direct_score_ctc\manual_filtered_manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(
            r"E:\CodeWorking\model\AnJuXiaoBaoKWS\wekws\fsmn_ctc_m2_m1_neg2500_pretrained_finetune_20260602_001"
        ),
    )
    parser.add_argument(
        "--dict-dir",
        type=Path,
        default=Path(r"E:\CodeWorking\model\AnJuXiaoBaoKWS\wekws\dict\m1_rawonly_20260528"),
    )
    parser.add_argument(
        "--cmvn",
        type=Path,
        default=Path(r"E:\CodeWorking\model\AnJuXiaoBaoKWS\wekws\pretrained\fsmn_ctc_wenwen\global_cmvn.kaldi"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\eval\usb_board_realvoice_20260603\sliding_window_score_ctc_eval"
        ),
    )
    parser.add_argument("--checkpoint-name", default="28.pt")
    parser.add_argument("--window-sec", type=float, default=2.4)
    parser.add_argument("--hop-sec", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--gpu", type=int, default=-1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(args.manifest)
    window_rows = build_windows(source_rows, args.output_dir, args.window_sec, args.hop_sec)
    data_list = args.output_dir / "windows_data.list"
    write_jsonl(data_list, [{"key": r["key"], "wav": r["wav"], "txt": r["txt"]} for r in window_rows])
    local_config = args.output_dir / "config.local.yaml"
    make_local_config(args.model_dir / "config.yaml", args.cmvn, local_config)
    score_file = args.output_dir / "score_ctc_windows.txt"
    run_score_ctc(
        repo=args.repo,
        python=args.python,
        config=local_config,
        data_list=data_list,
        dict_dir=args.dict_dir,
        checkpoint=args.model_dir / args.checkpoint_name,
        score_file=score_file,
        gpu=args.gpu,
    )
    scores = parse_score_file(score_file)
    for row in window_rows:
        parsed = scores.get(row["key"], {"detected": False, "score": 0.0})
        row["detected"] = bool(parsed["detected"])
        row["score"] = float(parsed["score"])
        row["pass_threshold"] = row["score"] >= args.threshold

    by_source: dict[str, dict[str, Any]] = {}
    for row in window_rows:
        current = by_source.get(row["source_key"])
        if current is None or row["score"] > current["best_score"]:
            by_source[row["source_key"]] = {
                "key": row["source_key"],
                "source_wav": row["source_wav"],
                "best_window_key": row["key"],
                "best_score": row["score"],
                "best_detected": row["detected"],
                "best_start_sec": row["start_sec"],
                "best_end_sec": row["end_sec"],
                "pass_threshold": row["score"] >= args.threshold,
            }
    file_rows = list(by_source.values())

    with (args.output_dir / "sliding_score_ctc_windows.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = [
            "source_key",
            "key",
            "source_wav",
            "start_sec",
            "end_sec",
            "duration",
            "detected",
            "score",
            "pass_threshold",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(window_rows)
    with (args.output_dir / "sliding_score_ctc_files.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(file_rows[0].keys()))
        writer.writeheader()
        writer.writerows(file_rows)

    best_scores = [float(row["best_score"]) for row in file_rows]
    summary = {
        "manifest": str(args.manifest),
        "model_dir": str(args.model_dir),
        "checkpoint": str(args.model_dir / args.checkpoint_name),
        "output_dir": str(args.output_dir),
        "file_count": len(file_rows),
        "window_count": len(window_rows),
        "window_sec": args.window_sec,
        "hop_sec": args.hop_sec,
        "threshold": args.threshold,
        "max_best_score": round(max(best_scores), 6) if best_scores else 0.0,
        "avg_best_score": round(sum(best_scores) / len(best_scores), 6) if best_scores else 0.0,
        "pass_count": sum(1 for score in best_scores if score >= args.threshold),
        "files": file_rows,
    }
    (args.output_dir / "sliding_score_ctc_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Sliding Window score_ctc Evaluation",
        "",
        f"- checkpoint: `{args.model_dir / args.checkpoint_name}`",
        f"- window/hop: `{args.window_sec}s / {args.hop_sec}s`",
        f"- threshold: `{args.threshold}`",
        f"- max best score: `{summary['max_best_score']}`",
        f"- pass count: `{summary['pass_count']}/{summary['file_count']}`",
        "",
        "| key | best score | best window | pass |",
        "| --- | ---: | --- | ---: |",
    ]
    for row in file_rows:
        lines.append(
            f"| {row['key']} | {row['best_score']:.3f} | {row['best_start_sec']:.3f}-{row['best_end_sec']:.3f}s | {row['pass_threshold']} |"
        )
    (args.output_dir / "SLIDING_SCORE_CTC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
