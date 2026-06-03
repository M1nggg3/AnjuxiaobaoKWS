from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import yaml

from anju_kws.eval.replay_android_streaming import NativeEquivalentReplay
from anju_kws.eval.replay_android_streaming import (
    DEFAULT_SILENCE_CHUNKS_BEFORE_RESET,
    DEFAULT_SOFT_RESET_INTERVAL_CHUNKS,
    DEFAULT_SPEECH_PEAK_THRESHOLD,
    DEFAULT_SPEECH_RMS_THRESHOLD,
)


SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
KEYWORD = r"\u5b89\u5c45\u5c0f\u5b9d"


MODEL_SPECS = {
    "original_mid_20260509": {
        "checkpoint": Path(r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\experiments\pretrain_posbalanced_mid_20260509_001\5.pt"),
        "config": Path(r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\deploy\rk3566_wekws_model_mid_20260509\config.yaml"),
        "onnx": Path(r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\third_party\wekws\runtime\android\app\src\main\assets\models\original_mid_20260509\kws.onnx"),
        "runtime_config": Path(r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\third_party\wekws\runtime\android\app\src\main\assets\models\original_mid_20260509\kws_runtime_config.json"),
    },
    "m1_rawonly_finetune": {
        "checkpoint": Path(r"E:\CodeWorking\datasets\anju_m1_rawonly_finetune_package_20260528\model\final.pt"),
        "config": Path(r"E:\CodeWorking\datasets\anju_m1_rawonly_finetune_package_20260528\model\config.yaml"),
        "onnx": Path(r"E:\CodeWorking\datasets\anju_m1_rawonly_finetune_package_20260528\deploy_android\kws.onnx"),
        "runtime_config": Path(r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\third_party\wekws\runtime\android\app\src\main\assets\models\m1_rawonly_finetune\kws_runtime_config.json"),
    },
    "m1_rawonly_pretrain_full": {
        "checkpoint": Path(r"E:\CodeWorking\model\AnJuXiaoBaoKWS\wekws\experiments\fsmn_ctc_m1_rawonly_pretrain_full_20260528_001\19.pt"),
        "config": Path(r"E:\CodeWorking\model\AnJuXiaoBaoKWS\wekws\experiments\fsmn_ctc_m1_rawonly_pretrain_full_20260528_001\config.yaml"),
        "onnx": Path(r"E:\CodeWorking\model\AnJuXiaoBaoKWS\wekws\experiments\fsmn_ctc_m1_rawonly_pretrain_full_20260528_001\deploy_android\kws.onnx"),
        "runtime_config": Path(r"E:\CodeWorking\Project\AnJuXiaoBaoKWS\third_party\wekws\runtime\android\app\src\main\assets\models\m1_rawonly_pretrain_full\kws_runtime_config.json"),
    },
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_capture_path_and_epoch(log_path: Path) -> tuple[str, float]:
    pattern = re.compile(r"^\s*(?P<epoch>\d+\.\d+).*audio_raw_capture path=(?P<path>\S+)")
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            return match.group("path"), float(match.group("epoch"))
    raise RuntimeError(f"audio_raw_capture line not found in {log_path}")


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(x) for x in cmd))
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, check=check)


def pull_capture(adb_serial: str, remote_path: str, local_path: Path, cwd: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and local_path.stat().st_size > 0:
        return
    result = run(["adb", "-s", adb_serial, "pull", remote_path, str(local_path)], cwd=cwd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"adb pull failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def pcm_to_wav(pcm_path: Path, wav_path: Path, pcm: np.ndarray | None = None) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if pcm is None:
        pcm = np.fromfile(pcm_path, dtype="<i2")
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.astype("<i2", copy=False).tobytes())


def audio_stats(pcm: np.ndarray) -> dict:
    if pcm.size == 0:
        return {"duration": 0.0, "rms": 0.0, "peak": 0}
    f = pcm.astype(np.float32)
    return {
        "duration": round(float(pcm.size) / SAMPLE_RATE, 4),
        "rms": round(float(math.sqrt(np.mean(f * f))), 3),
        "peak": int(np.max(np.abs(pcm.astype(np.int32)))),
    }


def make_local_config(src_config: Path, cmvn: Path, out_config: Path) -> None:
    cfg = yaml.safe_load(src_config.read_text(encoding="utf-8"))
    cfg.setdefault("model", {}).setdefault("cmvn", {})["cmvn_file"] = str(cmvn).replace("\\", "/")
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def parse_score_file(path: Path) -> dict[str, dict]:
    parsed = {}
    if not path.exists():
        return parsed
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[0]
        if len(parts) >= 4 and parts[1] == "detected":
            parsed[key] = {"detected": True, "keyword": parts[2], "score": float(parts[3])}
        else:
            parsed[key] = {"detected": False, "keyword": None, "score": 0.0}
    return parsed


def load_streaming_kwargs(runtime_config_path: Path) -> dict:
    config = json.loads(runtime_config_path.read_text(encoding="utf-8"))
    return {
        "speech_rms_threshold": float(config.get("speech_rms_threshold", DEFAULT_SPEECH_RMS_THRESHOLD)),
        "speech_peak_threshold": int(config.get("speech_peak_threshold", DEFAULT_SPEECH_PEAK_THRESHOLD)),
        "silence_chunks_before_reset": int(config.get("silence_chunks_before_reset", DEFAULT_SILENCE_CHUNKS_BEFORE_RESET)),
        "soft_reset_interval_chunks": int(config.get("soft_reset_interval_chunks", DEFAULT_SOFT_RESET_INTERVAL_CHUNKS)),
    }


def summarize(rows: list[dict], score_key: str, detected_key: str) -> dict:
    positives = [r for r in rows if r["label_type"] == "positive"]
    negatives = [r for r in rows if r["label_type"] == "negative"]
    pos_hits = sum(1 for r in positives if r[detected_key])
    neg_hits = sum(1 for r in negatives if r[detected_key])
    return {
        "positive_hits": pos_hits,
        "positive_total": len(positives),
        "recall": round(pos_hits / len(positives), 4) if positives else 0.0,
        "negative_hits": neg_hits,
        "negative_total": len(negatives),
        "false_trigger_rate": round(neg_hits / len(negatives), 4) if negatives else 0.0,
        "max_score": round(max((float(r[score_key]) for r in rows), default=0.0), 6),
    }


def summarize_threshold(rows: list[dict], score_key: str, threshold: float) -> dict:
    positives = [r for r in rows if r["label_type"] == "positive"]
    negatives = [r for r in rows if r["label_type"] == "negative"]
    pos_hits = sum(1 for r in positives if float(r[score_key]) >= threshold)
    neg_hits = sum(1 for r in negatives if float(r[score_key]) >= threshold)
    return {
        "positive_hits": pos_hits,
        "positive_total": len(positives),
        "recall": round(pos_hits / len(positives), 4) if positives else 0.0,
        "negative_hits": neg_hits,
        "negative_total": len(negatives),
        "false_trigger_rate": round(neg_hits / len(negatives), 4) if negatives else 0.0,
        "max_score": round(max((float(r[score_key]) for r in rows), default=0.0), 6),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adb-serial", default="192.168.3.232:36465")
    parser.add_argument("--python", default=r"D:\conda-envs\cosyvoice310\python.exe")
    parser.add_argument("--cmvn", type=Path, default=Path(r"E:\CodeWorking\model\AnJuXiaoBaoKWS\wekws\pretrained\fsmn_ctc_wenwen\global_cmvn.kaldi"))
    parser.add_argument("--dict-dir", type=Path, default=Path(r"E:\CodeWorking\model\AnJuXiaoBaoKWS\wekws\dict\m1_rawonly_20260528"))
    parser.add_argument("--margin-pre", type=float, default=0.3)
    parser.add_argument("--margin-post", type=float, default=0.8)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--onnx-provider", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    repo = Path.cwd()
    out_root = args.run_dir / "segment_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "third_party" / "wekws") + os.pathsep + str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")

    all_summaries = {}
    report_lines = [
        "# Raw Playback Window Segment Evaluation",
        "",
        f"- run_dir: `{args.run_dir}`",
        f"- margin: `{args.margin_pre}s` before playback, `{args.margin_post}s` after playback",
        f"- threshold: `{args.threshold}`",
        "",
        "| model | method | positive hits | recall | negative hits | false trigger | max score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model_name, spec in MODEL_SPECS.items():
        model_dir = args.run_dir / model_name
        events_path = model_dir / "playback_events.jsonl"
        log_path = model_dir / "logcat.txt"
        if not events_path.exists() or not log_path.exists():
            print(f"[skip] missing events/log for {model_name}")
            continue

        remote_capture, capture_epoch = find_capture_path_and_epoch(log_path)
        model_out = out_root / model_name
        captures_dir = args.run_dir / "board_captures" / model_name
        local_pcm = captures_dir / Path(remote_capture).name
        local_wav = local_pcm.with_suffix(".wav")
        pull_capture(args.adb_serial, remote_capture, local_pcm, repo)
        full_pcm = np.fromfile(local_pcm, dtype="<i2")
        pcm_to_wav(local_pcm, local_wav, full_pcm)

        segment_rows = []
        data_list = []
        for event in read_jsonl(events_path):
            start_sec = max(0.0, float(event["play_start_epoch"]) - capture_epoch - args.margin_pre)
            end_sec = min(float(full_pcm.size) / SAMPLE_RATE, float(event["play_end_epoch"]) - capture_epoch + args.margin_post)
            start_sample = int(round(start_sec * SAMPLE_RATE))
            end_sample = int(round(end_sec * SAMPLE_RATE))
            segment = full_pcm[start_sample:end_sample]
            label_type = event["label_type"]
            seg_dir = model_out / "segments" / label_type
            seg_pcm = seg_dir / f"{event['test_id']}.pcm"
            seg_wav = seg_dir / f"{event['test_id']}.wav"
            seg_dir.mkdir(parents=True, exist_ok=True)
            segment.astype("<i2", copy=False).tofile(seg_pcm)
            pcm_to_wav(seg_pcm, seg_wav, segment)
            stats = audio_stats(segment)
            row = {
                **event,
                "model": model_name,
                "capture_epoch": capture_epoch,
                "segment_start_sec": round(start_sec, 4),
                "segment_end_sec": round(end_sec, 4),
                "segment_pcm": str(seg_pcm),
                "segment_wav": str(seg_wav),
                **stats,
            }
            segment_rows.append(row)
            data_list.append({
                "key": event["test_id"],
                "wav": str(seg_wav),
                "txt": "安 居 小 宝" if label_type == "positive" else "<filler>",
                "duration": stats["duration"],
            })

        write_jsonl(model_out / "segments_manifest.jsonl", segment_rows)
        write_jsonl(model_out / "data.list", data_list)

        local_config = model_out / "config.local.yaml"
        make_local_config(spec["config"], args.cmvn, local_config)
        score_file = model_out / "score_ctc.txt"
        score_stdout = model_out / "score_ctc.stdout.log"
        score_stderr = model_out / "score_ctc.stderr.log"
        cmd = [
            args.python,
            str(repo / "third_party" / "wekws" / "wekws" / "bin" / "score_ctc.py"),
            "--config", str(local_config),
            "--test_data", str(model_out / "data.list"),
            "--dict", str(args.dict_dir),
            "--gpu", "0",
            "--checkpoint", str(spec["checkpoint"]),
            "--batch_size", "8",
            "--num_workers", "0",
            "--score_file", str(score_file),
            "--keywords", KEYWORD,
        ]
        result = run(cmd, repo, env=env, check=False)
        score_stdout.write_text(result.stdout, encoding="utf-8", errors="ignore")
        score_stderr.write_text(result.stderr, encoding="utf-8", errors="ignore")
        if result.returncode != 0:
            raise RuntimeError(f"score_ctc failed for {model_name}; see {score_stderr}")

        score_rows = parse_score_file(score_file)
        streaming_kwargs = load_streaming_kwargs(spec["runtime_config"])
        native_runner = NativeEquivalentReplay(
            spec["onnx"], args.threshold, args.onnx_provider, **streaming_kwargs
        )
        combined_rows = []
        for row in segment_rows:
            key = row["test_id"]
            pcm = np.fromfile(row["segment_pcm"], dtype="<i2")
            native = native_runner.replay(pcm, f"{model_name}_{key}")
            timeline_scores = [float(item.get("score", 0.0)) for item in native.get("timeline", [])]
            native_max = max(timeline_scores, default=0.0)
            score_item = score_rows.get(key, {"detected": False, "score": 0.0})
            combined = {
                **row,
                "score_ctc_detected": bool(score_item["detected"]),
                "score_ctc_score": float(score_item["score"]),
                "native_detected": bool(native.get("events")),
                "native_score": float(native_max),
                "native_event_count": len(native.get("events", [])),
            }
            combined_rows.append(combined)

        write_jsonl(model_out / "combined_segment_results.jsonl", combined_rows)
        score_summary = summarize(combined_rows, "score_ctc_score", "score_ctc_detected")
        score_threshold_summary = summarize_threshold(combined_rows, "score_ctc_score", args.threshold)
        native_summary = summarize(combined_rows, "native_score", "native_detected")
        all_summaries[model_name] = {
            "raw_capture": str(local_pcm),
            "raw_capture_wav": str(local_wav),
            "capture_epoch": capture_epoch,
            "segment_count": len(combined_rows),
            "score_ctc": score_summary,
            "score_ctc_thresholded": score_threshold_summary,
            "native_stream": native_summary,
        }
        for method, summary in [
            ("score_ctc_detected", score_summary),
            (f"score_ctc@{args.threshold}", score_threshold_summary),
            ("python_native_stream", native_summary),
        ]:
            report_lines.append(
                f"| {model_name} | {method} | "
                f"{summary['positive_hits']}/{summary['positive_total']} | {summary['recall']:.4f} | "
                f"{summary['negative_hits']}/{summary['negative_total']} | {summary['false_trigger_rate']:.4f} | "
                f"{summary['max_score']:.6f} |"
            )

    (out_root / "summary.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "SEGMENT_EVAL_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))
    print(f"report: {out_root / 'SEGMENT_EVAL_REPORT.md'}")


if __name__ == "__main__":
    main()
