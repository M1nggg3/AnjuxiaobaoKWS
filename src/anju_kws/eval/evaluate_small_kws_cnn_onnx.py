from __future__ import annotations

import argparse
import csv
import json
import math
import wave
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torchaudio


DEFAULT_MODEL_DIR = Path(
    r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data"
    r"\server_runs\baseline_kws_resume_from_epoch23"
)
DEFAULT_THRESHOLDS = (0.5, 0.7, 0.8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate SmallKwsCnn ONNX on captured far-field WAVs.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--thresholds", default="0.5,0.7,0.8")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--provider",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="ONNX Runtime provider preference.",
    )
    return parser


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def default_manifest(experiment_dir: Path) -> Path:
    preferred = experiment_dir / "manifests" / "training_raw_all.jsonl"
    if preferred.exists():
        return preferred
    return experiment_dir / "manifests" / "recordings.jsonl"


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / float(audio.getframerate())


def resolve_wav(row: dict) -> Path:
    for key in ("training_wav_path", "raw_wav_path", "wav", "path"):
        value = row.get(key)
        if value:
            return Path(value)
    raise ValueError(f"Row has no WAV path: {row.get('capture_id') or row.get('utt_id')}")


def load_waveform(path: Path, sample_rate: int) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform.squeeze(0)


def active_center_crop_or_pad(waveform: torch.Tensor, target_samples: int, sample_rate: int) -> torch.Tensor:
    if waveform.numel() == 0:
        return torch.zeros(target_samples)
    if waveform.numel() <= target_samples:
        left = (target_samples - waveform.numel()) // 2
        right = target_samples - waveform.numel() - left
        return torch.nn.functional.pad(waveform, (left, right))

    frame = max(1, int(0.02 * sample_rate))
    hop = max(1, int(0.01 * sample_rate))
    frames = waveform.unfold(0, frame, hop)
    rms = torch.sqrt(torch.mean(frames * frames, dim=1) + 1e-8)
    threshold = max(float(rms.max()) * 0.10, 0.002)
    active = torch.nonzero(rms > threshold, as_tuple=False).flatten()
    if active.numel() == 0:
        center = waveform.numel() // 2
    else:
        start = int(active[0]) * hop
        end = min(waveform.numel(), int(active[-1]) * hop + frame)
        center = (start + end) // 2
    start = max(0, min(center - target_samples // 2, waveform.numel() - target_samples))
    return waveform[start : start + target_samples]


class LogMelFrontend:
    def __init__(self, config: dict):
        frontend = config["default_audio_frontend"]
        self.sample_rate = int(frontend["sample_rate"])
        self.input_seconds = float(frontend["input_seconds"])
        self.target_samples = int(round(self.sample_rate * self.input_seconds))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=int(frontend["n_fft"]),
            win_length=int(frontend["win_length"]),
            hop_length=int(frontend["hop_length"]),
            n_mels=int(frontend["n_mels"]),
            center=True,
            power=2.0,
        )

    def __call__(self, path: Path) -> np.ndarray:
        waveform = load_waveform(path, self.sample_rate)
        waveform = active_center_crop_or_pad(waveform, self.target_samples, self.sample_rate)
        features = torch.log(self.mel(waveform).clamp_min(1e-10))
        features = (features - features.mean()) / (features.std() + 1e-5)
        return features.unsqueeze(0).numpy().astype(np.float32)


def provider_list(choice: str) -> list[str]:
    available = set(ort.get_available_providers())
    if choice == "cpu":
        return ["CPUExecutionProvider"]
    if choice == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def score_rows(rows: list[dict], model_dir: Path, batch_size: int, provider: str) -> list[dict]:
    export_config = json.loads((model_dir / "anjuxiaobao_kws.export.json").read_text(encoding="utf-8"))
    frontend = LogMelFrontend(export_config)
    session = ort.InferenceSession(
        str(model_dir / "anjuxiaobao_kws.onnx"),
        providers=provider_list(provider),
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output: list[dict] = []
    pending_features: list[np.ndarray] = []
    pending_rows: list[tuple[dict, Path]] = []

    def flush() -> None:
        if not pending_features:
            return
        batch = np.stack(pending_features, axis=0)
        scores = session.run([output_name], {input_name: batch})[0]
        for (row, wav_path), score in zip(pending_rows, scores):
            item = dict(row)
            item["wav_path"] = str(wav_path)
            item["duration_sec"] = row.get("duration_sec") or round(wav_duration(wav_path), 4)
            item["score"] = float(score)
            output.append(item)
        pending_features.clear()
        pending_rows.clear()

    for row in rows:
        if row.get("status", "verified") != "verified":
            continue
        wav_path = resolve_wav(row)
        if not wav_path.exists():
            item = dict(row)
            item["score"] = math.nan
            item["error"] = f"missing wav: {wav_path}"
            output.append(item)
            continue
        pending_features.append(frontend(wav_path))
        pending_rows.append((row, wav_path))
        if len(pending_features) >= batch_size:
            flush()
    flush()
    return output


def metrics_for_threshold(rows: list[dict], threshold: float) -> dict:
    valid = [row for row in rows if not math.isnan(float(row["score"]))]
    positives = [row for row in valid if row.get("label_type") == "positive" or int(row.get("label", 0)) == 1]
    negatives = [row for row in valid if row.get("label_type") == "negative" or int(row.get("label", 1)) == 0]
    tp = sum(float(row["score"]) >= threshold for row in positives)
    fp = sum(float(row["score"]) >= threshold for row in negatives)
    return {
        "threshold": threshold,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "recall": tp / len(positives) if positives else 0.0,
        "false_alarm": fp / len(negatives) if negatives else 0.0,
        "accuracy": (tp + (len(negatives) - fp)) / len(valid) if valid else 0.0,
        "tp": int(tp),
        "fn": len(positives) - int(tp),
        "fp": int(fp),
        "tn": len(negatives) - int(fp),
    }


def grouped_metrics(rows: list[dict], threshold: float) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = row.get("label_type", "unknown")
        distance = row.get("distance", "unknown")
        groups[f"{label}|distance={distance}"].append(row)
    output = {}
    for key, group_rows in sorted(groups.items()):
        scores = [float(row["score"]) for row in group_rows if not math.isnan(float(row["score"]))]
        detected = sum(score >= threshold for score in scores)
        output[key] = {
            "count": len(scores),
            "detected": int(detected),
            "rate": detected / len(scores) if scores else 0.0,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
        }
    return output


def write_scores(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "capture_id",
        "utt_id",
        "label_type",
        "label",
        "distance",
        "source_sample_id",
        "text",
        "wav_path",
        "duration_sec",
        "score",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def recommend_threshold(rows: list[dict], thresholds: list[float]) -> float:
    candidates = []
    for threshold in thresholds:
        overall = metrics_for_threshold(rows, threshold)
        grouped = grouped_metrics(rows, threshold)
        pos_1m = grouped.get("positive|distance=1m", {}).get("rate", 0.0)
        pos_3m = grouped.get("positive|distance=3m", {}).get("rate", 0.0)
        pos_5m = grouped.get("positive|distance=5m", {}).get("rate", 0.0)
        ok = pos_1m >= 0.90 and pos_3m >= 0.80 and pos_5m >= 0.70 and overall["false_alarm"] <= 0.20
        candidates.append((ok, threshold, overall["recall"], overall["false_alarm"]))
    valid = [item for item in candidates if item[0]]
    if valid:
        return max(valid, key=lambda item: item[1])[1]
    return max(candidates, key=lambda item: (item[2] - item[3], item[2]))[1]


def write_report(path: Path, rows: list[dict], metrics: dict, recommended: float) -> None:
    lines = [
        "# SmallKwsCnn ONNX 离线评估报告",
        "",
        f"推荐阈值：`{recommended:.2f}`",
        "",
        "## 阈值指标",
        "",
        "| threshold | recall | false_alarm | accuracy | TP | FN | FP | TN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, value in metrics["thresholds"].items():
        lines.append(
            f"| {key} | {value['recall']:.3f} | {value['false_alarm']:.3f} | "
            f"{value['accuracy']:.3f} | {value['tp']} | {value['fn']} | {value['fp']} | {value['tn']} |"
        )
    lines.extend(["", "## 分距离 / 分标签统计", ""])
    for key, value in metrics[f"grouped@{recommended:.2f}"].items():
        lines.append(
            f"- {key}: count={value['count']}, detected={value['detected']}, "
            f"rate={value['rate']:.3f}, mean_score={value['mean_score']:.3f}"
        )

    misses = [
        row for row in rows
        if row.get("label_type") == "positive" and float(row["score"]) < recommended
    ]
    false_alarms = [
        row for row in rows
        if row.get("label_type") == "negative" and float(row["score"]) >= recommended
    ]
    lines.extend(["", "## 漏检样本", ""])
    for row in misses[:50]:
        lines.append(
            f"- {row.get('distance')} {row.get('capture_id') or row.get('utt_id')} score={float(row['score']):.3f}"
        )
    if len(misses) > 50:
        lines.append(f"- ... 还有 {len(misses) - 50} 条")
    lines.extend(["", "## 误触发样本", ""])
    for row in false_alarms[:50]:
        lines.append(
            f"- {row.get('distance')} {row.get('capture_id') or row.get('utt_id')} "
            f"score={float(row['score']):.3f} text={row.get('text', '')}"
        )
    if len(false_alarms) > 50:
        lines.append(f"- ... 还有 {len(false_alarms) - 50} 条")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    manifest = args.manifest or default_manifest(args.experiment_dir)
    output_dir = args.output_dir or args.experiment_dir / "offline_eval" / args.model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(manifest)
    scored = score_rows(rows, args.model_dir, args.batch_size, args.provider)
    write_scores(output_dir / "offline_scores.csv", scored)

    metrics = {
        "manifest": str(manifest),
        "model_dir": str(args.model_dir),
        "provider_request": args.provider,
        "rows": len(scored),
        "by_label": dict(Counter(row.get("label_type", "unknown") for row in scored)),
        "thresholds": {
            f"{threshold:.2f}": metrics_for_threshold(scored, threshold)
            for threshold in thresholds
        },
    }
    recommended = recommend_threshold(scored, thresholds)
    metrics[f"grouped@{recommended:.2f}"] = grouped_metrics(scored, recommended)
    metrics["recommended_threshold"] = recommended
    (output_dir / "offline_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "OFFLINE_EVAL_REPORT.md", scored, metrics, recommended)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
