#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.anju_xiaobao_kws_env" ]; then
  # shellcheck disable=SC1090
  source "$HOME/.anju_xiaobao_kws_env"
fi

PROJECT="${ANJU_PROJECT:-$HOME/projects/AnJuXiaoBaoKWS}"
PYTHON="${PYTHON:-$HOME/miniforge3/envs/cosyvoice310/bin/python}"
EXP_DIR="${WEKWS_EXP_DIR:-$HOME/models/AnJuXiaoBaoKWS/wekws/experiments/fsmn_ctc_farfield_main_20260521_001}"
DICT_DIR="${WEKWS_DICT_DIR:-$HOME/models/AnJuXiaoBaoKWS/wekws/dict/farfield_main_20260521}"
PREPARED_DIR="${PREPARED_DIR:-$HOME/datasets/AnJuXiaoBaoKWS/data/prepared_wekws_farfield_main_20260521}"
BASE_DATASET="${ANJU_BASE_DATASET:-$HOME/datasets/AnJuXiaoBaoKWS/data/anju_xiaobao_kws_dataset_20260508}"
RUN_ROOT="${ANJU_RUN_ROOT:-$HOME/runs/AnJuXiaoBaoKWS}"

RUN_DIR="$RUN_ROOT/wekws_score_ctc_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR/data" "$RUN_DIR/scores"

export PYTHONPATH="$PROJECT/src:$PROJECT/third_party/wekws:${PYTHONPATH:-}"

CONFIG="$EXP_DIR/config.yaml"
CHECKPOINT="$EXP_DIR/final.pt"
KEYWORD_UNICODE="\\u5b89\\u5c45\\u5c0f\\u5b9d"
SCORE_CTC="$PROJECT/third_party/wekws/wekws/bin/score_ctc.py"

cat > "$RUN_DIR/build_eval_data.py" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torchaudio


run_dir = Path(sys.argv[1])
base_dataset = Path(sys.argv[2])
prepared_dir = Path(sys.argv[3])

real_manifest = base_dataset / "manifests" / "real_positive_wakeup_trimmed.jsonl"
continuous_manifest = base_dataset / "manifests" / "continuous_false_alarm_candidates.jsonl"

prepared_test = prepared_dir / "test" / "data.list"
if not prepared_test.exists():
    raise FileNotFoundError(prepared_test)
if not real_manifest.exists():
    raise FileNotFoundError(real_manifest)
if not continuous_manifest.exists():
    raise FileNotFoundError(continuous_manifest)

real_dir = run_dir / "data" / "real27_trimmed"
continuous_dir = run_dir / "data" / "continuous_5s"
chunk_wav_dir = continuous_dir / "wav"
real_dir.mkdir(parents=True, exist_ok=True)
chunk_wav_dir.mkdir(parents=True, exist_ok=True)

real_list = real_dir / "data.list"
continuous_list = continuous_dir / "data.list"
chunk_sec = 5.0
target_sr = 16000

real_count = 0
with real_manifest.open("r", encoding="utf-8") as fin, real_list.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        wav = base_dataset / row["relative_path"]
        if not wav.exists():
            raise FileNotFoundError(wav)
        item = {
            "key": row.get("sample_id") or wav.stem,
            "wav": str(wav),
            "txt": "安 居 小 宝",
            "duration": float(row.get("duration_sec", 0.0)),
        }
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
        real_count += 1

continuous_source_count = 0
continuous_chunk_count = 0
continuous_duration = 0.0
with continuous_manifest.open("r", encoding="utf-8") as fin, continuous_list.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        wav = base_dataset / row["relative_path"]
        if not wav.exists():
            raise FileNotFoundError(wav)
        audio, sr = torchaudio.load(str(wav))
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
            sr = target_sr
        audio = audio[:1].contiguous()
        total = audio.shape[1]
        chunk_len = int(chunk_sec * sr)
        if total <= 0:
            continue
        continuous_source_count += 1
        for start in range(0, total, chunk_len):
            end = min(start + chunk_len, total)
            duration = (end - start) / sr
            if duration < 1.0:
                continue
            chunk = torch.clamp(audio[:, start:end], -1.0, 1.0)
            key = f"{row.get('sample_id') or wav.stem}_chunk{continuous_chunk_count:04d}_s{start // sr:06d}"
            out_wav = chunk_wav_dir / f"{key}.wav"
            torchaudio.save(str(out_wav), chunk, sr, encoding="PCM_S", bits_per_sample=16)
            item = {
                "key": key,
                "wav": str(out_wav),
                "txt": "<filler>",
                "duration": duration,
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            continuous_chunk_count += 1
            continuous_duration += duration

summary = {
    "prepared_test_data": str(prepared_test),
    "real27_data": str(real_list),
    "continuous_5s_data": str(continuous_list),
    "real27_count": real_count,
    "continuous_source_count": continuous_source_count,
    "continuous_chunk_count": continuous_chunk_count,
    "continuous_duration_sec": continuous_duration,
    "chunk_sec": chunk_sec,
}
(run_dir / "eval_data_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

"$PYTHON" "$RUN_DIR/build_eval_data.py" "$RUN_DIR" "$BASE_DATASET" "$PREPARED_DIR" | tee "$RUN_DIR/build_eval_data.log"

run_score() {
  local name="$1"
  local data_list="$2"
  local batch_size="$3"
  local score_file="$RUN_DIR/scores/${name}.score"
  local log_file="$RUN_DIR/scores/${name}.log"
  echo "[score] $name"
  "$PYTHON" "$SCORE_CTC" \
    --config "$CONFIG" \
    --test_data "$data_list" \
    --dict "$DICT_DIR" \
    --gpu 0 \
    --checkpoint "$CHECKPOINT" \
    --batch_size "$batch_size" \
    --num_workers 1 \
    --prefetch 8 \
    --score_file "$score_file" \
    --keywords "$KEYWORD_UNICODE" 2>&1 | tee "$log_file"
}

run_score "prepared_test" "$PREPARED_DIR/test/data.list" 32
run_score "real27_trimmed" "$RUN_DIR/data/real27_trimmed/data.list" 16
run_score "continuous_5s" "$RUN_DIR/data/continuous_5s/data.list" 16

cat > "$RUN_DIR/summarize_scores.py" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


run_dir = Path(sys.argv[1])


def read_labels(data_list: Path) -> dict[str, dict]:
    labels = {}
    with data_list.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[row["key"]] = {
                "label": "positive" if row.get("txt") != "<filler>" else "negative",
                "duration": float(row.get("duration", 0.0)),
            }
    return labels


def read_scores(score_file: Path) -> dict[str, dict]:
    scores = {}
    with score_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            key = parts[0]
            if len(parts) >= 4 and parts[1] == "detected":
                scores[key] = {
                    "detected": True,
                    "keyword": parts[2],
                    "score": float(parts[3]),
                }
            else:
                scores[key] = {"detected": False, "keyword": None, "score": 0.0}
    return scores


def summarize(name: str, data_list: Path, score_file: Path) -> dict:
    labels = read_labels(data_list)
    scores = read_scores(score_file)
    total = len(labels)
    positives = sum(1 for v in labels.values() if v["label"] == "positive")
    negatives = total - positives
    tp = fn = fp = tn = 0
    detected_scores = []
    false_alarm_examples = []
    miss_examples = []
    for key, meta in labels.items():
        result = scores.get(key, {"detected": False, "score": 0.0})
        detected = bool(result["detected"])
        score = float(result["score"])
        if detected:
            detected_scores.append(score)
        if meta["label"] == "positive":
            if detected:
                tp += 1
            else:
                fn += 1
                if len(miss_examples) < 20:
                    miss_examples.append(key)
        else:
            if detected:
                fp += 1
                false_alarm_examples.append((score, key))
            else:
                tn += 1
    false_alarm_examples.sort(reverse=True)
    duration_sec = sum(v["duration"] for v in labels.values())
    out = {
        "name": name,
        "total": total,
        "positive": positives,
        "negative": negatives,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "recall": tp / positives if positives else None,
        "false_alarm_rate": fp / negatives if negatives else None,
        "duration_sec": duration_sec,
        "false_alarms_per_hour": fp / (duration_sec / 3600.0) if duration_sec > 0 else None,
        "detected_score_min": min(detected_scores) if detected_scores else None,
        "detected_score_max": max(detected_scores) if detected_scores else None,
        "detected_score_avg": sum(detected_scores) / len(detected_scores) if detected_scores else None,
        "top_false_alarm_examples": [
            {"key": key, "score": score} for score, key in false_alarm_examples[:20]
        ],
        "miss_examples": miss_examples,
    }
    return out


items = {
    "prepared_test": (
        Path("/home/clm/datasets/AnJuXiaoBaoKWS/data/prepared_wekws_farfield_main_20260521/test/data.list"),
        run_dir / "scores" / "prepared_test.score",
    ),
    "real27_trimmed": (
        run_dir / "data" / "real27_trimmed" / "data.list",
        run_dir / "scores" / "real27_trimmed.score",
    ),
    "continuous_5s": (
        run_dir / "data" / "continuous_5s" / "data.list",
        run_dir / "scores" / "continuous_5s.score",
    ),
}
summary = {
    "run_dir": str(run_dir),
    "items": [summarize(name, data, score) for name, (data, score) in items.items()],
}
(run_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

"$PYTHON" "$RUN_DIR/summarize_scores.py" "$RUN_DIR" | tee "$RUN_DIR/summary.log"

echo "Run directory: $RUN_DIR"
