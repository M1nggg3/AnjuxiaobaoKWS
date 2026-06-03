from __future__ import annotations

import hashlib
import json
import math
import random
import re
import wave
from array import array
from pathlib import Path
from typing import Iterable, Sequence


EXCLUDED_HOMOPHONE_NEGATIVES = {
    "安居小包",
    "安居小保",
    "安居小饱",
    "安居晓宝",
}
NEAR_NEGATIVE_TEXTS = {
    "安静小宝",
    "安居宝宝",
    "小宝小宝",
    "你好小宝",
    "安居小贝",
    "安居小播",
    "安居小白",
    "安居小帮",
}
DISTANCES = ("1m", "3m", "5m")
SAMPLE_RATE = 16000


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def compact_text(text: str) -> str:
    return re.sub(r"[\s，。！？!?、,.；;：:]+", "", text or "")


def is_allowed_negative(row: dict) -> bool:
    if row.get("label_type") != "negative":
        return False
    text = compact_text(str(row.get("text", "")))
    return text in NEAR_NEGATIVE_TEXTS and text not in EXCLUDED_HOMOPHONE_NEGATIVES


def _balanced_take(rows: Sequence[dict], count: int, seed: int) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"Requested {count} rows but only {len(rows)} are available")
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("gender", "unknown")), str(row.get("source_dataset", "unknown")))
        buckets.setdefault(key, []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)
    chosen: list[dict] = []
    while len(chosen) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(chosen) < count:
                chosen.append(buckets[key].pop())
                progressed = True
        if not progressed:
            raise ValueError("Unable to fill balanced selection")
    return chosen


def select_source_samples(
    rows: Sequence[dict], positive_count: int, negative_count: int, seed: int
) -> list[dict]:
    positives = [row for row in rows if row.get("label_type") == "positive"]
    negatives = [row for row in rows if is_allowed_negative(row)]
    selected = _balanced_take(positives, positive_count, seed)
    selected.extend(_balanced_take(negatives, negative_count, seed + 1))
    output: list[dict] = []
    for index, row in enumerate(selected):
        item = dict(row)
        item["selection_index"] = index
        item["m1_source_role"] = item["label_type"]
        output.append(item)
    return output


def safe_component(value: str, max_length: int = 72) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return (cleaned or "sample")[:max_length]


def build_capture_plan(
    selected: Sequence[dict], distances: Sequence[str] = DISTANCES, seed: int = 20260526
) -> list[dict]:
    plan: list[dict] = []
    for distance_index, distance in enumerate(distances):
        shuffled = list(selected)
        random.Random(seed + distance_index).shuffle(shuffled)
        for sequence, row in enumerate(shuffled, start=1):
            label = str(row["label_type"])
            source_sample_id = str(row["sample_id"])
            capture_id = (
                f"m1_{safe_component(distance)}_{safe_component(label)}_"
                f"{sequence:04d}_{safe_component(source_sample_id)}"
            )
            item = dict(row)
            item.update(
                {
                    "capture_id": capture_id,
                    "source_sample_id": source_sample_id,
                    "source_path": row["path"],
                    "distance": distance,
                    "distance_m": float(distance.rstrip("m")),
                    "layout_id": "layout_M1_A",
                    "status": "pending",
                }
            )
            plan.append(item)
    return plan


def read_pcm16(path: Path) -> array:
    samples = array("h")
    with path.open("rb") as stream:
        samples.frombytes(stream.read())
    if samples.itemsize != 2:
        raise ValueError("PCM reader requires 16-bit samples")
    return samples


def write_pcm16(path: Path, samples: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = array("h", samples)
    with path.open("wb") as stream:
        stream.write(values.tobytes())


def pcm_stats(path: Path) -> dict:
    samples = read_pcm16(path)
    count = len(samples)
    if count == 0:
        return {"samples": 0, "duration_sec": 0.0, "rms": 0.0, "peak": 0, "clipping_ratio": 0.0}
    sum_squares = sum(int(value) * int(value) for value in samples)
    peak = max(abs(int(value)) for value in samples)
    clipped = sum(1 for value in samples if abs(int(value)) >= 30000)
    return {
        "samples": count,
        "duration_sec": count / float(SAMPLE_RATE),
        "rms": math.sqrt(sum_squares / float(count)),
        "peak": peak,
        "clipping_ratio": clipped / float(count),
    }


def validate_pcm_capture(
    path: Path,
    expected_duration_sec: float,
    duration_tolerance_sec: float = 0.3,
    minimum_rms: float = 20.0,
    max_clipping_ratio: float = 0.01,
) -> dict:
    stats = pcm_stats(path)
    reasons: list[str] = []
    if stats["samples"] == 0 or stats["rms"] < minimum_rms:
        reasons.append("silent")
    if abs(stats["duration_sec"] - expected_duration_sec) > duration_tolerance_sec:
        reasons.append("duration")
    if stats["clipping_ratio"] > max_clipping_ratio:
        reasons.append("clipping")
    stats["valid"] = not reasons
    stats["reasons"] = reasons
    return stats


def convert_pcm16_to_wav(pcm_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm_path.read_bytes())


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() <= 0:
            return 0.0
        return audio.getnframes() / float(audio.getframerate())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upsert_record(path: Path, record: dict) -> None:
    existing = read_jsonl(path) if path.exists() else []
    by_id = {item["capture_id"]: item for item in existing}
    by_id[record["capture_id"]] = record
    ordered = [by_id[key] for key in sorted(by_id)]
    write_jsonl(path, ordered)
