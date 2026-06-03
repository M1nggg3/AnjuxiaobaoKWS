from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .m1_common import read_jsonl, write_json, write_jsonl


def metric_summary(rows: list[dict], field: str) -> dict:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    if not values:
        return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "avg": sum(values) / len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize M2 parallel capture results.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()

    record_path = args.experiment_dir / "manifests" / "recordings.jsonl"
    rows = read_jsonl(record_path) if record_path.exists() else []
    verified = [row for row in rows if row.get("status") == "verified"]
    for row in verified:
        row.setdefault("training_wav_path", row.get("raw_wav_path", ""))
        row.setdefault("capture_mode", "raw_only")

    positive = [row for row in verified if row.get("label_type") == "positive"]
    negative = [row for row in verified if row.get("label_type") == "negative"]
    failed = [row for row in rows if row.get("status") != "verified"]
    write_jsonl(args.experiment_dir / "manifests" / "training_raw_positive.jsonl", positive)
    write_jsonl(args.experiment_dir / "manifests" / "training_raw_negative.jsonl", negative)
    write_jsonl(args.experiment_dir / "manifests" / "training_raw_all.jsonl", positive + negative)
    write_jsonl(args.experiment_dir / "qc" / "failed_or_retake.jsonl", failed)

    by_distance_rows: dict[str, list[dict]] = defaultdict(list)
    for row in verified:
        by_distance_rows[str(row.get("distance", "unknown"))].append(row)

    summary = {
        "record_count": len(rows),
        "verified_count": len(verified),
        "training_raw_positive_count": len(positive),
        "training_raw_negative_count": len(negative),
        "training_raw_all_count": len(positive) + len(negative),
        "failed_or_retake_count": len(failed),
        "by_status": dict(Counter(row.get("status", "unknown") for row in rows)),
        "by_distance": dict(Counter(row.get("distance", "unknown") for row in verified)),
        "by_distance_and_label": dict(
            Counter(f"{row.get('distance', 'unknown')}:{row.get('label_type', 'unknown')}" for row in verified)
        ),
        "by_gender": dict(Counter(row.get("gender", "unknown") for row in verified)),
        "by_source_dataset": dict(Counter(row.get("source_dataset", "unknown") for row in verified)),
        "metrics_by_distance": {
            distance: {
                "duration_sec": metric_summary(distance_rows, "duration_sec"),
                "raw_rms": metric_summary(distance_rows, "raw_rms"),
                "raw_peak": metric_summary(distance_rows, "raw_peak"),
            }
            for distance, distance_rows in sorted(by_distance_rows.items())
        },
    }
    write_json(args.experiment_dir / "qc" / "capture_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
