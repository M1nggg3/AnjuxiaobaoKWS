from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .m1_common import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize M1 capture results and export raw training manifests.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    record_path = args.experiment_dir / "manifests" / "recordings.jsonl"
    rows = read_jsonl(record_path) if record_path.exists() else []
    verified = [row for row in rows if row.get("status") == "verified"]
    for row in verified:
        row.setdefault("training_wav_path", row.get("raw_wav_path", ""))
        row.setdefault("capture_mode", "raw_only")
    positive = [row for row in verified if row["label_type"] == "positive"]
    negative = [row for row in verified if row["label_type"] == "negative"]
    write_jsonl(args.experiment_dir / "manifests" / "training_raw_positive.jsonl", positive)
    write_jsonl(args.experiment_dir / "manifests" / "training_raw_negative.jsonl", negative)
    failed = [row for row in rows if row.get("status") != "verified"]
    write_jsonl(args.experiment_dir / "qc" / "failed_or_retake.jsonl", failed)
    summary = {
        "record_count": len(rows),
        "verified_count": len(verified),
        "failed_or_retake_count": len(failed),
        "by_status": dict(Counter(row.get("status", "unknown") for row in rows)),
        "by_distance_and_label": dict(
            Counter(f"{row['distance']}:{row['label_type']}" for row in verified)
        ),
        "by_gender": dict(Counter(row.get("gender", "unknown") for row in verified)),
        "by_source_dataset": dict(Counter(row.get("source_dataset", "unknown") for row in verified)),
    }
    write_json(args.experiment_dir / "qc" / "capture_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
