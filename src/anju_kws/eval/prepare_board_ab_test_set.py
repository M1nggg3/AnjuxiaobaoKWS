from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from anju_kws.collection.m1_common import read_jsonl, write_json, write_jsonl


def balanced_take(rows: list[dict], count: int, seed: int) -> list[dict]:
    if len(rows) < count:
        raise ValueError(f"requested {count} rows but only {len(rows)} are available")
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("gender", "unknown")), str(row.get("source_dataset", "unknown")))
        buckets.setdefault(key, []).append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = sorted(buckets)
    rng.shuffle(keys)
    selected: list[dict] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) < count:
        remaining = [row for bucket in buckets.values() for row in bucket]
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return selected


def build_plan(rows: list[dict], positive_count: int, negative_count: int, seed: int) -> list[dict]:
    positives = [row for row in rows if row.get("label_type") == "positive"]
    negatives = [row for row in rows if row.get("label_type") == "negative"]
    selected = balanced_take(positives, positive_count, seed)
    selected.extend(balanced_take(negatives, negative_count, seed + 1))
    random.Random(seed + 2).shuffle(selected)
    plan: list[dict] = []
    for index, row in enumerate(selected, start=1):
        source_path = Path(str(row.get("path", "")))
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        plan.append(
            {
                "test_id": f"board_ab_{index:03d}_{row['label_type']}_{row['sample_id']}",
                "sample_id": row["sample_id"],
                "label_type": row["label_type"],
                "expected_wakeup": row["label_type"] == "positive",
                "text": row.get("text", ""),
                "source_path": str(source_path),
                "source_dataset": row.get("source_dataset", ""),
                "speaker": row.get("speaker", ""),
                "gender": row.get("gender", "unknown"),
                "selection_index": row.get("selection_index"),
            }
        )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a fixed board A/B playback test set.")
    parser.add_argument(
        "--source-selection",
        type=Path,
        default=Path(
            r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\experiments"
            r"\physical_farfield_m1_training_20260526\source_selection"
            r"\selected_playback_samples.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\eval\board_ab_farfield_20260529"),
    )
    parser.add_argument("--positive-count", type=int, default=12)
    parser.add_argument("--negative-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260529)
    args = parser.parse_args()

    rows = read_jsonl(args.source_selection)
    plan = build_plan(rows, args.positive_count, args.negative_count, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "test_plan.jsonl", plan)
    write_json(
        args.output_dir / "summary.json",
        {
            "source_selection": str(args.source_selection),
            "test_plan": str(args.output_dir / "test_plan.jsonl"),
            "positive_count": args.positive_count,
            "negative_count": args.negative_count,
            "seed": args.seed,
            "total": len(plan),
        },
    )
    print(json.dumps({"output_dir": str(args.output_dir), "total": len(plan)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
