from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .m1_common import (
    DISTANCES,
    build_capture_plan,
    read_jsonl,
    select_source_samples,
    write_json,
    write_jsonl,
)
from .prepare_m2_parallel_capture_plan import device_config_template


DEFAULT_MANIFEST = Path(
    r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data"
    r"\anju_xiaobao_cosyvoice3_clean_purewake_20260520_155613"
    r"\manifests\all_clean.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\eval"
) / ("m2_parallel_30pos10neg_" + dt.datetime.now().strftime("%Y%m%d"))
DEFAULT_EXCLUDE_RECORDINGS = [
    Path(
        r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\experiments"
        r"\physical_farfield_m1_training_20260526\manifests\recordings.jsonl"
    ),
    Path(
        r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\experiments"
        r"\physical_farfield_m2_parallel_positive_20260601\manifests\recordings.jsonl"
    ),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a three-board 30 positive + 10 near-negative evaluation capture plan."
    )
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positive-count", type=int, default=30)
    parser.add_argument("--negative-count", type=int, default=10)
    parser.add_argument(
        "--exclude-recordings",
        type=Path,
        action="append",
        default=None,
        help="Existing recordings.jsonl whose source_sample_id values must be excluded.",
    )
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--volume-percent", type=int, default=70)
    parser.add_argument("--playback-device", default="Windows default output device")
    return parser


def excluded_source_ids(paths: list[Path]) -> set[str]:
    output: set[str] = set()
    for path in paths:
        if path.exists():
            output.update(
                str(row.get("source_sample_id", ""))
                for row in read_jsonl(path)
                if row.get("source_sample_id")
            )
    return output


def add_eval_playback_ids(plan: list[dict]) -> list[dict]:
    output: list[dict] = []
    grouped: dict[str, int] = {}
    for row in plan:
        source_sample_id = str(row["source_sample_id"])
        if source_sample_id not in grouped:
            grouped[source_sample_id] = len(grouped) + 1
        playback_index = grouped[source_sample_id]
        label = str(row["label_type"])
        item = dict(row)
        item["playback_index"] = playback_index
        item["playback_id"] = f"m2_eval_playback_{playback_index:04d}_{source_sample_id}"
        item["capture_id"] = (
            f"m2_eval_{item['distance']}_{label}_{playback_index:04d}_{source_sample_id}"
        )
        item["collection_phase"] = "eval"
        output.append(item)
    return output


def write_readme(output_dir: Path, positive_count: int, negative_count: int) -> None:
    (output_dir / "README.md").write_text(
        "# M2 三板 30 正 + 10 负远场验证采集\n\n"
        "本目录用于保存一轮小型闭环验证数据。本机每次播放一条源音频，"
        "1m、3m、5m 三台 RK3566 同步录制 raw-only 音频。\n\n"
        f"- 正样本源音频：{positive_count} 条\n"
        f"- 近音负样本源音频：{negative_count} 条\n"
        f"- 总播放次数：{positive_count + negative_count} 次\n"
        f"- 总录音数量：{(positive_count + negative_count) * len(DISTANCES)} 条\n"
        "- 训练/评估输入：仅使用质检通过的 raw.wav\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    exclude_paths = args.exclude_recordings or DEFAULT_EXCLUDE_RECORDINGS
    excluded = excluded_source_ids(exclude_paths)
    rows = read_jsonl(args.input_manifest)
    if excluded:
        rows = [row for row in rows if str(row.get("sample_id", "")) not in excluded]

    selected = select_source_samples(rows, args.positive_count, args.negative_count, args.seed)
    plan = add_eval_playback_ids(build_capture_plan(selected, DISTANCES, args.seed))

    selected_path = args.output_dir / "source_selection" / "selected_eval_samples.jsonl"
    plan_path = args.output_dir / "protocol" / "parallel_capture_plan.jsonl"
    environment_path = args.output_dir / "protocol" / "environment.json"
    devices_template_path = args.output_dir / "protocol" / "devices.template.json"
    devices_path = args.output_dir / "protocol" / "devices.json"

    write_jsonl(selected_path, selected)
    write_jsonl(plan_path, plan)
    devices_template = device_config_template(args.volume_percent, args.playback_device)
    write_json(devices_template_path, devices_template)
    if not devices_path.exists():
        write_json(devices_path, devices_template)
    write_json(
        environment_path,
        {
            "experiment": "m2_parallel_30pos10neg",
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "input_manifest": str(args.input_manifest),
            "exclude_recordings": [str(path) for path in exclude_paths],
            "excluded_source_count": len(excluded),
            "seed": args.seed,
            "positive_source_count": args.positive_count,
            "negative_source_count": args.negative_count,
            "distances": list(DISTANCES),
            "target_capture_count": len(plan),
            "capture_mode": "raw_only",
            "layout_id": "layout_M2_parallel_eval",
            "playback_volume_percent": args.volume_percent,
            "playback_device": args.playback_device,
            "connection_mode": "wifi_adb",
        },
    )
    write_readme(args.output_dir, args.positive_count, args.negative_count)

    summary = {
        "output_dir": str(args.output_dir),
        "excluded_source_count": len(excluded),
        "selected_positive_count": sum(row["label_type"] == "positive" for row in selected),
        "selected_negative_count": sum(row["label_type"] == "negative" for row in selected),
        "playback_count": len({row["playback_id"] for row in plan}),
        "capture_plan_count": len(plan),
        "by_distance": {distance: sum(row["distance"] == distance for row in plan) for distance in DISTANCES},
        "by_label": {
            "positive": sum(row["label_type"] == "positive" for row in plan),
            "negative": sum(row["label_type"] == "negative" for row in plan),
        },
        "plan": str(plan_path),
        "devices": str(devices_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
