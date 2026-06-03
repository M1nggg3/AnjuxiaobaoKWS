from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from .m1_common import DISTANCES, build_capture_plan, read_jsonl, select_source_samples, write_json, write_jsonl


DEFAULT_MANIFEST = Path(
    r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\data"
    r"\anju_xiaobao_cosyvoice3_clean_purewake_20260520_155613"
    r"\manifests\all_clean.jsonl"
)
DEFAULT_EXPERIMENT_ROOT = Path(r"E:\CodeWorking\datasets\AnJuXiaoBaoKWS\experiments")
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_ROOT / "physical_farfield_m2_parallel_positive_20260601"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a three-board parallel far-field positive capture plan."
    )
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--positive-count", type=int, default=1000)
    parser.add_argument(
        "--exclude-recordings",
        type=Path,
        action="append",
        default=None,
        help="Existing recordings.jsonl whose source_sample_id values must be excluded.",
    )
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--volume-percent", type=int, default=70)
    parser.add_argument("--playback-device", default="Windows default output device")
    return parser


def excluded_source_ids(paths: list[Path] | None) -> set[str]:
    output: set[str] = set()
    for path in paths or []:
        if path.exists():
            output.update(
                str(row.get("source_sample_id", "")) for row in read_jsonl(path) if row.get("source_sample_id")
            )
    return output


def add_playback_ids(plan: list[dict]) -> list[dict]:
    output: list[dict] = []
    grouped: dict[str, int] = {}
    for row in plan:
        source_sample_id = str(row["source_sample_id"])
        if source_sample_id not in grouped:
            grouped[source_sample_id] = len(grouped) + 1
        playback_index = grouped[source_sample_id]
        item = dict(row)
        item["playback_index"] = playback_index
        item["playback_id"] = f"m2_playback_{playback_index:04d}_{source_sample_id}"
        item["capture_id"] = f"m2_{item['distance']}_positive_{playback_index:04d}_{source_sample_id}"
        item["collection_phase"] = "formal"
        output.append(item)
    return output


def device_config_template(volume_percent: int, playback_device: str) -> dict:
    return {
        "devices": [
            {"distance": "1m", "adb_serial": "", "device_name": "rk3566_1m"},
            {"distance": "3m", "adb_serial": "", "device_name": "rk3566_3m"},
            {"distance": "5m", "adb_serial": "", "device_name": "rk3566_5m"},
        ],
        "playback_device": playback_device,
        "playback_volume_percent": volume_percent,
        "pre_roll_sec": 0.5,
        "post_roll_sec": 0.5,
        "duration_tolerance_sec": 0.5,
    }


def write_readme(output_dir: Path, input_manifest: Path, positive_count: int) -> None:
    (output_dir / "README.md").write_text(
        "# M2 三板同步远场正样本采集\n\n"
        "本目录用于保存三台 RK3566 同步录制的真实物理远场正样本。"
        "本机每次只播放一条 clean TTS 唤醒词音频，1m、3m、5m 三台板端同时录制 raw-only 音频。\n\n"
        f"- 输入 manifest：`{input_manifest}`\n"
        f"- 源音频数量：`{positive_count}` 条 positive\n"
        "- 距离：`1m`、`3m`、`5m`\n"
        f"- 目标录音数量：`{positive_count * len(DISTANCES)}` 条\n"
        "- 训练输入：仅使用质检通过的 `raw.wav`\n"
        "- 使用前请复制或填写 `protocol/devices.json` 中三台板端的 Wi-Fi ADB serial。\n",
        encoding="utf-8",
    )


def main() -> None:
    args = build_parser().parse_args()
    excluded = excluded_source_ids(args.exclude_recordings)
    rows = read_jsonl(args.input_manifest)
    if excluded:
        rows = [row for row in rows if str(row.get("sample_id", "")) not in excluded]
    selected = select_source_samples(rows, args.positive_count, 0, args.seed)
    plan = add_playback_ids(build_capture_plan(selected, DISTANCES, args.seed))

    selected_path = args.output_dir / "source_selection" / "selected_positive_samples.jsonl"
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
            "experiment": "physical_farfield_m2_parallel_positive",
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "input_manifest": str(args.input_manifest),
            "exclude_recordings": [str(path) for path in args.exclude_recordings or []],
            "excluded_source_count": len(excluded),
            "seed": args.seed,
            "positive_source_count": args.positive_count,
            "negative_source_count": 0,
            "distances": list(DISTANCES),
            "target_capture_count": len(plan),
            "capture_mode": "raw_only",
            "layout_id": "layout_M2_parallel",
            "playback_volume_percent": args.volume_percent,
            "playback_device": args.playback_device,
            "connection_mode": "wifi_adb",
        },
    )
    write_readme(args.output_dir, args.input_manifest, args.positive_count)

    summary = {
        "output_dir": str(args.output_dir),
        "excluded_source_count": len(excluded),
        "selected_positive_count": len(selected),
        "playback_count": len({row["playback_id"] for row in plan}),
        "capture_plan_count": len(plan),
        "by_distance": {distance: sum(row["distance"] == distance for row in plan) for distance in DISTANCES},
        "plan": str(plan_path),
        "devices": str(devices_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
