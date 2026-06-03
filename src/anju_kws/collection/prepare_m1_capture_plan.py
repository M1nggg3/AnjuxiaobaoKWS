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


def default_output_dir() -> Path:
    stamp = dt.datetime.now().strftime("%Y%m%d")
    return DEFAULT_EXPERIMENT_ROOT / f"physical_farfield_m1_training_{stamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the fixed M1 physical far-field capture plan.")
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--positive-count", type=int, default=100)
    parser.add_argument("--negative-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--adb-serial", default="192.168.3.228:44891")
    parser.add_argument("--volume-percent", type=int, default=70)
    parser.add_argument("--playback-device", default="Windows default output device")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir or default_output_dir()
    rows = read_jsonl(args.input_manifest)
    selected = select_source_samples(rows, args.positive_count, args.negative_count, args.seed)
    plan = build_capture_plan(selected, DISTANCES, args.seed)

    selected_path = output_dir / "source_selection" / "selected_playback_samples.jsonl"
    plan_path = output_dir / "protocol" / "capture_plan.jsonl"
    environment_path = output_dir / "protocol" / "environment.json"
    write_jsonl(selected_path, selected)
    write_jsonl(plan_path, plan)
    write_json(
        environment_path,
        {
            "experiment": "physical_farfield_m1_training",
            "created_at": dt.datetime.now().isoformat(timespec="seconds"),
            "input_manifest": str(args.input_manifest),
            "seed": args.seed,
            "positive_count_per_distance": args.positive_count,
            "negative_count_per_distance": args.negative_count,
            "distances": list(DISTANCES),
            "layout_id": "layout_M1_A",
            "playback_volume_percent": args.volume_percent,
            "playback_device": args.playback_device,
            "adb_serial": args.adb_serial,
            "connection_mode": "wifi_adb",
        },
    )
    (output_dir / "README.md").write_text(
        "# M1 物理远场训练数据采集\n\n"
        "本目录用于保存通过笔记本扬声器播放 clean TTS、由 RK3566 麦克风物理录制的远场训练数据。\n\n"
        f"- 输入 manifest：`{args.input_manifest}`\n"
        f"- 无线 ADB：`{args.adb_serial}`\n"
        f"- 播放设备：`{args.playback_device}`\n"
        f"- 初始音量：`{args.volume_percent}%`\n"
        "- 布局：`layout_M1_A`\n"
        "- 距离：`1m`、`3m`、`5m`\n"
        f"- 每档计划：正样本 `{args.positive_count}` 条，近音负样本 `{args.negative_count}` 条\n"
        "- 训练输入：仅使用校验通过的 `raw.wav`；日志用于质检。\n\n"
        "正式采集前先完成 1m 冒烟批次并人工抽听所有样本。冒烟音频写入 `qc/smoke_captures`，"
        "不会计入正式训练 manifest。\n",
        encoding="utf-8",
    )
    summary = {
        "output_dir": str(output_dir),
        "selected_source_count": len(selected),
        "capture_plan_count": len(plan),
        "positive_source_count": sum(item["label_type"] == "positive" for item in selected),
        "negative_source_count": sum(item["label_type"] == "negative" for item in selected),
        "plan": str(plan_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
