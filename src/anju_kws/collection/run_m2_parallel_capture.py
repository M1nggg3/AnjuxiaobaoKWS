from __future__ import annotations

import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .m1_common import (
    convert_pcm16_to_wav,
    read_jsonl,
    sha256_file,
    upsert_record,
    validate_pcm_capture,
    wav_duration,
)
from .run_m1_capture import (
    REMOTE_ROOT,
    AdbClient,
    play_wav,
    preflight,
    read_status,
    resolve_adb,
    wait_for_completed_status,
)


DISTANCES = ("1m", "3m", "5m")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run three RK3566 boards in parallel for far-field capture.")
    result.add_argument("--experiment-dir", type=Path, required=True)
    result.add_argument("--adb-path", default=None)
    result.add_argument("--playback-device", default=None)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--start-index", type=int, default=1, help="1-based playback index to start from.")
    result.add_argument("--limit", type=int, default=None, help="Maximum playback items to process.")
    return result


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_devices(experiment_dir: Path, adb_path: str | None, require_serial: bool = True) -> tuple[list[dict], dict]:
    devices_path = experiment_dir / "protocol" / "devices.json"
    if not devices_path.exists():
        raise FileNotFoundError(f"Missing devices.json: {devices_path}")
    config = load_json(devices_path)
    devices = list(config.get("devices", []))
    by_distance = {item.get("distance"): item for item in devices}
    missing = [distance for distance in DISTANCES if distance not in by_distance]
    if missing:
        raise ValueError(f"devices.json missing distance entries: {missing}")
    for distance in DISTANCES:
        serial = str(by_distance[distance].get("adb_serial", "")).strip()
        if require_serial and not serial:
            raise ValueError(f"devices.json has empty adb_serial for {distance}")
        by_distance[distance]["adb"] = AdbClient(resolve_adb(adb_path), serial) if serial else None
    return [by_distance[distance] for distance in DISTANCES], config


def artifact_dir(experiment_dir: Path, row: dict) -> Path:
    label = str(row.get("label_type", "positive"))
    if row.get("collection_phase") == "smoke":
        return experiment_dir / "qc" / "smoke_captures" / row["distance"] / label / row["capture_id"]
    return experiment_dir / "captures" / row["distance"] / label / row["capture_id"]


def capture_id_for(row: dict, phase: str) -> str:
    prefix = "smoke_" if phase == "smoke" else ""
    return prefix + str(row["capture_id"])


def group_plan(plan: list[dict]) -> list[dict]:
    by_playback: dict[str, dict] = {}
    for row in plan:
        playback_id = str(row["playback_id"])
        entry = by_playback.setdefault(
            playback_id,
            {
                "playback_id": playback_id,
                "playback_index": int(row["playback_index"]),
                "source_path": row["source_path"],
                "source_sample_id": row["source_sample_id"],
                "label_type": row.get("label_type", "positive"),
                "rows": {},
            },
        )
        entry["rows"][row["distance"]] = dict(row)
    output = [value for _, value in sorted(by_playback.items(), key=lambda item: item[1]["playback_index"])]
    for entry in output:
        missing = [distance for distance in DISTANCES if distance not in entry["rows"]]
        if missing:
            raise ValueError(f"Playback {entry['playback_id']} missing distances: {missing}")
    return output


def select_playbacks(playbacks: list[dict], smoke: bool, start_index: int, limit: int | None) -> list[dict]:
    selected = playbacks[:10] if smoke else playbacks
    selected = [item for item in selected if item["playback_index"] >= start_index]
    if limit is not None:
        selected = selected[:limit]
    phase = "smoke" if smoke else "formal"
    for playback in selected:
        for row in playback["rows"].values():
            row["collection_phase"] = phase
            row["capture_id"] = capture_id_for(row, phase)
    return selected


def start_capture(device: dict, row: dict) -> None:
    adb: AdbClient = device["adb"]
    adb.service(
        "COLLECT_START",
        {
            "capture_id": row["capture_id"],
            "label": str(row.get("label_type", "positive")),
            "distance_m": str(row["distance_m"]),
            "source_sample_id": row["source_sample_id"],
        },
    )


def stop_capture(device: dict, capture_id: str) -> None:
    adb: AdbClient = device["adb"]
    adb.service("COLLECT_STOP", {"capture_id": capture_id})


def pull_validate_and_finalize(
    device: dict,
    experiment_dir: Path,
    row: dict,
    pre_roll_sec: float,
    post_roll_sec: float,
    duration_tolerance_sec: float,
) -> dict:
    adb: AdbClient = device["adb"]
    capture_id = row["capture_id"]
    output = artifact_dir(experiment_dir, row)
    output.mkdir(parents=True, exist_ok=True)

    source_path = Path(row["source_path"])
    status_path = output / "status.json"
    raw_pcm = output / "raw_16k_s16le.pcm"
    raw_wav = output / "raw.wav"
    log_path = output / "session.log"

    adb.pull(f"{REMOTE_ROOT}/status/{capture_id}.json", status_path)
    status = read_status(status_path)
    remote_raw_pcm = status.get("raw_pcm") or f"{REMOTE_ROOT}/captures/{capture_id}_raw_16k_s16le.pcm"
    remote_log = status.get("log") or f"{REMOTE_ROOT}/logs/{capture_id}.log"
    adb.pull(remote_raw_pcm, raw_pcm)
    adb.pull(remote_log, log_path)
    convert_pcm16_to_wav(raw_pcm, raw_wav)

    expected_duration = wav_duration(source_path) + pre_roll_sec + post_roll_sec
    raw_quality = validate_pcm_capture(raw_pcm, expected_duration, duration_tolerance_sec=duration_tolerance_sec)
    record = dict(row)
    record.update(
        {
            "adb_serial": adb.serial,
            "device_name": device.get("device_name", ""),
            "connection_mode": "wifi_adb",
            "capture_mode": status.get("capture_mode", "raw_only"),
            "raw_wav_path": str(raw_wav),
            "training_wav_path": str(raw_wav),
            "log_path": str(log_path),
            "duration_sec": raw_quality["duration_sec"],
            "raw_rms": raw_quality["rms"],
            "raw_peak": raw_quality["peak"],
            "clipping_ratio": raw_quality["clipping_ratio"],
            "sha256": sha256_file(raw_wav),
            "quality_reasons": raw_quality["reasons"],
            "status": "verified" if raw_quality["valid"] else "retake",
        }
    )
    if record["status"] == "verified":
        adb.service("COLLECT_DELETE", {"capture_id": capture_id})
    return record


def run_for_all_devices(function, devices: list[dict], rows: dict[str, dict]) -> list[object]:
    results: list[object] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = []
        for device in devices:
            row = rows[str(device["distance"])]
            futures.append(executor.submit(function, device, row))
        for future in as_completed(futures):
            results.append(future.result())
    return results


def start_all_devices(devices: list[dict], rows: dict[str, dict]) -> list[tuple[dict, str]]:
    started: list[tuple[dict, str]] = []
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {}
        for device in devices:
            row = rows[str(device["distance"])]
            futures[executor.submit(start_capture, device, row)] = (device, row)
        for future in as_completed(futures):
            device, row = futures[future]
            try:
                future.result()
                started.append((device, row["capture_id"]))
            except Exception as error:
                errors[str(device["distance"])] = str(error)
    if errors:
        for device, capture_id in started:
            try:
                stop_capture(device, capture_id)
            except Exception:
                pass
        raise RuntimeError(f"COLLECT_START failed: {errors}")
    return started


def process_playback(
    devices: list[dict],
    experiment_dir: Path,
    playback: dict,
    records_path: Path,
    playback_device: str | None,
    pre_roll_sec: float,
    post_roll_sec: float,
    duration_tolerance_sec: float,
) -> list[dict]:
    rows = playback["rows"]
    source_path = Path(playback["source_path"])
    started: list[tuple[dict, str]] = []
    try:
        started = start_all_devices(devices, rows)
        time.sleep(pre_roll_sec)
        play_wav(source_path, playback_device)
        time.sleep(post_roll_sec)
    except Exception as error:
        for device, capture_id in started:
            try:
                stop_capture(device, capture_id)
            except Exception:
                pass
        records = []
        for row in rows.values():
            record = dict(row)
            record.update({"status": "awaiting_reconnect", "error": str(error)})
            records.append(record)
            upsert_record(records_path, record)
        return records

    stop_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {
            executor.submit(stop_capture, device, rows[str(device["distance"])]["capture_id"]): str(device["distance"])
            for device in devices
        }
        for future in as_completed(futures):
            distance = futures[future]
            try:
                future.result()
            except Exception as error:
                stop_errors[distance] = str(error)

    records: list[dict] = []
    for device in devices:
        distance = str(device["distance"])
        row = rows[distance]
        if distance in stop_errors:
            record = dict(row)
            record.update({"status": "awaiting_reconnect", "error": stop_errors[distance]})
        else:
            try:
                status_path = artifact_dir(experiment_dir, row) / "status.json"
                board_status = wait_for_completed_status(device["adb"], row["capture_id"], status_path)
                if board_status.get("state") != "complete":
                    raise RuntimeError(f"Board capture failed: {board_status}")
                record = pull_validate_and_finalize(
                    device,
                    experiment_dir,
                    row,
                    pre_roll_sec,
                    post_roll_sec,
                    duration_tolerance_sec,
                )
            except Exception as error:
                record = dict(row)
                record.update({"status": "awaiting_reconnect", "error": str(error)})
        records.append(record)
        upsert_record(records_path, record)
    return records


def main() -> None:
    args = parser().parse_args()
    plan_path = args.experiment_dir / "protocol" / "parallel_capture_plan.jsonl"
    records_path = (
        args.experiment_dir / "qc" / "smoke_recordings.jsonl"
        if args.smoke
        else args.experiment_dir / "manifests" / "recordings.jsonl"
    )
    devices, device_config = load_devices(args.experiment_dir, args.adb_path, require_serial=not args.dry_run)
    playbacks = select_playbacks(group_plan(read_jsonl(plan_path)), args.smoke, args.start_index, args.limit)
    recorded = {row["capture_id"]: row for row in read_jsonl(records_path)} if records_path.exists() else {}
    if args.resume:
        playbacks = [
            playback
            for playback in playbacks
            if any(recorded.get(row["capture_id"], {}).get("status") != "verified" for row in playback["rows"].values())
        ]
    if args.dry_run:
        print(
            json.dumps(
                {
                    "playback_count": len(playbacks),
                    "capture_count": sum(len(playback["rows"]) for playback in playbacks),
                    "devices": [
                        {
                            "distance": device["distance"],
                            "adb_serial": device["adb"].serial if device.get("adb") else "",
                            "device_name": device.get("device_name", ""),
                        }
                        for device in devices
                    ],
                    "first_playback_ids": [playback["playback_id"] for playback in playbacks[:10]],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    for device in devices:
        preflight(device["adb"])

    playback_device = args.playback_device or device_config.get("playback_device")
    if playback_device == "Windows default output device":
        playback_device = None
    pre_roll_sec = float(device_config.get("pre_roll_sec", 0.5))
    post_roll_sec = float(device_config.get("post_roll_sec", 0.5))
    duration_tolerance_sec = float(device_config.get("duration_tolerance_sec", 0.5))

    for sequence, playback in enumerate(playbacks, start=1):
        print(f"[{sequence}/{len(playbacks)}] {playback['playback_id']} source={playback['source_path']}")
        records = process_playback(
            devices,
            args.experiment_dir,
            playback,
            records_path,
            playback_device,
            pre_roll_sec,
            post_roll_sec,
            duration_tolerance_sec,
        )
        status_by_distance = {record["distance"]: record["status"] for record in records}
        print("  " + json.dumps(status_by_distance, ensure_ascii=False))
        if any(record["status"] == "awaiting_reconnect" for record in records):
            raise SystemExit("Parallel capture paused after ADB or board failure; reconnect and rerun with --resume")


if __name__ == "__main__":
    main()
