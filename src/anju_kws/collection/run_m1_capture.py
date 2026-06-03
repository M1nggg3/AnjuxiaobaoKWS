from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import winsound
from pathlib import Path

from .m1_common import (
    convert_pcm16_to_wav,
    read_jsonl,
    sha256_file,
    upsert_record,
    validate_pcm_capture,
    wav_duration,
)


PACKAGE = "cn.org.wenet.wekws"
SERVICE = f"{PACKAGE}/.FarfieldCaptureService"
ACTION_PREFIX = f"{PACKAGE}.action."
REMOTE_ROOT = f"/storage/emulated/0/Android/data/{PACKAGE}/files"


class AdbClient:
    def __init__(self, executable: str, serial: str) -> None:
        self.executable = executable
        self.serial = serial

    def command(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        command = [self.executable, "-s", self.serial, *args]
        return subprocess.run(command, capture_output=True, text=True, check=check)

    def service(self, action: str, extras: dict[str, str] | None = None) -> None:
        command = [
            "shell",
            "am",
            "start-foreground-service",
            "-n",
            SERVICE,
            "-a",
            ACTION_PREFIX + action,
        ]
        for key, value in (extras or {}).items():
            command.extend(["--es", key, str(value)])
        self.command(*command)

    def pull(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self.command("pull", remote, str(local))


def resolve_adb(requested: str | None) -> str:
    candidates = [
        requested,
        os.environ.get("ADB_PATH"),
        str(Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe"),
        "adb",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if candidate == "adb" or Path(candidate).exists():
            return candidate
    raise FileNotFoundError("adb.exe was not found; pass --adb-path")


def play_wav(path: Path, playback_device: str | None) -> None:
    if playback_device:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as error:
            raise RuntimeError(
                "--playback-device requires sounddevice; install it or omit the option for the Windows default device"
            ) from error
        samples, sample_rate = sf.read(str(path), dtype="float32")
        sd.play(samples, sample_rate, device=playback_device, blocking=True)
        return
    if sys.platform != "win32":
        raise RuntimeError("Default playback backend is Windows winsound; use --playback-device with sounddevice")
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def preflight(adb: AdbClient) -> None:
    state = adb.command("get-state").stdout.strip()
    if state != "device":
        raise RuntimeError(f"ADB target is not online: {adb.serial} state={state!r}")
    storage = adb.command("shell", "df -k /sdcard | tail -n 1").stdout.strip()
    parts = storage.split()
    if len(parts) >= 4 and parts[3].isdigit() and int(parts[3]) < 100 * 1024:
        raise RuntimeError(f"Board free storage is below 100 MB: {storage}")


def list_recoverable_statuses(adb: AdbClient, local_path: Path) -> list[str]:
    adb.service("COLLECT_RECOVER")
    time.sleep(0.2)
    try:
        adb.pull(f"{REMOTE_ROOT}/status/recover.json", local_path)
    except subprocess.CalledProcessError:
        return []
    value = json.loads(local_path.read_text(encoding="utf-8"))
    return list(value.get("entries", []))


def read_status(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def wait_for_completed_status(adb: AdbClient, capture_id: str, local_status: Path, timeout_sec: float = 8.0) -> dict:
    remote_status = f"{REMOTE_ROOT}/status/{capture_id}.json"
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            adb.pull(remote_status, local_status)
            status = read_status(local_status)
            if status.get("state") in {"complete", "failed"}:
                return status
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for completed board status: {capture_id}")


def artifact_dir(experiment_dir: Path, row: dict) -> Path:
    if row.get("collection_phase") == "smoke":
        return experiment_dir / "qc" / "smoke_captures" / row["distance"] / row["label_type"] / row["capture_id"]
    return experiment_dir / "captures" / row["distance"] / row["label_type"] / row["capture_id"]


def select_capture_batch(plan: list[dict], distance: str, smoke: bool) -> list[dict]:
    selected = [dict(row) for row in plan if row["distance"] == distance]
    if not smoke:
        for row in selected:
            row["collection_phase"] = "formal"
        return selected
    positives = [row for row in selected if row["label_type"] == "positive"][:5]
    negatives = [row for row in selected if row["label_type"] == "negative"][:3]
    output = positives + negatives
    for row in output:
        row["capture_id"] = "smoke_" + row["capture_id"]
        row["collection_phase"] = "smoke"
    return output


def has_remote_status(adb: AdbClient, capture_id: str) -> bool:
    remote_status = f"{REMOTE_ROOT}/status/{capture_id}.json"
    result = adb.command("shell", f"test -f {remote_status} && echo present", check=False)
    return "present" in result.stdout


def pull_validate_and_finalize(
    adb: AdbClient,
    experiment_dir: Path,
    row: dict,
    pre_roll_sec: float,
    post_roll_sec: float,
    duration_tolerance_sec: float,
) -> dict:
    capture_id = row["capture_id"]
    output = artifact_dir(experiment_dir, row)
    output.mkdir(parents=True, exist_ok=True)
    for legacy_artifact in ("enhanced_16k_s16le.pcm", "enhanced.wav"):
        (output / legacy_artifact).unlink(missing_ok=True)
    source_path = Path(row["source_path"])
    raw_pcm = output / "raw_16k_s16le.pcm"
    log_path = output / "session.log"
    status_path = output / "status.json"
    adb.pull(f"{REMOTE_ROOT}/status/{capture_id}.json", status_path)
    status = read_status(status_path)
    remote_raw_pcm = status.get("raw_pcm") or f"{REMOTE_ROOT}/captures/{capture_id}_raw_16k_s16le.pcm"
    remote_log = status.get("log") or f"{REMOTE_ROOT}/logs/{capture_id}.log"
    adb.pull(remote_raw_pcm, raw_pcm)
    adb.pull(remote_log, log_path)
    raw_wav = output / "raw.wav"
    convert_pcm16_to_wav(raw_pcm, raw_wav)
    expected_duration = wav_duration(source_path) + pre_roll_sec + post_roll_sec
    raw_quality = validate_pcm_capture(raw_pcm, expected_duration, duration_tolerance_sec=duration_tolerance_sec)
    record = dict(row)
    record.update(
        {
            "adb_serial": adb.serial,
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
            "retry_count": row.get("retry_count", 0),
        }
    )
    if record["status"] == "verified":
        adb.service("COLLECT_DELETE", {"capture_id": capture_id})
    return record


def recover_existing_capture(
    adb: AdbClient,
    experiment_dir: Path,
    row: dict,
    pre_roll_sec: float,
    post_roll_sec: float,
    duration_tolerance_sec: float,
) -> dict:
    output = artifact_dir(experiment_dir, row)
    status_path = output / "status.json"
    adb.pull(f"{REMOTE_ROOT}/status/{row['capture_id']}.json", status_path)
    status = read_status(status_path)
    if status.get("state") in {"recording", "stopping"}:
        adb.service("COLLECT_STOP", {"capture_id": row["capture_id"]})
        status = wait_for_completed_status(adb, row["capture_id"], status_path)
    if status.get("state") != "complete":
        result = dict(row)
        result.update({"status": "retake", "error": f"Cannot recover board state: {status}"})
        return result
    return pull_validate_and_finalize(adb, experiment_dir, row, pre_roll_sec, post_roll_sec, duration_tolerance_sec)


def capture_one(
    adb: AdbClient,
    experiment_dir: Path,
    row: dict,
    playback_device: str | None,
    pre_roll_sec: float,
    post_roll_sec: float,
    duration_tolerance_sec: float = 0.5,
) -> dict:
    capture_id = row["capture_id"]
    output = artifact_dir(experiment_dir, row)
    output.mkdir(parents=True, exist_ok=True)
    source_path = Path(row["source_path"])
    record = dict(row)
    record.update({"adb_serial": adb.serial, "connection_mode": "wifi_adb", "retry_count": row.get("retry_count", 0)})
    capture_started = False
    try:
        adb.service(
            "COLLECT_START",
            {
                "capture_id": capture_id,
                "label": row["label_type"],
                "distance_m": str(row["distance_m"]),
                "source_sample_id": row["source_sample_id"],
            },
        )
        capture_started = True
        time.sleep(pre_roll_sec)
        play_wav(source_path, playback_device)
        time.sleep(post_roll_sec)
        adb.service("COLLECT_STOP", {"capture_id": capture_id})
        status_path = output / "status.json"
        board_status = wait_for_completed_status(adb, capture_id, status_path)
        if board_status.get("state") != "complete":
            raise RuntimeError(f"Board capture failed: {board_status}")
        record = pull_validate_and_finalize(
            adb,
            experiment_dir,
            row,
            pre_roll_sec,
            post_roll_sec,
            duration_tolerance_sec,
        )
    except Exception as error:
        if capture_started:
            try:
                adb.service("COLLECT_STOP", {"capture_id": capture_id})
            except Exception as stop_error:
                record["stop_error"] = str(stop_error)
        record.update({"status": "awaiting_reconnect", "error": str(error)})
    return record


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one M1 physical far-field collection batch over Wi-Fi ADB.")
    result.add_argument("--experiment-dir", type=Path, required=True)
    result.add_argument("--distance", choices=["1m", "3m", "5m"], required=True)
    result.add_argument("--adb-serial", default="192.168.3.228:44891")
    result.add_argument("--adb-path", default=None)
    result.add_argument("--playback-device", default=None)
    result.add_argument("--pre-roll-sec", type=float, default=0.5)
    result.add_argument("--post-roll-sec", type=float, default=0.5)
    result.add_argument("--duration-tolerance-sec", type=float, default=0.5)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    plan_path = args.experiment_dir / "protocol" / "capture_plan.jsonl"
    records_path = (
        args.experiment_dir / "qc" / "smoke_recordings.jsonl"
        if args.smoke
        else args.experiment_dir / "manifests" / "recordings.jsonl"
    )
    plan = select_capture_batch(read_jsonl(plan_path), args.distance, args.smoke)
    environment_path = args.experiment_dir / "protocol" / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8-sig")) if environment_path.exists() else {}
    for row in plan:
        row.setdefault("layout_id", environment.get("layout_id", "layout_M1_A"))
        row["playback_device"] = args.playback_device or environment.get("playback_device", "Windows default output device")
        row["volume_percent"] = environment.get("playback_volume_percent", 70)
    recorded = {row["capture_id"]: row for row in read_jsonl(records_path)} if records_path.exists() else {}
    if args.resume:
        plan = [row for row in plan if recorded.get(row["capture_id"], {}).get("status") != "verified"]
    if args.dry_run:
        print(json.dumps({"distance": args.distance, "pending_count": len(plan), "capture_ids": [row["capture_id"] for row in plan]}, ensure_ascii=False, indent=2))
        return

    adb = AdbClient(resolve_adb(args.adb_path), args.adb_serial)
    preflight(adb)
    recoverable = list_recoverable_statuses(adb, args.experiment_dir / "qc" / "board_recovery_index.json")
    if recoverable:
        print(f"Board contains recoverable status files: {recoverable}")
    for sequence, row in enumerate(plan, start=1):
        print(f"[{sequence}/{len(plan)}] {row['capture_id']} source={row['source_path']}")
        if has_remote_status(adb, row["capture_id"]):
            print("  recovering board-resident capture")
            record = recover_existing_capture(
                adb,
                args.experiment_dir,
                row,
                args.pre_roll_sec,
                args.post_roll_sec,
                args.duration_tolerance_sec,
            )
        else:
            record = capture_one(
                adb,
                args.experiment_dir,
                row,
                args.playback_device,
                args.pre_roll_sec,
                args.post_roll_sec,
                args.duration_tolerance_sec,
            )
        upsert_record(records_path, record)
        print(f"  status={record['status']}")
        if record["status"] == "awaiting_reconnect":
            raise SystemExit("Capture paused after ADB or board failure; reconnect and rerun with --resume")


if __name__ == "__main__":
    main()
