from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import winsound
from datetime import datetime
from pathlib import Path

from anju_kws.collection.m1_common import read_jsonl, wav_duration, write_json, write_jsonl


PACKAGE = "cn.org.wenet.wekws"
MODEL_IDS = ("m1_rawonly_finetune", "m1_rawonly_pretrain_full")
WAKEUP_RE = re.compile(r"WAKEUP .*score=([0-9.]+)")
SCORE_RE = re.compile(r"score=([0-9.]+)")


class Adb:
    def __init__(self, executable: str, serial: str) -> None:
        self.executable = executable
        self.serial = serial

    def run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [self.executable, "-s", self.serial, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=check,
        )

    def shell(self, command: str, check: bool = True) -> str:
        return self.run("shell", command, check=check).stdout


def play_wav(path: Path) -> None:
    if sys.platform != "win32":
        raise RuntimeError("This playback runner currently uses Windows winsound.")
    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def set_model_pref(adb: Adb, model_id: str) -> None:
    prefs = (
        "<?xml version='1.0' encoding='utf-8' standalone='yes' ?>\n"
        "<map>\n"
        f"    <string name=\"selected_model_id\">{model_id}</string>\n"
        "</map>\n"
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".xml", delete=False) as handle:
        handle.write(prefs)
        local = Path(handle.name)
    try:
        adb.run("push", str(local), "/data/local/tmp/anju_kws.xml")
        adb.shell(f"run-as {PACKAGE} mkdir -p shared_prefs")
        adb.shell(f"run-as {PACKAGE} cp /data/local/tmp/anju_kws.xml shared_prefs/anju_kws.xml")
    finally:
        local.unlink(missing_ok=True)


def launch_app(adb: Adb, model_id: str) -> None:
    adb.shell(f"am force-stop {PACKAGE}")
    set_model_pref(adb, model_id)
    adb.shell(f"monkey -p {PACKAGE} -c android.intent.category.LAUNCHER 1 >/dev/null")
    time.sleep(2.0)


def tap_text(adb: Adb, text: str) -> bool:
    adb.shell("uiautomator dump /sdcard/anju_window.xml >/dev/null")
    xml = adb.shell("cat /sdcard/anju_window.xml", check=False)
    pattern = re.compile(r'text="' + re.escape(text) + r'".*?bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"')
    match = pattern.search(xml)
    if not match:
        return False
    left, top, right, bottom = [int(value) for value in match.groups()]
    adb.shell(f"input tap {(left + right) // 2} {(top + bottom) // 2}")
    return True


def fallback_tap_bottom_left(adb: Adb) -> None:
    size_text = adb.shell("wm size", check=False)
    match = re.search(r"(\d+)x(\d+)", size_text)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
        adb.shell(f"input tap {width // 4} {max(1, height - 70)}")
    else:
        adb.shell("input tap 480 625")


def start_listening(adb: Adb) -> None:
    if not tap_text(adb, "开始监听"):
        fallback_tap_bottom_left(adb)
    time.sleep(1.0)


def stop_listening(adb: Adb) -> None:
    if not tap_text(adb, "停止监听"):
        fallback_tap_bottom_left(adb)
    time.sleep(1.0)


def capture_logcat(adb: Adb, output_path: Path, stop_event: threading.Event) -> subprocess.Popen:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = output_path.open("w", encoding="utf-8", newline="\n")
    proc = subprocess.Popen(
        [adb.executable, "-s", adb.serial, "logcat", "-v", "epoch", "-s", "WEKWS", "WEKWS_NATIVE"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    def reader() -> None:
        assert proc.stdout is not None
        with handle:
            for line in proc.stdout:
                handle.write(line)
                handle.flush()
                if stop_event.is_set():
                    break

    threading.Thread(target=reader, daemon=True).start()
    return proc


def parse_logcat(log_path: Path) -> list[dict]:
    events: list[dict] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            timestamp = float(parts[0])
        except ValueError:
            continue
        wake = WAKEUP_RE.search(line)
        score = SCORE_RE.search(line)
        if wake or score:
            events.append(
                {
                    "timestamp": timestamp,
                    "wakeup": bool(wake),
                    "score": float((wake or score).group(1)),
                    "line": line,
                }
            )
    return events


def evaluate(events: list[dict], playbacks: list[dict], pre_margin: float, post_margin: float) -> dict:
    rows: list[dict] = []
    used_wakeup_indices: set[int] = set()
    for playback in playbacks:
        start = playback["play_start_epoch"] - pre_margin
        end = playback["play_end_epoch"] + post_margin
        window = [event for event in events if start <= event["timestamp"] <= end]
        wakeups = [(i, event) for i, event in enumerate(events) if event["wakeup"] and start <= event["timestamp"] <= end]
        for index, _ in wakeups:
            used_wakeup_indices.add(index)
        max_score = max([event["score"] for event in window], default=0.0)
        rows.append(
            {
                **playback,
                "detected": bool(wakeups),
                "max_score": max_score,
                "wakeup_count": len(wakeups),
            }
        )
    positives = [row for row in rows if row["expected_wakeup"]]
    negatives = [row for row in rows if not row["expected_wakeup"]]
    detected_positives = [row for row in positives if row["detected"]]
    false_positive_negatives = [row for row in negatives if row["detected"]]
    unmatched = [
        event for index, event in enumerate(events) if event["wakeup"] and index not in used_wakeup_indices
    ]
    return {
        "attempts": rows,
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "positive_detected": len(detected_positives),
        "negative_false_trigger": len(false_positive_negatives),
        "recall": len(detected_positives) / max(1, len(positives)),
        "false_trigger_rate_on_negatives": len(false_positive_negatives) / max(1, len(negatives)),
        "unmatched_wakeup_count": len(unmatched),
        "max_score_overall": max([event["score"] for event in events], default=0.0),
    }


def run_one_model(adb: Adb, model_id: str, plan: list[dict], output_dir: Path, gap_sec: float) -> dict:
    model_dir = output_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    adb.run("logcat", "-c")
    launch_app(adb, model_id)
    log_path = model_dir / "logcat.txt"
    stop_event = threading.Event()
    proc = capture_logcat(adb, log_path, stop_event)
    playbacks: list[dict] = []
    try:
        start_listening(adb)
        time.sleep(gap_sec)
        for index, item in enumerate(plan, start=1):
            path = Path(item["source_path"])
            print(f"[{model_id}] {index}/{len(plan)} {item['test_id']} {item['label_type']} {path.name}")
            play_start = time.time()
            play_wav(path)
            play_end = time.time()
            playbacks.append(
                {
                    **item,
                    "play_start_epoch": play_start,
                    "play_end_epoch": play_end,
                    "play_start_local": datetime.fromtimestamp(play_start).isoformat(timespec="milliseconds"),
                    "play_end_local": datetime.fromtimestamp(play_end).isoformat(timespec="milliseconds"),
                    "source_duration_sec": wav_duration(path),
                }
            )
            time.sleep(gap_sec)
        stop_listening(adb)
        time.sleep(1.0)
    finally:
        stop_event.set()
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    write_jsonl(model_dir / "playback_events.jsonl", playbacks)
    events = parse_logcat(log_path)
    summary = evaluate(events, playbacks, pre_margin=0.3, post_margin=1.8)
    summary.update({"model_id": model_id, "logcat": str(log_path), "event_count": len(events)})
    write_json(model_dir / "summary.json", summary)
    write_jsonl(model_dir / "attempt_results.jsonl", summary["attempts"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed playback A/B tests for two Android WeKWS models.")
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adb-serial", required=True)
    parser.add_argument("--adb-path", default="adb")
    parser.add_argument("--models", nargs="+", default=list(MODEL_IDS))
    parser.add_argument("--gap-sec", type=float, default=2.5)
    args = parser.parse_args()

    plan = read_jsonl(args.test_plan)
    adb = Adb(args.adb_path, args.adb_serial)
    state = adb.run("get-state").stdout.strip()
    if state != "device":
        raise RuntimeError(f"ADB target is not online: {args.adb_serial} state={state!r}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "run_config.json",
        {
            "test_plan": str(args.test_plan),
            "adb_serial": args.adb_serial,
            "models": args.models,
            "gap_sec": args.gap_sec,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    summaries = [run_one_model(adb, model_id, plan, args.output_dir, args.gap_sec) for model_id in args.models]
    write_json(args.output_dir / "summary.json", {"summaries": summaries})
    print(json.dumps({"output_dir": str(args.output_dir), "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
