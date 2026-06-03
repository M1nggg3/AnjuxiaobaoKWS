import json
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anju_kws.collection.m1_common import (  # noqa: E402
    build_capture_plan,
    select_source_samples,
    validate_pcm_capture,
    write_pcm16,
)
from anju_kws.collection.run_m1_capture import (  # noqa: E402
    REMOTE_ROOT,
    capture_one,
    play_wav,
    recover_existing_capture,
    select_capture_batch,
)


def sample_row(sample_id: str, label: str, text: str, gender: str, source: str) -> dict:
    return {
        "sample_id": sample_id,
        "label_type": label,
        "text": text,
        "gender": gender,
        "source_dataset": source,
        "path": f"C:/{sample_id}.wav",
        "duration_sec": 1.2,
    }


class SourceSelectionTest(unittest.TestCase):
    def test_selects_balanced_positives_and_excludes_homophone_negatives(self) -> None:
        rows = []
        for index in range(8):
            gender = "male" if index % 2 == 0 else "female"
            source = "data_aishell" if index < 4 else "data_aishell3"
            rows.append(sample_row(f"pos_{index}", "positive", "安居小宝。", gender, source))
        rows.extend(
            [
                sample_row("neg_ok_1", "negative", "安静小宝", "male", "data_aishell"),
                sample_row("neg_ok_2", "negative", "你好小宝", "female", "data_aishell3"),
                sample_row("neg_bad_1", "negative", "安居小包", "male", "data_aishell"),
                sample_row("neg_bad_2", "negative", "安居小饱", "female", "data_aishell3"),
                sample_row("neg_ordinary", "negative", "请把资料发给我", "male", "data_aishell"),
            ]
        )

        selected = select_source_samples(rows, positive_count=4, negative_count=2, seed=13)

        positive = [row for row in selected if row["label_type"] == "positive"]
        negative = [row for row in selected if row["label_type"] == "negative"]
        self.assertEqual(4, len(positive))
        self.assertEqual({"male", "female"}, {row["gender"] for row in positive})
        self.assertEqual({"data_aishell", "data_aishell3"}, {row["source_dataset"] for row in positive})
        self.assertEqual({"neg_ok_1", "neg_ok_2"}, {row["sample_id"] for row in negative})

    def test_plan_reuses_source_set_for_each_distance_with_unique_capture_ids(self) -> None:
        selected = [
            sample_row("pos_a", "positive", "安居小宝", "male", "data_aishell"),
            sample_row("neg_a", "negative", "安静小宝", "female", "data_aishell3"),
        ]

        plan = build_capture_plan(selected, distances=("1m", "2m", "3m"), seed=7)

        self.assertEqual(6, len(plan))
        for distance in ("1m", "2m", "3m"):
            source_ids = {row["source_sample_id"] for row in plan if row["distance"] == distance}
            self.assertEqual({"pos_a", "neg_a"}, source_ids)
        self.assertEqual(6, len({row["capture_id"] for row in plan}))

    def test_smoke_batch_uses_isolated_capture_identifiers(self) -> None:
        selected = [
            sample_row(f"pos_{index}", "positive", "安居小宝", "male", "data_aishell")
            for index in range(5)
        ] + [
            sample_row(f"neg_{index}", "negative", "安静小宝", "female", "data_aishell3")
            for index in range(3)
        ]
        plan = build_capture_plan(selected, distances=("1m",), seed=9)

        batch = select_capture_batch(plan, "1m", smoke=True)

        self.assertEqual(8, len(batch))
        self.assertTrue(all(item["capture_id"].startswith("smoke_") for item in batch))
        self.assertTrue(all(item["collection_phase"] == "smoke" for item in batch))

    def test_ordinary_speech_is_not_selected_as_near_negative(self) -> None:
        rows = [
            sample_row("pos", "positive", "安居小宝", "male", "data_aishell"),
            sample_row("near", "negative", "安静小宝", "female", "data_aishell3"),
            sample_row("ordinary", "negative", "请把资料发给我", "male", "data_aishell"),
        ]

        with self.assertRaises(ValueError):
            select_source_samples(rows, positive_count=1, negative_count=2, seed=3)


class AudioQualityTest(unittest.TestCase):
    def test_validation_accepts_non_silent_pcm_and_rejects_silence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            good = Path(temp_dir) / "good.pcm"
            silent = Path(temp_dir) / "silent.pcm"
            samples = [1500 if index % 2 == 0 else -1500 for index in range(16000 * 2)]
            write_pcm16(good, samples)
            write_pcm16(silent, [0] * (16000 * 2))

            result = validate_pcm_capture(good, expected_duration_sec=2.0)
            silent_result = validate_pcm_capture(silent, expected_duration_sec=2.0)

            self.assertTrue(result["valid"])
            self.assertGreater(result["rms"], 1000)
            self.assertFalse(silent_result["valid"])
            self.assertIn("silent", silent_result["reasons"])


class RecoveryTest(unittest.TestCase):
    def test_windows_default_playback_uses_valid_winsound_flags(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            wav_path = Path(temp_dir) / "source.wav"
            with wave.open(str(wav_path), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(array("h", [0] * 160).tobytes())

            with patch("anju_kws.collection.run_m1_capture.winsound.PlaySound") as play_sound:
                play_wav(wav_path, None)

            play_sound.assert_called_once()
            self.assertEqual(str(wav_path), play_sound.call_args.args[0])

    def test_recovered_verified_capture_is_deleted_without_replaying(self) -> None:
        class FakeAdb:
            serial = "192.168.3.228:44891"

            def __init__(self, remote_files: dict[str, Path]) -> None:
                self.remote_files = remote_files
                self.services: list[tuple[str, dict | None]] = []

            def pull(self, remote: str, local: Path) -> None:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(self.remote_files[remote].read_bytes())

            def service(self, action: str, extras: dict | None = None) -> None:
                self.services.append((action, extras))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_wav = root / "source.wav"
            source_signal = array("h", [1200, -1200] * 8000)
            signal = array("h", [1200, -1200] * 16000)
            with wave.open(str(source_wav), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(source_signal.tobytes())
            raw_pcm = root / "remote_raw.pcm"
            write_pcm16(raw_pcm, signal)
            log_file = root / "remote.log"
            log_file.write_text("complete\n", encoding="utf-8")
            status_file = root / "remote.json"
            capture_id = "m1_1m_positive_0001_pos_a"
            status_file.write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "capture_mode": "raw_only",
                        "raw_pcm": f"{REMOTE_ROOT}/captures/{capture_id}_raw_16k_s16le.pcm",
                        "log": f"{REMOTE_ROOT}/logs/{capture_id}.log",
                    }
                ),
                encoding="utf-8",
            )
            adb = FakeAdb(
                {
                    f"{REMOTE_ROOT}/captures/{capture_id}_raw_16k_s16le.pcm": raw_pcm,
                    f"{REMOTE_ROOT}/logs/{capture_id}.log": log_file,
                    f"{REMOTE_ROOT}/status/{capture_id}.json": status_file,
                }
            )
            row = {
                "capture_id": capture_id,
                "source_sample_id": "pos_a",
                "source_path": str(source_wav),
                "label_type": "positive",
                "distance": "1m",
                "distance_m": 1.0,
            }

            with patch("anju_kws.collection.run_m1_capture.play_wav") as playback:
                recovered = recover_existing_capture(adb, root / "experiment", row, 0.5, 0.5, 0.5)

            playback.assert_not_called()
            self.assertEqual("verified", recovered["status"])
            self.assertEqual("raw_only", recovered["capture_mode"])
            self.assertEqual(recovered["raw_wav_path"], recovered["training_wav_path"])
            self.assertNotIn("enhanced_wav_path", recovered)
            self.assertEqual("COLLECT_DELETE", adb.services[-1][0])

    def test_new_capture_is_verified_after_playback_and_deleted(self) -> None:
        class FakeAdb:
            serial = "192.168.3.228:44891"

            def __init__(self, remote_files: dict[str, Path]) -> None:
                self.remote_files = remote_files
                self.services: list[tuple[str, dict | None]] = []

            def pull(self, remote: str, local: Path) -> None:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(self.remote_files[remote].read_bytes())

            def service(self, action: str, extras: dict | None = None) -> None:
                self.services.append((action, extras))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source_wav = root / "source.wav"
            source_signal = array("h", [1000, -1000] * 8000)
            captured_signal = array("h", [1000, -1000] * 16000)
            with wave.open(str(source_wav), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(source_signal.tobytes())
            raw_pcm = root / "remote_raw.pcm"
            write_pcm16(raw_pcm, captured_signal)
            log_file = root / "remote.log"
            log_file.write_text("complete\n", encoding="utf-8")
            status_file = root / "remote.json"
            status_file.write_text('{"state":"complete"}', encoding="utf-8")
            capture_id = "m1_1m_positive_0001_pos_b"
            adb = FakeAdb(
                {
                    f"{REMOTE_ROOT}/captures/{capture_id}_raw_16k_s16le.pcm": raw_pcm,
                    f"{REMOTE_ROOT}/logs/{capture_id}.log": log_file,
                    f"{REMOTE_ROOT}/status/{capture_id}.json": status_file,
                }
            )
            row = {
                "capture_id": capture_id,
                "source_sample_id": "pos_b",
                "source_path": str(source_wav),
                "label_type": "positive",
                "distance": "1m",
                "distance_m": 1.0,
            }

            with patch("anju_kws.collection.run_m1_capture.play_wav") as playback:
                record = capture_one(adb, root / "experiment", row, None, 0.5, 0.5)

            playback.assert_called_once_with(source_wav, None)
            self.assertEqual("verified", record["status"])
            self.assertEqual("raw_only", record["capture_mode"])
            self.assertEqual(record["raw_wav_path"], record["training_wav_path"])
            self.assertNotIn("enhanced_wav_path", record)
            self.assertEqual("COLLECT_DELETE", adb.services[-1][0])

    def test_failed_playback_stops_board_capture_before_returning(self) -> None:
        class FakeAdb:
            serial = "192.168.3.228:44891"

            def __init__(self) -> None:
                self.services: list[tuple[str, dict | None]] = []

            def pull(self, remote: str, local: Path) -> None:
                raise AssertionError("pull should not happen after playback failure")

            def service(self, action: str, extras: dict | None = None) -> None:
                self.services.append((action, extras))

        with tempfile.TemporaryDirectory() as temp_dir:
            source_wav = Path(temp_dir) / "source.wav"
            with wave.open(str(source_wav), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(array("h", [0] * 160).tobytes())
            row = {
                "capture_id": "smoke_m1_1m_positive_0001_pos_failure",
                "source_sample_id": "pos_failure",
                "source_path": str(source_wav),
                "label_type": "positive",
                "distance": "1m",
                "distance_m": 1.0,
            }
            adb = FakeAdb()

            with patch("anju_kws.collection.run_m1_capture.play_wav", side_effect=RuntimeError("playback failed")):
                record = capture_one(adb, Path(temp_dir) / "experiment", row, None, 0.0, 0.0)

            self.assertEqual("awaiting_reconnect", record["status"])
            self.assertEqual("COLLECT_START", adb.services[0][0])
            self.assertIn(("COLLECT_STOP", {"capture_id": row["capture_id"]}), adb.services)


if __name__ == "__main__":
    unittest.main()
