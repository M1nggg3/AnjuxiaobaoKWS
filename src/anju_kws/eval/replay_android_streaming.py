"""Replay an RK3566 Android listening session with the native WeKWS pipeline.

This tool intentionally follows the current Android C++ runtime constants and
state transitions.  The device currently enqueues enhanced PCM for inference,
so ``enhanced`` is the default board-equivalent input and ``raw`` is emitted as
an A/B reference.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi


SAMPLE_RATE = 16000
CHUNK_SAMPLES = 640
READ_FRAMES = 80
FBANK_DIM = 80
CONTEXT_LEFT = 2
CONTEXT_RIGHT = 2
FRAME_SKIP = 3
BLANK_ID = 0
FILLER_ID = 5
KEYWORD_IDS = (1, 2, 3, 4)
SCORE_BEAM_SIZE = 5
PATH_BEAM_SIZE = 30
SCORE_PRUNE_THRESHOLD = 0.02
MIN_FRAMES = 3
MAX_FRAMES = 250
INTERVAL_FRAMES = 50
DEFAULT_SPEECH_RMS_THRESHOLD = 70.0
DEFAULT_SPEECH_PEAK_THRESHOLD = 500
DEFAULT_SILENCE_CHUNKS_BEFORE_RESET = 20
DEFAULT_SOFT_RESET_INTERVAL_CHUNKS = 50

TOKEN_NAMES = {
    0: "<blk>",
    1: "an",
    2: "ju",
    3: "xiao",
    4: "bao",
    5: "<filler>",
}

RESULT_PATTERN = re.compile(
    r"^(?P<clock>\d{2}:\d{2}:\d{2}\.\d{3}) "
    r"(?P<state>WAKEUP keyword=\S+|listening)"
    r" score=(?P<score>[\d.]+) threshold=(?P<threshold>[\d.]+)"
    r" frame=(?P<frame>\d+) infer_ms=(?P<infer_ms>[\d.]+)"
)
EPOCH_RESULT_PATTERN = re.compile(
    r"^\s*(?P<epoch>\d+\.\d+)\s+\d+\s+\d+\s+I\s+WEKWS\s+:\s+"
    r"(?P<state>WAKEUP keyword=\S+|listening)"
    r" score=(?P<score>[\d.]+) threshold=(?P<threshold>[\d.]+)"
    r" frame=(?P<frame>\d+) infer_ms=(?P<infer_ms>[\d.]+)"
)


@dataclass
class Node:
    token: int
    frame: int
    prob: float


@dataclass
class Hyp:
    prefix: tuple[int, ...]
    pb: float
    pnb: float
    nodes: list[Node] = field(default_factory=list)


@dataclass
class Detection:
    has_keyword: bool = False
    wakeup: bool = False
    start: int = 0
    end: int = 0
    score: float = 0.0


def read_pcm(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype="<i2")
    if data.size == 0:
        raise ValueError(f"PCM input is empty: {path}")
    return data


def pcm_rms_peak(chunk: np.ndarray) -> tuple[float, int]:
    values = chunk.astype(np.float64, copy=False)
    rms = float(np.sqrt(np.mean(values * values))) if values.size else 0.0
    peak = int(np.max(np.abs(chunk.astype(np.int32)))) if chunk.size else 0
    return rms, peak


def is_speech_chunk(chunk: np.ndarray, rms_threshold: float, peak_threshold: int) -> bool:
    rms, peak = pcm_rms_peak(chunk)
    return rms >= rms_threshold or peak >= peak_threshold


def parse_clock(clock: str, base: Optional[datetime], previous: Optional[datetime]) -> datetime:
    current = datetime.strptime(clock, "%H:%M:%S.%f")
    if base is not None:
        current = current.replace(year=base.year, month=base.month, day=base.day)
    if previous is not None and current < previous:
        current += timedelta(days=1)
    return current


def parse_board_log(log_path: Optional[Path]) -> dict[str, Any]:
    timeline: list[dict[str, Any]] = []
    base_time: Optional[datetime] = None
    previous: Optional[datetime] = None
    base_epoch: Optional[float] = None
    session_meta: dict[str, Any] = {}
    if log_path is None or not log_path.exists():
        return {
            "available": False,
            "session_meta": {"reason": "Android session log not available."},
            "timeline": [],
            "events": [],
        }
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        capture_match = re.search(r"^\s*(\d+\.\d+).*audio_raw_capture path=", line)
        if capture_match and base_epoch is None:
            base_epoch = float(capture_match.group(1))
            session_meta["audio_raw_capture_epoch"] = base_epoch
            session_meta["audio_raw_capture_line"] = line
            continue
        epoch_match = EPOCH_RESULT_PATTERN.match(line)
        if epoch_match:
            epoch = float(epoch_match.group("epoch"))
            if base_epoch is None:
                base_epoch = epoch
            timeline.append(
                {
                    "source": "board",
                    "time_sec": round(epoch - base_epoch, 3),
                    "wakeup": epoch_match.group("state").startswith("WAKEUP"),
                    "score": float(epoch_match.group("score")),
                    "threshold": float(epoch_match.group("threshold")),
                    "frame": int(epoch_match.group("frame")),
                    "infer_ms": float(epoch_match.group("infer_ms")),
                }
            )
            continue
        if "session_start" in line:
            clock = line[:12]
            base_time = parse_clock(clock, None, None)
            previous = base_time
            session_meta["session_start_line"] = line
            threshold_match = re.search(r"threshold=([\d.]+)", line)
            if threshold_match:
                session_meta["threshold"] = float(threshold_match.group(1))
        match = RESULT_PATTERN.match(line)
        if not match:
            continue
        moment = parse_clock(match.group("clock"), base_time, previous)
        if base_time is None:
            base_time = moment
        previous = moment
        timeline.append(
            {
                "source": "board",
                "time_sec": round((moment - base_time).total_seconds(), 3),
                "wakeup": match.group("state").startswith("WAKEUP"),
                "score": float(match.group("score")),
                "threshold": float(match.group("threshold")),
                "frame": int(match.group("frame")),
                "infer_ms": float(match.group("infer_ms")),
            }
        )
    events = [row for row in timeline if row["wakeup"]]
    return {"available": True, "session_meta": session_meta, "timeline": timeline, "events": events}


class NativeEquivalentReplay:
    """Port of ``runtime/android/app/src/main/cpp/wekws.cc`` for one PCM input."""

    def __init__(
        self,
        model_path: Path,
        threshold: float,
        provider: str = "cpu",
        speech_rms_threshold: float = DEFAULT_SPEECH_RMS_THRESHOLD,
        speech_peak_threshold: int = DEFAULT_SPEECH_PEAK_THRESHOLD,
        silence_chunks_before_reset: int = DEFAULT_SILENCE_CHUNKS_BEFORE_RESET,
        soft_reset_interval_chunks: int = DEFAULT_SOFT_RESET_INTERVAL_CHUNKS,
    ) -> None:
        providers = ["CPUExecutionProvider"]
        if provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.threshold = threshold
        self.model_path = model_path
        self.providers = self.session.get_providers()
        self.speech_rms_threshold = speech_rms_threshold
        self.speech_peak_threshold = speech_peak_threshold
        self.silence_chunks_before_reset = silence_chunks_before_reset
        self.soft_reset_interval_chunks = soft_reset_interval_chunks
        self.reset_all()

    def reset_all(self) -> None:
        self.cache = np.zeros((1, 128, 11, 4), dtype=np.float32)
        self.total_frames = 0
        self.last_active_frame = -1
        self.feats_ctx_offset = 0
        self.feature_remained: list[np.ndarray] = []
        self.wave_remained = np.zeros((0,), dtype=np.int16)
        self.feature_queue: list[np.ndarray] = []
        self.cur_hyps = [Hyp((), 1.0, 0.0, [])]
        self.consecutive_silence_chunks = 0
        self.stream_chunk_index = 0
        self.chunks_since_stream_reset = 0
        self.recent_keyword_score = 0.0
        self.recent_keyword_end_frame = -1

    def reset_streaming_state(self) -> None:
        self.cache = np.zeros((1, 128, 11, 4), dtype=np.float32)
        self.total_frames = 0
        self.last_active_frame = -1
        self.feats_ctx_offset = 0
        self.feature_remained = []
        self.wave_remained = np.zeros((0,), dtype=np.int16)
        self.feature_queue = []
        self.cur_hyps = [Hyp((), 1.0, 0.0, [])]

    def reset_decoder(self) -> None:
        self.cur_hyps = [Hyp((), 1.0, 0.0, [])]

    def accept_waveform(self, chunk: np.ndarray) -> Optional[dict[str, Any]]:
        self.stream_chunk_index += 1
        self.chunks_since_stream_reset += 1
        rms, peak = pcm_rms_peak(chunk)
        speech = is_speech_chunk(chunk, self.speech_rms_threshold, self.speech_peak_threshold)
        reset: Optional[dict[str, Any]] = None
        if speech:
            if self.consecutive_silence_chunks >= self.silence_chunks_before_reset:
                reset_silence_chunks = self.consecutive_silence_chunks
                self.reset_streaming_state()
                self.chunks_since_stream_reset = 0
                reset = {
                    "type": "speech_onset_reset",
                    "chunk": self.stream_chunk_index,
                    "rms": round(rms, 3),
                    "peak": peak,
                    "silence_chunks": reset_silence_chunks,
                }
            self.consecutive_silence_chunks = 0
        else:
            self.consecutive_silence_chunks += 1
            if (
                self.soft_reset_interval_chunks > 0
                and self.chunks_since_stream_reset >= self.soft_reset_interval_chunks
            ):
                reset_silence_chunks = self.consecutive_silence_chunks
                self.reset_streaming_state()
                self.chunks_since_stream_reset = 0
                self.consecutive_silence_chunks = 0
                reset = {
                    "type": "soft_reset",
                    "chunk": self.stream_chunk_index,
                    "rms": round(rms, 3),
                    "peak": peak,
                    "silence_chunks": reset_silence_chunks,
                }

        wave = np.concatenate((self.wave_remained, chunk))
        if wave.size >= 400:
            tensor = torch.from_numpy(wave.astype(np.float32, copy=False)).unsqueeze(0)
            feats = kaldi.fbank(
                tensor,
                num_mel_bins=FBANK_DIM,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                energy_floor=0.0,
                sample_frequency=SAMPLE_RATE,
            ).cpu().numpy()
            self.feature_queue.extend(feats)
            self.wave_remained = wave[feats.shape[0] * 160 :]
        else:
            self.wave_remained = wave
        return reset

    def expand_and_skip(self, feats: list[np.ndarray]) -> np.ndarray:
        if not feats:
            return np.zeros((0, FBANK_DIM * 5), dtype=np.float32)
        if not self.feature_remained:
            padded = [feats[0], feats[0], *feats]
        else:
            padded = [*self.feature_remained, *feats]
        ctx_frames = len(padded) - (CONTEXT_RIGHT + CONTEXT_RIGHT)
        expanded = [
            np.concatenate(padded[i : i + CONTEXT_LEFT + CONTEXT_RIGHT + 1])
            for i in range(max(0, ctx_frames))
        ]
        self.feature_remained = feats[-(CONTEXT_LEFT + CONTEXT_RIGHT) :]
        if not expanded:
            return np.zeros((0, FBANK_DIM * 5), dtype=np.float32)
        last_remainder = 0 if self.feats_ctx_offset == 0 else FRAME_SKIP - self.feats_ctx_offset
        remainder = (len(expanded) + last_remainder) % FRAME_SKIP
        skipped = expanded[self.feats_ctx_offset :: FRAME_SKIP]
        self.feats_ctx_offset = 0 if remainder == 0 else FRAME_SKIP - remainder
        return np.asarray(skipped, dtype=np.float32)

    @staticmethod
    def _keyword_token(token: int) -> bool:
        return token == BLANK_ID or token == FILLER_ID or token in KEYWORD_IDS

    def ctc_prefix_beam_search(self, frame: int, probs: np.ndarray) -> None:
        ranked = sorted(((float(prob), token) for token, prob in enumerate(probs)), reverse=True)
        filtered = [
            token
            for prob, token in ranked[:SCORE_BEAM_SIZE]
            if prob > SCORE_PRUNE_THRESHOLD and self._keyword_token(token)
        ]
        if not filtered:
            return

        next_hyps: dict[tuple[int, ...], Hyp] = {}

        def state(prefix: tuple[int, ...]) -> Hyp:
            if prefix not in next_hyps:
                next_hyps[prefix] = Hyp(prefix, 0.0, 0.0, [])
            return next_hyps[prefix]

        for token in filtered:
            ps = float(probs[token])
            for hyp in self.cur_hyps:
                last = hyp.prefix[-1] if hyp.prefix else -1
                if token == BLANK_ID:
                    out = state(hyp.prefix)
                    out.pb += hyp.pb * ps + hyp.pnb * ps
                    out.nodes = [Node(n.token, n.frame, n.prob) for n in hyp.nodes]
                elif token == last:
                    if not math.isclose(hyp.pnb, 0.0, abs_tol=0.000001):
                        out = state(hyp.prefix)
                        out.pnb += hyp.pnb * ps
                        out.nodes = [Node(n.token, n.frame, n.prob) for n in hyp.nodes]
                        if out.nodes and ps > out.nodes[-1].prob:
                            out.nodes[-1].prob = ps
                            out.nodes[-1].frame = frame
                    if not math.isclose(hyp.pb, 0.0, abs_tol=0.000001):
                        prefix = (*hyp.prefix, token)
                        out = state(prefix)
                        out.pnb += hyp.pb * ps
                        out.nodes = [Node(n.token, n.frame, n.prob) for n in hyp.nodes]
                        out.nodes.append(Node(token, frame, ps))
                else:
                    prefix = (*hyp.prefix, token)
                    out = state(prefix)
                    out.pnb += hyp.pb * ps + hyp.pnb * ps
                    if out.nodes:
                        if ps > out.nodes[-1].prob:
                            out.nodes.pop()
                            out.nodes.append(Node(token, frame, ps))
                    else:
                        out.nodes = [Node(n.token, n.frame, n.prob) for n in hyp.nodes]
                        out.nodes.append(Node(token, frame, ps))
        self.cur_hyps = sorted(next_hyps.values(), key=lambda hyp: hyp.pb + hyp.pnb, reverse=True)[
            :PATH_BEAM_SIZE
        ]

    def prune_stale_partial_prefixes(self, frame: int) -> None:
        kept: list[Hyp] = []
        for hyp in self.cur_hyps:
            if not hyp.prefix or not hyp.nodes or frame - hyp.nodes[0].frame <= MAX_FRAMES:
                kept.append(hyp)
        if not kept:
            self.reset_decoder()
        else:
            self.cur_hyps = sorted(kept, key=lambda hyp: hyp.pb + hyp.pnb, reverse=True)[
                :PATH_BEAM_SIZE
            ]

    def execute_detection(self) -> Detection:
        result = Detection()
        for hyp in self.cur_hyps:
            indexes = find_keyword_node_indexes(hyp.prefix)
            if len(indexes) != len(KEYWORD_IDS) or indexes[-1] >= len(hyp.nodes):
                continue
            product = 1.0
            for index in indexes:
                node = hyp.nodes[index]
                product *= max(1e-6, node.prob)
            score = product ** (1.0 / len(KEYWORD_IDS))
            if not result.has_keyword or score > result.score:
                result = Detection(
                    has_keyword=True,
                    start=hyp.nodes[indexes[0]].frame,
                    end=hyp.nodes[indexes[-1]].frame,
                    score=score,
                )
        if result.has_keyword:
            duration = result.end - result.start
            if result.score > self.recent_keyword_score or result.end - self.recent_keyword_end_frame > INTERVAL_FRAMES:
                self.recent_keyword_score = result.score
                self.recent_keyword_end_frame = result.end
            result.wakeup = (
                result.score >= self.threshold
                and MIN_FRAMES <= duration <= MAX_FRAMES
                and (self.last_active_frame == -1 or result.end - self.last_active_frame >= INTERVAL_FRAMES)
            )
        return result

    def run_inference_block(self, consumed_samples: int, source: str) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        raw_feats = self.feature_queue[:READ_FRAMES]
        del self.feature_queue[:READ_FRAMES]
        model_input = self.expand_and_skip(raw_feats)
        if model_input.shape[0] == 0:
            return {}, None
        output, self.cache = self.session.run(
            None,
            {"input": model_input[np.newaxis, ...], "cache": self.cache},
        )
        probs = output[0]
        best_score = 0.0
        event: Optional[dict[str, Any]] = None
        for index, frame_probs in enumerate(probs):
            frame = self.total_frames + index * FRAME_SKIP
            self.ctc_prefix_beam_search(frame, frame_probs)
            detection = self.execute_detection()
            best_score = max(best_score, detection.score)
            if detection.wakeup:
                self.last_active_frame = detection.end
                self.chunks_since_stream_reset = 0
                event = {
                    "source": source,
                    "time_sec": round(consumed_samples / SAMPLE_RATE, 3),
                    "wakeup": True,
                    "score": round(detection.score, 6),
                    "start_frame": detection.start,
                    "end_frame": detection.end,
                }
                self.reset_decoder()
                break
            self.prune_stale_partial_prefixes(frame)
        self.total_frames += int(probs.shape[0]) * FRAME_SKIP
        row = {
            "source": source,
            "time_sec": round(consumed_samples / SAMPLE_RATE, 3),
            "wakeup": event is not None,
            "score": round(best_score, 6),
            "threshold": self.threshold,
            "frame": self.total_frames,
        }
        return row, event

    def replay(self, pcm: np.ndarray, source: str) -> dict[str, Any]:
        self.reset_all()
        timeline: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        resets: list[dict[str, Any]] = []
        for offset in range(0, pcm.size, CHUNK_SAMPLES):
            chunk = pcm[offset : offset + CHUNK_SAMPLES]
            consumed = min(offset + chunk.size, pcm.size)
            reset = self.accept_waveform(chunk)
            if reset:
                resets.append({"source": source, "time_sec": round(offset / SAMPLE_RATE, 3), **reset})
            while len(self.feature_queue) >= READ_FRAMES:
                row, event = self.run_inference_block(consumed, source)
                if row:
                    timeline.append(row)
                if event:
                    events.append(event)
        return {
            "source": source,
            "duration_sec": round(pcm.size / SAMPLE_RATE, 3),
            "threshold": self.threshold,
            "speech_rms_threshold": self.speech_rms_threshold,
            "speech_peak_threshold": self.speech_peak_threshold,
            "silence_chunks_before_reset": self.silence_chunks_before_reset,
            "soft_reset_interval_chunks": self.soft_reset_interval_chunks,
            "providers": self.providers,
            "events": events,
            "timeline": timeline,
            "speech_onset_resets": resets,
        }


def find_subsequence(prefix: Iterable[int], keyword: Iterable[int]) -> int:
    values = tuple(prefix)
    target = tuple(keyword)
    for offset in range(len(values) - len(target) + 1):
        if values[offset : offset + len(target)] == target:
            return offset
    return -1


def find_keyword_node_indexes(prefix: Iterable[int]) -> list[int]:
    values = tuple(prefix)
    target = KEYWORD_IDS
    if len(values) < len(target):
        return []
    for start in range(len(values)):
        indexes: list[int] = []
        target_index = 0
        for position in range(start, len(values)):
            token = values[position]
            if token == target[target_index]:
                indexes.append(position)
                target_index += 1
                if target_index == len(target):
                    return indexes
            elif token == FILLER_ID and indexes:
                continue
            else:
                break
    return []


def match_events(reference: list[dict[str, Any]], other: list[dict[str, Any]], tolerance: float = 0.8) -> dict[str, Any]:
    available = set(range(len(other)))
    matched: list[dict[str, Any]] = []
    for ref in reference:
        candidates = [
            (abs(ref["time_sec"] - other[index]["time_sec"]), index)
            for index in available
            if abs(ref["time_sec"] - other[index]["time_sec"]) <= tolerance
        ]
        if not candidates:
            matched.append({"board_time_sec": ref["time_sec"], "matched": False})
            continue
        delta, index = min(candidates)
        available.remove(index)
        matched.append(
            {
                "board_time_sec": ref["time_sec"],
                "matched": True,
                "other_time_sec": other[index]["time_sec"],
                "delta_sec": round(delta, 3),
                "board_score": ref["score"],
                "other_score": other[index]["score"],
            }
        )
    return {
        "board_event_count": len(reference),
        "replay_event_count": len(other),
        "matched_count": sum(1 for item in matched if item["matched"]),
        "unmatched_replay_count": len(available),
        "matches": matched,
    }


def load_annotations(path: Optional[Path]) -> list[dict[str, Any]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["expected_time_sec"] = float(row["expected_time_sec"])
    return rows


def event_near(events: list[dict[str, Any]], expected_time: float, tolerance: float = 0.8) -> Optional[dict[str, Any]]:
    candidates = [event for event in events if abs(event["time_sec"] - expected_time) <= tolerance]
    return min(candidates, key=lambda event: abs(event["time_sec"] - expected_time)) if candidates else None


def write_annotation_comparison(
    path: Path,
    annotations: list[dict[str, Any]],
    board: list[dict[str, Any]],
    enhanced: list[dict[str, Any]],
    raw: list[dict[str, Any]],
    official: list[dict[str, Any]],
) -> None:
    fields = [
        "attempt_id",
        "distance",
        "expected_time_sec",
        "expected_text",
        "board_wakeup",
        "board_score",
        "pc_enhanced_wakeup",
        "pc_enhanced_score",
        "pc_raw_wakeup",
        "pc_raw_score",
        "official_stream_wakeup",
        "official_stream_score",
        "conclusion",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for annotation in annotations:
            expected = annotation["expected_time_sec"]
            board_event = event_near(board, expected)
            enhanced_event = event_near(enhanced, expected)
            raw_event = event_near(raw, expected)
            official_event = event_near(official, expected)
            if enhanced_event and not board_event:
                conclusion = "pc_enhanced_detected_but_board_missed"
            elif raw_event and not enhanced_event:
                conclusion = "raw_detected_but_enhanced_missed"
            elif board_event and enhanced_event:
                conclusion = "board_and_pc_enhanced_consistent"
            else:
                conclusion = "missed_or_requires_manual_review"
            writer.writerow(
                {
                    "attempt_id": annotation.get("attempt_id", ""),
                    "distance": annotation.get("distance", ""),
                    "expected_time_sec": expected,
                    "expected_text": annotation.get("expected_text", "anju_xiaobao"),
                    "board_wakeup": bool(board_event),
                    "board_score": board_event["score"] if board_event else "",
                    "pc_enhanced_wakeup": bool(enhanced_event),
                    "pc_enhanced_score": enhanced_event["score"] if enhanced_event else "",
                    "pc_raw_wakeup": bool(raw_event),
                    "pc_raw_score": raw_event["score"] if raw_event else "",
                    "official_stream_wakeup": bool(official_event),
                    "official_stream_score": official_event["score"] if official_event else "",
                    "conclusion": conclusion,
                }
            )


def run_official_reference(args: argparse.Namespace, pcm: np.ndarray) -> dict[str, Any]:
    required = [
        args.official_script,
        args.official_checkpoint,
        args.official_config,
        args.official_token_file,
        args.official_lexicon_file,
    ]
    if not all(required):
        return {
            "status": "not_run",
            "reason": "Pass official script/checkpoint/config/token/lexicon paths to enable this reference.",
            "events": [],
        }
    script_path = args.official_script.resolve()
    sys.path.insert(0, str(script_path.parents[2]))
    spec = importlib.util.spec_from_file_location("anju_official_stream_kws_ctc", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official WeKWS script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    spotter = module.KeyWordSpotter(
        str(args.official_checkpoint),
        str(args.official_config),
        str(args.official_token_file),
        str(args.official_lexicon_file),
        args.threshold,
        MIN_FRAMES,
        MAX_FRAMES,
        INTERVAL_FRAMES,
        SCORE_BEAM_SIZE,
        PATH_BEAM_SIZE,
        args.official_gpu,
        False,
    )
    keyword_ids = tuple(int(token) for token in args.official_keyword_ids.split(",") if token.strip())
    if keyword_ids:
        spotter.keywords_idxset = {BLANK_ID, *keyword_ids}
        spotter.keywords_token = {args.keyword: {"token_id": keyword_ids}}
    else:
        spotter.set_keywords(args.keyword)
    events: list[dict[str, Any]] = []
    for offset in range(0, pcm.size, CHUNK_SAMPLES):
        chunk = pcm[offset : offset + CHUNK_SAMPLES]
        result = spotter.forward(chunk.astype("<i2", copy=False).tobytes())
        if result.get("state") == 1:
            events.append(
                {
                    "source": "official_stream",
                    "time_sec": round((offset + chunk.size) / SAMPLE_RATE, 3),
                    "wakeup": True,
                    "score": round(float(result["score"]), 6),
                }
            )
    return {"status": "completed", "events": events, "input": "enhanced"}


def make_report(
    output_dir: Path,
    log_path: Optional[Path],
    model_path: Path,
    board: dict[str, Any],
    enhanced: dict[str, Any],
    raw: dict[str, Any],
    official: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> None:
    enhanced_match = match_events(board["events"], enhanced["events"])
    raw_match = match_events(board["events"], raw["events"])
    lines = [
        "# Android Streaming Replay Comparison",
        "",
        f"- Session log: `{log_path}`" if board["available"] else "- Session log: unavailable; board real-time wakeups cannot be reconstructed.",
        f"- ONNX model: `{model_path}`",
        "- Current Android input path: `AudioPreprocessor -> bufferQueue(enhanced) -> Spot.acceptWaveform`.",
        f"- Board wakeups: `{len(board['events'])}`" if board["available"] else "- Board wakeups: unavailable",
        f"- PC enhanced native-equivalent wakeups: `{len(enhanced['events'])}`; matched within 0.8s: `{enhanced_match['matched_count']}`",
        f"- PC raw native-equivalent wakeups: `{len(raw['events'])}`; matched within 0.8s: `{raw_match['matched_count']}`",
        f"- Official WeKWS reference: `{official['status']}`; wakeups: `{len(official.get('events', []))}`",
        "",
        "## Interpretation",
        "",
        "- `enhanced` is the board-equivalent PCM path for the current APK; `raw` is an A/B reference only.",
        "- A board/enhanced mismatch suggests differences outside audio content, such as real-time scheduling or native state timing.",
        "- A raw success with enhanced failure suggests the Java preprocessing harms the far-field signal.",
        "- If all streaming paths miss a manually confirmed utterance, use offline segment scoring before changing runtime logic.",
        "",
        "## Outputs",
        "",
        "- `replay_enhanced_result.json`",
        "- `replay_raw_result.json`",
        "- `replay_events.csv`",
        "- `score_timeline.csv`",
        "- `attempt_comparison.csv` when annotations are provided",
    ]
    if not annotations:
        lines.extend(
            [
                "",
                "No manual attempt annotation was provided. Supply `--annotations` after the controlled 2m/3m recording to compute per-attempt conclusions.",
            ]
        )
    (output_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    output_dir: Path,
    log_path: Optional[Path],
    model_path: Path,
    board: dict[str, Any],
    enhanced: dict[str, Any],
    raw: dict[str, Any],
    official: dict[str, Any],
    annotations: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "replay_enhanced_result.json").write_text(
        json.dumps(enhanced, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "replay_raw_result.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "official_stream_result.json").write_text(
        json.dumps(official, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    event_fields = ["source", "time_sec", "wakeup", "score", "start_frame", "end_frame"]
    with (output_dir / "replay_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields, extrasaction="ignore")
        writer.writeheader()
        for event in [*board["events"], *enhanced["events"], *raw["events"], *official.get("events", [])]:
            writer.writerow(event)
    timeline_fields = ["source", "time_sec", "wakeup", "score", "threshold", "frame", "infer_ms"]
    with (output_dir / "score_timeline.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=timeline_fields, extrasaction="ignore")
        writer.writeheader()
        for row in [*board["timeline"], *enhanced["timeline"], *raw["timeline"]]:
            writer.writerow(row)
    if annotations:
        write_annotation_comparison(
            output_dir / "attempt_comparison.csv",
            annotations,
            board["events"],
            enhanced["events"],
            raw["events"],
            official.get("events", []),
        )
    else:
        with (output_dir / "attempt_annotations_template.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["attempt_id", "distance", "expected_time_sec", "expected_text", "note"])
            writer.writerow(["1", "2m", "", "anju_xiaobao", "Fill after controlled recording"])
    make_report(output_dir, log_path, model_path, board, enhanced, raw, official, annotations)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Android WeKWS streaming audio in Python.")
    parser.add_argument("--log", type=Path, default=None, help="Android listen_session log, if it was saved.")
    parser.add_argument("--raw_pcm", type=Path, required=True, help="Captured raw s16le mono PCM.")
    parser.add_argument("--enhanced_pcm", type=Path, required=True, help="Captured enhanced s16le mono PCM.")
    parser.add_argument("--model", type=Path, required=True, help="Deployed kws.onnx asset.")
    parser.add_argument("--runtime_config", type=Path, required=True, help="Deployed runtime JSON config.")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--onnx_provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--keyword", default="安居小宝")
    parser.add_argument("--official_script", type=Path, default=None)
    parser.add_argument("--official_checkpoint", type=Path, default=None)
    parser.add_argument("--official_config", type=Path, default=None)
    parser.add_argument("--official_token_file", type=Path, default=None)
    parser.add_argument("--official_lexicon_file", type=Path, default=None)
    parser.add_argument("--official_gpu", type=int, default=-1)
    parser.add_argument(
        "--official_keyword_ids",
        default="1,2,3,4",
        help="Direct model token ids for the official stream reference; empty uses lexicon lookup.",
    )
    return parser


def validate_runtime_config(config: dict[str, Any]) -> None:
    expected = {
        "sample_rate": SAMPLE_RATE,
        "fbank_dim": FBANK_DIM,
        "model_input_dim": FBANK_DIM * (CONTEXT_LEFT + CONTEXT_RIGHT + 1),
        "context_left": CONTEXT_LEFT,
        "context_right": CONTEXT_RIGHT,
        "frame_skip": FRAME_SKIP,
    }
    mismatches = [
        f"{key}={config.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if config.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "Runtime config does not match the Android-native constants ported by this tool: "
            + "; ".join(mismatches)
        )


def main() -> None:
    args = build_parser().parse_args()
    config = json.loads(args.runtime_config.read_text(encoding="utf-8"))
    validate_runtime_config(config)
    args.threshold = args.threshold if args.threshold is not None else float(config.get("threshold_initial", 0.2))
    streaming_kwargs = {
        "speech_rms_threshold": float(config.get("speech_rms_threshold", DEFAULT_SPEECH_RMS_THRESHOLD)),
        "speech_peak_threshold": int(config.get("speech_peak_threshold", DEFAULT_SPEECH_PEAK_THRESHOLD)),
        "silence_chunks_before_reset": int(config.get("silence_chunks_before_reset", DEFAULT_SILENCE_CHUNKS_BEFORE_RESET)),
        "soft_reset_interval_chunks": int(config.get("soft_reset_interval_chunks", DEFAULT_SOFT_RESET_INTERVAL_CHUNKS)),
    }
    board = parse_board_log(args.log)
    raw_pcm = read_pcm(args.raw_pcm)
    enhanced_pcm = read_pcm(args.enhanced_pcm)
    if raw_pcm.size != enhanced_pcm.size:
        raise ValueError(f"raw/enhanced length mismatch: {raw_pcm.size} vs {enhanced_pcm.size}")

    enhanced_runner = NativeEquivalentReplay(
        args.model, args.threshold, args.onnx_provider, **streaming_kwargs
    )
    raw_runner = NativeEquivalentReplay(
        args.model, args.threshold, args.onnx_provider, **streaming_kwargs
    )
    enhanced = enhanced_runner.replay(enhanced_pcm, "pc_enhanced_native")
    raw = raw_runner.replay(raw_pcm, "pc_raw_native")
    official = run_official_reference(args, enhanced_pcm)
    annotations = load_annotations(args.annotations)
    write_outputs(args.output_dir, args.log, args.model, board, enhanced, raw, official, annotations)
    summary = {
        "output_dir": str(args.output_dir),
        "threshold": args.threshold,
        **streaming_kwargs,
        "board_log_available": board["available"],
        "board_events": len(board["events"]),
        "pc_enhanced_events": len(enhanced["events"]),
        "pc_raw_events": len(raw["events"]),
        "official_events": len(official.get("events", [])),
        "official_status": official["status"],
        "board_vs_enhanced": match_events(board["events"], enhanced["events"]),
        "board_vs_raw": match_events(board["events"], raw["events"]),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
