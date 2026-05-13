from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import wave
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from pypinyin import Style, lazy_pinyin


KEYWORD = "\u5b89\u5c45\u5c0f\u5b9d"
SPEEDS_POSITIVE = [0.95, 1.0]
SPEEDS_NEGATIVE = [0.95, 1.05]
SNR_VALUES = [5.0, 10.0, 15.0, 20.0]

SIMILAR_NEGATIVES = [
    "安居小白", "安居小贝", "安居小北", "安居小班", "安居小板", "安居小帮",
    "安居小本", "安居小标", "安居小表", "安居小播", "安居小布", "安居小杜",
    "安居小梦", "安居小明", "安居小敏", "安居小门", "安居小王", "安居小张",
    "安居小李", "安居小赵", "安区小宝", "安住小宝", "按住小宝", "安居家小宝",
    "安居的小宝贝", "安居帮我看一下", "安居小屏", "安居小灯", "安居小厅",
    "安记小宝", "安聚小宝", "安具小宝", "安居小包子",
]
SMART_HOME_NEGATIVES = [
    "打开客厅灯", "关闭客厅灯", "打开卧室空调", "关闭卧室空调", "调高一点空调",
    "调低一点空调", "打开书房风扇", "关闭厨房净化器", "把窗帘打开", "把电视关闭",
    "打开回家模式", "关闭所有灯光", "设置明早七点闹钟", "播放一首歌", "音量调大一点",
    "音量调小一点", "检查空气质量", "打开阳台地暖", "关闭主卧热水器", "切换睡眠模式",
]
DAILY_NEGATIVES = [
    "今天下午去公司", "明天早一点出门", "中午准备开会", "晚上记得取快递",
    "周末看电影再说", "等会儿买菜提醒我", "现在先这样", "这个问题稍后处理",
    "把文件放在桌上", "我们下午再确认", "路上注意安全", "我马上就回来",
    "不用着急", "可以再说一遍吗", "刚才听见了吗", "这个方案还要再看",
    "先把门关上", "明天再联系", "我在办公室", "等一下再开始",
]
OTHER_WAKEWORD_NEGATIVES = [
    "小爱同学", "小度小度", "天猫精灵", "小艺小艺", "你好小迪", "小白小白",
    "小智小智", "小米小米", "你好小乐", "语音助手", "智能管家", "家庭助手",
    "小助手你好", "你好小云", "小精灵小精灵", "小管家小管家",
]
FILLER_NEGATIVES = [
    "嗯", "啊", "哦", "好", "可以", "不用", "等一下", "稍等", "这个", "那个",
    "然后", "就是", "你说", "我在", "知道了", "没事", "好的", "对", "不对",
]


def add_cosyvoice_repo(cosyvoice_repo: Path) -> None:
    if not (cosyvoice_repo / "cosyvoice").exists():
        raise FileNotFoundError(f"CosyVoice repo not found: {cosyvoice_repo}")
    sys.path.insert(0, str(cosyvoice_repo))
    matcha = cosyvoice_repo / "third_party" / "Matcha-TTS"
    if matcha.exists():
        sys.path.insert(0, str(matcha))


def wav_meta(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        sr = wf.getframerate()
        return {
            "duration_sec": frames / sr,
            "duration_ms": round(frames / sr * 1000),
            "sample_rate": sr,
            "channels": wf.getnchannels(),
            "sample_width_bytes": wf.getsampwidth(),
        }


def save_16k_mono(speech: torch.Tensor, sample_rate: int, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    speech = speech.detach().cpu()
    if speech.ndim == 1:
        speech = speech.unsqueeze(0)
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        speech = torchaudio.transforms.Resample(sample_rate, 16000)(speech)
    speech = speech.clamp(-1.0, 1.0)
    torchaudio.save(str(dst), speech, 16000, encoding="PCM_S", bits_per_sample=16)


def read_spk_info(path: Path) -> dict[str, dict]:
    info = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                info[parts[0]] = {
                    "age_group": parts[1],
                    "gender": parts[2],
                    "accent": parts[3],
                }
    return info


def aishell_text_from_pairs(text_with_phone: str) -> str:
    return "".join(text_with_phone.strip().split()[0::2])


def collect_prompt_candidates(aishell_root: Path, spk_info: dict[str, dict]) -> list[dict]:
    candidates = []
    for split in ("train", "test"):
        content = aishell_root / split / "content.txt"
        wav_root = aishell_root / split / "wav"
        if not content.exists():
            continue
        with content.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line.strip() or "\t" not in line:
                    continue
                wav_name, text_pairs = line.split("\t", 1)
                speaker = wav_name[:7]
                if speaker not in spk_info:
                    continue
                wav = wav_root / speaker / wav_name
                if not wav.exists():
                    continue
                candidates.append({
                    "speaker": speaker,
                    "split": split,
                    "source_wav": wav,
                    "prompt_text": aishell_text_from_pairs(text_pairs),
                    **spk_info[speaker],
                })
    return candidates


def select_one_prompt_per_speaker(candidates: list[dict], seed: int) -> list[dict]:
    by_speaker = defaultdict(list)
    for row in candidates:
        by_speaker[row["speaker"]].append(row)
    selected = []
    rng = random.Random(seed)
    for speaker in sorted(by_speaker):
        rows = sorted(by_speaker[speaker], key=lambda item: item["source_wav"].name)
        selected.append(rows[0])
    rng.shuffle(selected)
    return selected


def normalize_text(text: str) -> str:
    return re.sub(r"[\s，。！？、,.!?;；:：\"'“”‘’（）()《》<>\\/_-]+", "", text)


def pinyin_signature(text: str) -> tuple[str, ...]:
    return tuple(lazy_pinyin(normalize_text(text), style=Style.NORMAL, errors="ignore"))


def safe_negative_texts(count: int, seed: int) -> list[str]:
    keyword_sig = pinyin_signature(KEYWORD)
    pool = (
        SIMILAR_NEGATIVES * 8
        + SMART_HOME_NEGATIVES * 6
        + DAILY_NEGATIVES * 5
        + OTHER_WAKEWORD_NEGATIVES * 4
        + FILLER_NEGATIVES * 3
    )
    rng = random.Random(seed)
    rng.shuffle(pool)
    out = []
    for text in pool:
        norm = normalize_text(text)
        sig = pinyin_signature(norm)
        if not norm or norm == KEYWORD or KEYWORD in norm:
            continue
        if sig == keyword_sig:
            continue
        out.append(norm)
        if len(out) >= count:
            return out
    raise RuntimeError(f"not enough negative texts: {len(out)} < {count}")


def load_audio_mono(path: Path, sample_rate: int = 16000) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path), backend="soundfile")
    wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)
    return wav


def crop_or_tile_noise(noise: torch.Tensor, target_len: int, rng: random.Random) -> torch.Tensor:
    if noise.shape[1] >= target_len:
        start = rng.randint(0, noise.shape[1] - target_len)
        return noise[:, start:start + target_len]
    repeat = math.ceil(target_len / noise.shape[1])
    return noise.repeat(1, repeat)[:, :target_len]


def mix_with_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    clean_rms = torch.sqrt(torch.mean(clean ** 2) + 1e-12)
    noise_rms = torch.sqrt(torch.mean(noise ** 2) + 1e-12)
    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    mixed = clean + noise * (target_noise_rms / noise_rms)
    peak = mixed.abs().max().item()
    if peak > 0.98:
        mixed = mixed / peak * 0.98
    return mixed.clamp(-1.0, 1.0)


def generate_clean_tts(cosyvoice, rows: list[dict], wav_dir: Path, overwrite: bool) -> list[dict]:
    manifest_rows = []
    total = len(rows)
    for idx, row in enumerate(rows, 1):
        dst = wav_dir / f"{row['sample_id']}.wav"
        if not dst.exists() or overwrite:
            generated = False
            for output in cosyvoice.inference_zero_shot(
                row["text"],
                row["prompt_text"],
                row["prompt_wav"],
                stream=False,
                speed=float(row["speed"]),
            ):
                save_16k_mono(output["tts_speech"], cosyvoice.sample_rate, dst)
                generated = True
                break
            if not generated:
                raise RuntimeError(f"CosyVoice generated no audio for {row['sample_id']}")
        meta = wav_meta(dst)
        record = {**row, "path": str(dst), **meta}
        manifest_rows.append(record)
        print(f"[tts {idx:04d}/{total}] {dst.name} label={row['label_type']} speaker={row['prompt_speaker']} speed={row['speed']}")
    return manifest_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def create_noisy_set(clean_rows: list[dict], noise_paths: list[Path], out_dir: Path, seed: int) -> list[dict]:
    rng = random.Random(seed)
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    noise_cache: dict[Path, torch.Tensor] = {}
    noisy_rows = []
    for idx, row in enumerate(clean_rows, 1):
        clean = load_audio_mono(Path(row["path"]))
        noise_path = noise_paths[(idx - 1) % len(noise_paths)]
        if noise_path not in noise_cache:
            noise_cache[noise_path] = load_audio_mono(noise_path)
        snr = SNR_VALUES[(idx - 1) % len(SNR_VALUES)]
        noise = crop_or_tile_noise(noise_cache[noise_path], clean.shape[1], rng)
        mixed = mix_with_snr(clean, noise, snr)
        dst = wav_dir / f"{row['sample_id']}_snr{int(snr):02d}.wav"
        torchaudio.save(str(dst), mixed, 16000, encoding="PCM_S", bits_per_sample=16)
        meta = wav_meta(dst)
        noisy_rows.append({
            **row,
            "sample_id": dst.stem,
            "path": str(dst),
            "category": f"{row['category']}_rk_noise",
            "source_path": row["path"],
            "noise_path": str(noise_path),
            "snr_db": snr,
            **meta,
        })
    return noisy_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", default=r"E:\CodeWorking\Dataset\anju_xiaobao_kws_dataset_20260508")
    parser.add_argument("--output_dir", default=r"E:\CodeWorking\Dataset\anju_xiaobao_aishell218_posneg_rknoise_20260509")
    parser.add_argument("--model_dir", default=r"D:\models\CosyVoice2-0.5B")
    parser.add_argument("--cosyvoice_repo", default=r"D:\codeWorking\TTS\CosyVoice")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    aishell_root = dataset_root / "external_reference" / "data_aishell3"
    noise_dir = dataset_root / "real_recordings" / "office_noise" / "rk3566_noise_segments"
    noise_paths = sorted(noise_dir.glob("*.wav"))
    if not noise_paths:
        raise FileNotFoundError(f"no noise wav files in {noise_dir}")

    spk_info = read_spk_info(aishell_root / "spk-info.txt")
    prompts = select_one_prompt_per_speaker(
        collect_prompt_candidates(aishell_root, spk_info),
        args.seed,
    )
    prompt_dir = out_dir / "prompt_voices"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = []
    for idx, prompt in enumerate(prompts, 1):
        prompt_copy = prompt_dir / f"{idx:03d}_{prompt['speaker']}_{prompt['source_wav'].name}"
        if not prompt_copy.exists() or args.overwrite:
            shutil.copy2(prompt["source_wav"], prompt_copy)
        prompt_rows.append({
            **prompt,
            "prompt_index": idx,
            "source_wav": str(prompt["source_wav"]),
            "prompt_wav": str(prompt_copy),
        })
    write_jsonl(out_dir / "selected_prompts.jsonl", prompt_rows)

    negative_texts = safe_negative_texts(len(prompts) * len(SPEEDS_NEGATIVE), args.seed)
    tts_rows = []
    n = 0
    for prompt in prompt_rows:
        for speed in SPEEDS_POSITIVE:
            n += 1
            tts_rows.append({
                "sample_id": f"anju_xiaobao_aishell218_pos_{n:04d}",
                "text": KEYWORD,
                "label_type": "positive",
                "category": "tts_positive_aishell218",
                "source_group": "tts_aishell218",
                "prompt_text": prompt["prompt_text"],
                "prompt_wav": prompt["prompt_wav"],
                "prompt_speaker": prompt["speaker"],
                "prompt_gender": prompt["gender"],
                "prompt_age_group": prompt["age_group"],
                "prompt_accent": prompt["accent"],
                "source_split": prompt["split"],
                "speed": speed,
            })
    n = 0
    text_idx = 0
    for prompt in prompt_rows:
        for speed in SPEEDS_NEGATIVE:
            n += 1
            text = negative_texts[text_idx]
            text_idx += 1
            tts_rows.append({
                "sample_id": f"anju_xiaobao_aishell218_neg_{n:04d}",
                "text": text,
                "label_type": "negative",
                "category": "tts_negative_aishell218",
                "source_group": "tts_aishell218",
                "prompt_text": prompt["prompt_text"],
                "prompt_wav": prompt["prompt_wav"],
                "prompt_speaker": prompt["speaker"],
                "prompt_gender": prompt["gender"],
                "prompt_age_group": prompt["age_group"],
                "prompt_accent": prompt["accent"],
                "source_split": prompt["split"],
                "speed": speed,
            })

    add_cosyvoice_repo(Path(args.cosyvoice_repo))
    from cosyvoice.cli.cosyvoice import AutoModel

    cosyvoice = AutoModel(model_dir=args.model_dir, fp16=args.fp16)
    clean_rows = generate_clean_tts(cosyvoice, tts_rows, out_dir / "clean_wav", args.overwrite)
    write_jsonl(out_dir / "clean_manifest.jsonl", clean_rows)

    noisy_rows = create_noisy_set(clean_rows, noise_paths, out_dir / "noisy", args.seed)
    write_jsonl(out_dir / "manifest.jsonl", noisy_rows)
    summary = {
        "output_dir": str(out_dir),
        "speaker_count": len(prompt_rows),
        "clean_count": len(clean_rows),
        "noisy_count": len(noisy_rows),
        "positive_noisy_count": sum(row["label_type"] == "positive" for row in noisy_rows),
        "negative_noisy_count": sum(row["label_type"] == "negative" for row in noisy_rows),
        "noise_count": len(noise_paths),
        "snr_values": SNR_VALUES,
        "keyword": KEYWORD,
        "model_dir": args.model_dir,
        "cosyvoice_repo": args.cosyvoice_repo,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "fp16": bool(args.fp16),
        "manifest": str(out_dir / "manifest.jsonl"),
        "clean_manifest": str(out_dir / "clean_manifest.jsonl"),
    }
    (out_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    main()
