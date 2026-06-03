from __future__ import annotations

import re
import sys
from functools import partial
from pathlib import Path

import torch
import torchaudio.functional as F_audio
from hyperpyyaml import load_hyperpyyaml


_MEL_BASIS: dict[str, torch.Tensor] = {}
_HANN_WINDOW: dict[str, torch.Tensor] = {}


def add_cosyvoice_repo(cosyvoice_repo: Path) -> None:
    if not (cosyvoice_repo / "cosyvoice").exists():
        raise FileNotFoundError(f"CosyVoice repo not found: {cosyvoice_repo}")
    repo = str(cosyvoice_repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    matcha = cosyvoice_repo / "third_party" / "Matcha-TTS"
    if matcha.exists() and str(matcha) not in sys.path:
        sys.path.insert(0, str(matcha))


def _dynamic_range_compression_torch(x: torch.Tensor, c: float = 1, clip_val: float = 1e-5) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * c)


def mel_spectrogram(
    y: torch.Tensor,
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: int = 0,
    fmax=None,
    center: bool = False,
) -> torch.Tensor:
    key = f"{fmax}_{y.device}"
    if key not in _MEL_BASIS:
        mel = F_audio.melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=float(fmin),
            f_max=float(fmax) if fmax is not None else float(sampling_rate // 2),
            n_mels=num_mels,
            sample_rate=sampling_rate,
            norm=None,
            mel_scale="slaney",
        ).transpose(0, 1)
        _MEL_BASIS[key] = mel.float().to(y.device)
        _HANN_WINDOW[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)
    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=_HANN_WINDOW[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_MEL_BASIS[key], spec)
    return _dynamic_range_compression_torch(spec)


def extract_core_yaml(yaml_text: str) -> str:
    fixed = yaml_text.split("# model params")[0]

    def section(start: str, end: str) -> str:
        match = re.search(rf"(?ms)^{start}: !new:.*?(?=^{end}: !new:)", yaml_text)
        if match is None:
            raise RuntimeError(f"Cannot find yaml section {start}")
        return match.group(0) + "\n\n"

    llm = section("llm", "flow")
    flow = section("flow", "hift")
    hift_match = re.search(r"(?ms)^hift: !new:.*?(?=^# gan related module)", yaml_text)
    if hift_match is None:
        raise RuntimeError("Cannot find yaml section hift")
    return fixed + llm + flow + hift_match.group(0) + "\n"


class SlimCosyVoice3:
    """CosyVoice3 zero-shot loader that avoids Matcha/librosa mel imports."""

    def __init__(self, model_dir: Path, cosyvoice_repo: Path | None = None, fp16: bool = False):
        if cosyvoice_repo is not None:
            add_cosyvoice_repo(cosyvoice_repo)

        from cosyvoice.cli.frontend import CosyVoiceFrontEnd
        from cosyvoice.cli.model import CosyVoice3Model
        from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

        self.model_dir = model_dir
        yaml_text = (model_dir / "cosyvoice3.yaml").read_text(encoding="utf-8")
        core_yaml = extract_core_yaml(yaml_text)
        with torch.no_grad():
            configs = load_hyperpyyaml(
                core_yaml,
                overrides={"qwen_pretrain_path": str(model_dir / "CosyVoice-BlankEN")},
            )
        self.sample_rate = int(configs["sample_rate"])
        get_tokenizer = partial(
            get_qwen_tokenizer,
            token_path=str(model_dir / "CosyVoice-BlankEN"),
            skip_special_tokens=True,
            version="cosyvoice3",
        )
        self.frontend = CosyVoiceFrontEnd(
            get_tokenizer,
            mel_spectrogram,
            str(model_dir / "campplus.onnx"),
            str(model_dir / "speech_tokenizer_v3.onnx"),
            str(model_dir / "spk2info.pt"),
            "all",
        )
        self.model = CosyVoice3Model(configs["llm"], configs["flow"], configs["hift"], fp16)
        self.model.load(
            str(model_dir / "llm.pt"),
            str(model_dir / "flow.pt"),
            str(model_dir / "hift.pt"),
        )

    def inference_zero_shot(
        self,
        tts_text: str,
        prompt_text: str,
        prompt_wav: str,
        stream: bool = True,
        speed: float = 1.0,
        text_frontend: bool = False,
        zero_shot_spk_id: str = "",
    ):
        if "<|endofprompt|>" not in tts_text and "<|endofprompt|>" not in prompt_text:
            prompt_text = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
        prompt_text = self.frontend.text_normalize(prompt_text, split=False, text_frontend=text_frontend)
        for text in self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend):
            model_input = self.frontend.frontend_zero_shot(
                text,
                prompt_text,
                prompt_wav,
                self.sample_rate,
                zero_shot_spk_id,
            )
            for model_output in self.model.tts(**model_input, stream=stream, speed=speed):
                yield model_output
