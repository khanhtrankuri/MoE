from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import shutil
import tempfile
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.utils.rnn import (
        pad_packed_sequence,
        pad_sequence,
        pack_padded_sequence,
    )
    from torch.utils.data import DataLoader, Dataset, Sampler

    TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    nn = None
    F = None
    pad_packed_sequence = None
    pad_sequence = None
    pack_padded_sequence = None
    DataLoader = None
    Dataset = object
    Sampler = object
    TORCH_IMPORT_ERROR = exc

from text_utils import normalize_transcript, split_graphemes


def add_data_pipeline_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-mode",
        choices=("raw", "cached"),
        default="raw",
        help="Read raw audio manifests or precomputed cached feature manifests.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional base directory used to resolve relative cached feature paths.",
    )
    parser.add_argument(
        "--vocab-json",
        default=None,
        help="Optional vocabulary JSON. Cached mode will try to infer it if omitted.",
    )
    parser.add_argument(
        "--preload-cache",
        action="store_true",
        help="Preload cached feature files into host RAM before training.",
    )
    parser.add_argument(
        "--preload-to-gpu",
        action="store_true",
        help="Try to preload cached features to GPU before training. Falls back safely if VRAM is insufficient.",
    )
    parser.add_argument(
        "--pin-memory",
        choices=("auto", "on", "off"),
        default="auto",
        help="Pinned host memory for DataLoader batches.",
    )
    parser.add_argument(
        "--persistent-workers",
        choices=("auto", "on", "off"),
        default="auto",
        help="Keep DataLoader workers alive between epochs when workers > 0.",
    )


def add_profiling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile-performance",
        action="store_true",
        help="Log timing for data loading, transfer, forward, backward, and optimizer steps.",
    )
    parser.add_argument(
        "--log-timing-every",
        type=int,
        default=50,
        help="Log average timing metrics every N steps when performance profiling is enabled.",
    )


def add_tokenizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tokenizer-type",
        choices=("char", "grapheme", "sentencepiece"),
        default="char",
        help="Tokenizer used to encode transcripts.",
    )
    parser.add_argument(
        "--sentencepiece-vocab-size",
        type=int,
        default=256,
        help="SentencePiece vocabulary size when --tokenizer-type sentencepiece.",
    )
    parser.add_argument(
        "--sentencepiece-character-coverage",
        type=float,
        default=1.0,
        help="SentencePiece character coverage when --tokenizer-type sentencepiece.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact ASR simulation with dense/token-MoE/SMEAR projector variants."
    )
    parser.add_argument("--train-manifest", required=True, help="Training JSONL manifest.")
    parser.add_argument("--valid-manifest", required=True, help="Validation JSONL manifest.")
    parser.add_argument("--test-manifest", default=None, help="Optional test JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument(
        "--allow-existing-output-dir",
        action="store_true",
        help="Allow reusing a non-empty output directory. Disabled by default to avoid overwriting prior runs.",
    )
    parser.add_argument(
        "--model-type",
        choices=("dense", "token_moe", "smear"),
        default="smear",
        help="Projector type.",
    )
    parser.add_argument("--num-experts", type=int, default=4, help="Number of experts.")
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for routing.",
    )
    parser.add_argument(
        "--load-balance-weight",
        type=float,
        default=0.01,
        help="Load-balance regularization weight.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor.")
    add_data_pipeline_args(parser)
    add_tokenizer_args(parser)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clip value.")
    parser.add_argument(
        "--spec-augment",
        action="store_true",
        help="Enable SpecAugment (time and frequency masking) during training.",
    )
    parser.add_argument("--freq-mask-param", type=int, default=27, help="Max frequency mask width for SpecAugment.")
    parser.add_argument("--time-mask-param", type=int, default=100, help="Max time mask width for SpecAugment.")
    parser.add_argument("--num-freq-masks", type=int, default=2, help="Number of frequency masks for SpecAugment.")
    parser.add_argument("--num-time-masks", type=int, default=2, help="Number of time masks for SpecAugment.")
    parser.add_argument(
        "--scheduler",
        choices=("none", "warmup_cosine"),
        default="warmup_cosine",
        help="Learning-rate scheduler.",
    )
    parser.add_argument("--warmup-steps", type=int, default=0, help="Warmup steps. Overrides --warmup-ratio when > 0.")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio when --warmup-steps is 0.")
    parser.add_argument("--min-lr-scale", type=float, default=0.1, help="Minimum LR scale for cosine scheduler.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument(
        "--decode-mode",
        choices=("greedy", "beam"),
        default="greedy",
        help="Decoder used for validation/test metrics and checkpoint selection.",
    )
    parser.add_argument("--beam-width", type=int, default=1, help="Beam width for beam decoding. 1=greedy fallback.")
    parser.add_argument("--encoder-dim", type=int, default=256, help="Encoder hidden size.")
    parser.add_argument("--encoder-layers", type=int, default=3, help="BiGRU layers.")
    parser.add_argument("--projector-dim", type=int, default=256, help="Projector output size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Expected sample rate.")
    parser.add_argument("--n-fft", type=int, default=512, help="FFT size.")
    parser.add_argument("--hop-length", type=int, default=160, help="STFT hop length.")
    parser.add_argument("--win-length", type=int, default=400, help="STFT window length.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins.")
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=0.0,
        help="Max audio duration in seconds. 0 disables cropping.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="auto | cpu | cuda | cuda:0 ...",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Step logging interval.")
    parser.add_argument(
        "--amp",
        choices=("auto", "on", "off"),
        default="auto",
        help="Mixed precision mode.",
    )
    parser.add_argument(
        "--tf32",
        choices=("auto", "on", "off"),
        default="auto",
        help="TF32 matmul mode for CUDA.",
    )
    parser.add_argument(
        "--eval-every-epochs",
        type=int,
        default=1,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if no validation CER improvement for N eval rounds (0 disables).",
    )
    parser.add_argument(
        "--wandb-project",
        default="moe-asr",
        help="Weights & Biases project. Empty value disables wandb.",
    )
    parser.add_argument("--wandb-entity", default=None, help="Optional wandb entity.")
    parser.add_argument("--wandb-run-name", default="dme-sim-hi", help="Optional wandb run name.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="wandb mode.",
    )
    parser.add_argument(
        "--normalize-eval-text",
        action="store_true",
        help="Normalize reference and hypothesis text before computing CER/WER.",
    )
    add_profiling_args(parser)
    return parser.parse_args()


def ensure_torch() -> None:
    if TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "The 'torch' package is required. Install dependencies with "
            "'pip install -r /mnt/data/khanhtl/MoE/requirements.txt'."
        ) from TORCH_IMPORT_ERROR


def set_seed(seed: int) -> None:
    random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def prepare_output_dir(output_dir: Path, *, allow_existing: bool = False) -> Path:
    if output_dir.exists() and any(output_dir.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Pass --allow-existing-output-dir to overwrite an existing run directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def resolve_mode(mode: str, is_cuda: bool, default_on_cuda: bool = True) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    return is_cuda and default_on_cuda


def configure_runtime(device: str, args) -> tuple[bool, bool]:
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    use_amp = resolve_mode(getattr(args, "amp", "auto"), is_cuda, default_on_cuda=True)
    use_tf32 = resolve_mode(getattr(args, "tf32", "auto"), is_cuda, default_on_cuda=True)

    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")

    return use_amp, use_tf32


def create_grad_scaler(use_amp: bool, is_cuda: bool):
    if not is_cuda:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def infer_vocab_path(manifest_path: str | None, cache_dir: str | None = None) -> Path | None:
    candidates: list[Path] = []
    if manifest_path:
        candidates.append(Path(manifest_path).resolve().parent / "vocab.json")
    if cache_dir:
        candidates.append(Path(cache_dir).resolve() / "vocab.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_training_tokenizer(
    train_records: list[dict],
    *,
    args,
    train_manifest: str | None,
    output_dir: str | Path | None = None,
) -> CharTokenizer:
    vocab_path = Path(args.vocab_json).resolve() if getattr(args, "vocab_json", None) else None
    if vocab_path is None:
        vocab_path = infer_vocab_path(train_manifest, getattr(args, "cache_dir", None))
    if vocab_path is not None and vocab_path.exists():
        return CharTokenizer.load(vocab_path)
    return CharTokenizer.from_records(
        train_records,
        tokenizer_type=getattr(args, "tokenizer_type", "char"),
        output_dir=output_dir,
        sentencepiece_vocab_size=int(getattr(args, "sentencepiece_vocab_size", 256)),
        sentencepiece_character_coverage=float(
            getattr(args, "sentencepiece_character_coverage", 1.0)
        ),
    )


class CharTokenizer:
    def __init__(
        self,
        vocab: list[str],
        *,
        tokenizer_type: str = "char",
        sentencepiece_model_path: str | Path | None = None,
    ):
        self.tokenizer_type = str(tokenizer_type)
        self.sentencepiece_model_path = (
            Path(sentencepiece_model_path).resolve() if sentencepiece_model_path else None
        )
        self._sentencepiece = None
        if self.tokenizer_type == "sentencepiece":
            self._sentencepiece = self._load_sentencepiece_processor(self.sentencepiece_model_path)
        self.id_to_token = list(vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self.blank_id = self.token_to_id["<blank>"]
        self.unk_id = self.token_to_id["<unk>"]

    @classmethod
    def from_records(
        cls,
        records: list[dict],
        *,
        tokenizer_type: str = "char",
        output_dir: str | Path | None = None,
        sentencepiece_vocab_size: int = 256,
        sentencepiece_character_coverage: float = 1.0,
    ):
        normalized_texts = [normalize_transcript(item.get("text", "")) for item in records]
        if tokenizer_type == "sentencepiece":
            if output_dir is None:
                raise ValueError("output_dir is required to build a SentencePiece tokenizer.")
            return cls._train_sentencepiece(
                normalized_texts,
                output_dir=output_dir,
                vocab_size=sentencepiece_vocab_size,
                character_coverage=sentencepiece_character_coverage,
            )

        if tokenizer_type == "grapheme":
            units = sorted({unit for text in normalized_texts for unit in split_graphemes(text)})
        else:
            units = sorted({char for text in normalized_texts for char in text})
        return cls(["<blank>", "<unk>"] + units, tokenizer_type=tokenizer_type)

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        if isinstance(vocab, list):
            return cls([str(token) for token in vocab], tokenizer_type="char")
        if not isinstance(vocab, dict):
            raise ValueError(f"Vocabulary file must contain a JSON list or object: {path}")
        tokens = vocab.get("vocab")
        if not isinstance(tokens, list):
            raise ValueError(f"Vocabulary payload must contain a 'vocab' list: {path}")
        tokenizer_type = str(vocab.get("type", "char"))
        sentencepiece_model = vocab.get("sentencepiece_model")
        model_path = None
        if sentencepiece_model:
            model_path = (Path(path).resolve().parent / str(sentencepiece_model)).resolve()
        return cls(
            [str(token) for token in tokens],
            tokenizer_type=tokenizer_type,
            sentencepiece_model_path=model_path,
        )

    @staticmethod
    def _load_sentencepiece_processor(model_path: str | Path | None):
        if model_path is None:
            raise ValueError("sentencepiece_model_path is required for a SentencePiece tokenizer.")
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError(
                "The 'sentencepiece' package is required for sentencepiece tokenization. "
                "Install dependencies with 'pip install -r requirements.txt'."
            ) from exc
        processor = spm.SentencePieceProcessor()
        processor.load(str(model_path))
        return processor

    @classmethod
    def _train_sentencepiece(
        cls,
        normalized_texts: list[str],
        *,
        output_dir: str | Path,
        vocab_size: int,
        character_coverage: float,
    ) -> CharTokenizer:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise RuntimeError(
                "The 'sentencepiece' package is required for sentencepiece tokenization. "
                "Install dependencies with 'pip install -r requirements.txt'."
            ) from exc

        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        model_prefix = output_path / "sentencepiece"
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".txt",
            delete=False,
            dir=output_path,
        ) as handle:
            for text in normalized_texts:
                handle.write(text + "\n")
            input_path = Path(handle.name)

        try:
            spm.SentencePieceTrainer.train(
                input=str(input_path),
                model_prefix=str(model_prefix),
                vocab_size=max(32, int(vocab_size)),
                model_type="unigram",
                character_coverage=float(character_coverage),
                bos_id=-1,
                eos_id=-1,
                pad_id=-1,
                unk_id=0,
                hard_vocab_limit=False,
            )
        finally:
            input_path.unlink(missing_ok=True)

        model_path = model_prefix.with_suffix(".model")
        processor = cls._load_sentencepiece_processor(model_path)
        vocab = ["<blank>"] + [
            processor.id_to_piece(idx)
            for idx in range(processor.get_piece_size())
        ]
        return cls(
            vocab,
            tokenizer_type="sentencepiece",
            sentencepiece_model_path=model_path,
        )

    def _token_units(self, text: str) -> list[str]:
        if self.tokenizer_type == "grapheme":
            return split_graphemes(text)
        return list(text)

    def encode(self, text: str) -> list[int]:
        normalized = normalize_transcript(text)
        if self.tokenizer_type == "sentencepiece":
            piece_ids = self._sentencepiece.encode(normalized, out_type=int)
            ids = [piece_id + 1 for piece_id in piece_ids]
        else:
            ids = [self.token_to_id.get(unit, self.unk_id) for unit in self._token_units(normalized)]
        return ids if ids else [self.unk_id]

    def decode(self, token_ids: list[int]) -> str:
        collapsed: list[int] = []
        prev = None
        for idx in token_ids:
            if idx == self.blank_id or idx == prev:
                prev = idx
                continue
            if idx == self.unk_id:
                prev = idx
                continue
            collapsed.append(idx)
            prev = idx
        return self.decode_tokens(collapsed)

    def decode_tokens(self, token_ids: list[int]) -> str:
        if self.tokenizer_type == "sentencepiece":
            piece_ids = [idx - 1 for idx in token_ids if idx >= 1]
            if not piece_ids:
                return ""
            return normalize_transcript(self._sentencepiece.decode_ids(piece_ids))
        units = [
            self.id_to_token[idx]
            for idx in token_ids
            if 0 <= idx < len(self.id_to_token) and self.id_to_token[idx] not in {"<blank>", "<unk>"}
        ]
        return "".join(units)

    def save(self, path: Path) -> None:
        sentencepiece_model_name = None
        if self.tokenizer_type == "sentencepiece" and self.sentencepiece_model_path is not None:
            destination = path.resolve().parent / self.sentencepiece_model_path.name
            if self.sentencepiece_model_path.resolve() != destination.resolve():
                shutil.copy2(self.sentencepiece_model_path, destination)
            sentencepiece_model_name = destination.name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "type": self.tokenizer_type,
                    "vocab": self.id_to_token,
                    "sentencepiece_model": sentencepiece_model_name,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return edit_distance(list(reference), list(hypothesis)) / len(reference)


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def compute_text_error_totals(reference: str, hypothesis: str) -> dict[str, int]:
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return {
        "char_edits": edit_distance(ref_chars, hyp_chars),
        "char_length": len(ref_chars),
        "word_edits": edit_distance(ref_words, hyp_words),
        "word_length": len(ref_words),
    }


def normalize_eval_text(text: str, args) -> str:
    if getattr(args, "normalize_eval_text", False):
        return normalize_transcript(text)
    return text


def estimate_subsampled_output_length(feature_length: int) -> int:
    return ((int(feature_length) + 1) // 2 + 1) // 2


def load_waveform(audio_path: str):
    with wave.open(audio_path, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_bytes = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        waveform = torch.frombuffer(bytearray(frame_bytes), dtype=torch.uint8).to(torch.float32)
        waveform = (waveform - 128.0) / 128.0
    elif sample_width == 2:
        waveform = torch.frombuffer(bytearray(frame_bytes), dtype=torch.int16).to(torch.float32)
        waveform = waveform / 32768.0
    elif sample_width == 4:
        waveform = torch.frombuffer(bytearray(frame_bytes), dtype=torch.int32).to(torch.float32)
        waveform = waveform / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes ({audio_path}).")

    if num_channels > 1:
        waveform = waveform.view(-1, num_channels).mean(dim=1)

    return waveform.contiguous(), sample_rate


def resample_waveform(waveform: torch.Tensor, original_sample_rate: int, target_sample_rate: int) -> torch.Tensor:
    if original_sample_rate == target_sample_rate:
        return waveform
    target_length = max(1, int(round(waveform.numel() * target_sample_rate / original_sample_rate)))
    return F.interpolate(
        waveform.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).view(-1).contiguous()


def apply_variant(waveform, variant: str):
    if variant == "clean":
        return waveform

    if variant.startswith("speed_"):
        speed = float(variant.split("_", 1)[1])
        target_length = max(1, int(round(waveform.numel() / speed)))
        return F.interpolate(
            waveform.view(1, 1, -1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).view(-1)

    if variant.startswith("noise_"):
        scale = float(variant.split("_", 1)[1])
        noise = torch.randn_like(waveform) * scale
        return torch.clamp(waveform + noise, -1.0, 1.0)

    raise ValueError(f"Unknown simulation variant: {variant}")


def hz_to_mel(freq_hz):
    return 2595.0 * math.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(mel_value):
    return 700.0 * (10 ** (mel_value / 2595.0) - 1.0)


def build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int):
    bins = n_fft // 2 + 1
    min_mel = hz_to_mel(0.0)
    max_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = torch.linspace(min_mel, max_mel, steps=n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_freqs = torch.linspace(0.0, sample_rate / 2.0, steps=bins)

    fbank = torch.zeros(n_mels, bins)
    for i in range(n_mels):
        left = hz_points[i]
        center = hz_points[i + 1]
        right = hz_points[i + 2]
        left_slope = (fft_freqs - left) / (center - left + 1e-8)
        right_slope = (right - fft_freqs) / (right - center + 1e-8)
        fbank[i] = torch.clamp(torch.minimum(left_slope, right_slope), min=0.0)
    return fbank


class LogMelExtractor:
    def __init__(self, sample_rate: int, n_fft: int, hop_length: int, win_length: int, n_mels: int):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        self.fbank = build_mel_filterbank(sample_rate, n_fft, n_mels)

    def __call__(self, waveform):
        if waveform.numel() < self.win_length:
            waveform = F.pad(waveform, (0, self.win_length - waveform.numel()))

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        power = spec.abs().pow(2.0)
        mel = torch.matmul(self.fbank, power)
        log_mel = torch.log(mel + 1e-6).transpose(0, 1)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return log_mel


def spec_augment(
    features: torch.Tensor,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """Apply SpecAugment (time and frequency masking) to features (B, T, F)."""
    augmented = features.clone()
    _, max_time, freq_dim = augmented.shape

    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_param, freq_dim - 1))
        if f > 0:
            f0 = random.randint(0, freq_dim - f)
            augmented[:, :, f0:f0 + f] = 0.0

    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_param, max_time - 1))
        if t > 0:
            t0 = random.randint(0, max_time - t)
            augmented[:, t0:t0 + t, :] = 0.0

    return augmented


def prepare_feature_sample(
    sample: dict[str, Any],
    *,
    extractor: LogMelExtractor,
    expected_sample_rate: int,
    max_samples: int,
    tokenizer: CharTokenizer | None = None,
) -> dict[str, Any]:
    waveform, sample_rate = load_waveform(sample["audio_filepath"])
    if sample_rate != expected_sample_rate:
        waveform = resample_waveform(waveform, sample_rate, expected_sample_rate)

    domain = sample.get("domain", sample.get("simulation_domain", "clean"))
    waveform = apply_variant(waveform, domain)
    full_duration_seconds = float(waveform.numel() / expected_sample_rate)
    was_cropped = False
    if max_samples > 0 and waveform.numel() > max_samples:
        waveform = waveform[:max_samples]
        was_cropped = True

    features = extractor(waveform).contiguous()
    normalized_text = normalize_transcript(sample.get("text", ""))
    token_ids = sample.get("token_ids")
    if token_ids is None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when sample does not already carry token_ids.")
        token_ids = torch.tensor(tokenizer.encode(normalized_text), dtype=torch.long)
    elif not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)

    feature_length = int(features.size(0))
    target_length = int(token_ids.numel())

    return {
        "id": sample["id"],
        "audio_filepath": sample["audio_filepath"],
        "text": normalized_text,
        "domain": domain,
        "features": features,
        "feature_length": feature_length,
        "token_ids": token_ids.to(dtype=torch.long).contiguous(),
        "target_length": target_length,
        "source_duration_seconds": round(full_duration_seconds, 6),
        "processed_duration_seconds": round(float(waveform.numel() / expected_sample_rate), 6),
        "duration_seconds": round(float(waveform.numel() / expected_sample_rate), 6),
        "was_cropped": was_cropped,
        "estimated_dense_ctc_steps": feature_length,
        "estimated_ctc_steps": estimate_subsampled_output_length(feature_length),
    }


class SpeechSimulationDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: CharTokenizer, sample_rate: int):
        self.records = []
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        for item in records:
            text = normalize_transcript(item.get("text", ""))
            token_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
            self.records.append(
                {
                    "id": item["id"],
                    "audio_filepath": item["audio_filepath"],
                    "text": text,
                    "domain": item.get("simulation_domain", "clean"),
                    "token_ids": token_ids,
                    "source_duration_seconds": float(item.get("source_duration_seconds", 0.0)),
                    "processed_duration_seconds": float(
                        item.get("processed_duration_seconds", item.get("duration_seconds", 0.0))
                    ),
                    "duration_seconds": float(item.get("duration_seconds", item.get("processed_duration_seconds", 0.0))),
                    "estimated_dense_ctc_steps": int(item.get("estimated_dense_ctc_steps", 0)),
                    "estimated_ctc_steps": int(item.get("estimated_ctc_steps", 0)),
                }
            )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


class CachedFeatureDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        *,
        manifest_path: str | None = None,
        cache_dir: str | None = None,
        preload_cache: bool = False,
        preload_device: str | None = None,
    ):
        self.records = records
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.storage_device = "disk"
        self._samples: list[dict[str, Any]] | None = None

        if preload_device is not None:
            loaded = self._try_preload(device=preload_device)
            if loaded:
                self.storage_device = preload_device
            elif preload_cache:
                self._preload(device=None)
                self.storage_device = "cpu"
        elif preload_cache:
            self._preload(device=None)
            self.storage_device = "cpu"

    def __len__(self):
        return len(self.records)

    def _resolve_feature_path(self, record: dict[str, Any]) -> Path:
        feature_path = Path(record["feature_path"])
        if feature_path.is_absolute():
            return feature_path
        if self.cache_dir is not None:
            return (self.cache_dir / feature_path).resolve()
        if self.manifest_path is not None:
            return (self.manifest_path.parent / feature_path).resolve()
        return feature_path.resolve()

    def _load_record(self, record: dict[str, Any], *, device: str | None = None) -> dict[str, Any]:
        sample = torch.load(self._resolve_feature_path(record), map_location="cpu")
        token_ids = sample["target_ids"].to(dtype=torch.long).contiguous()
        features = sample["features"].to(dtype=torch.float32).contiguous()
        if device is not None:
            features = features.to(device)
            token_ids = token_ids.to(device)
        return {
            "id": sample["id"],
            "text": normalize_transcript(sample["text"]),
            "domain": sample["domain"],
            "features": features,
            "feature_length": int(sample["feature_length"]),
            "token_ids": token_ids,
            "target_length": int(sample["target_length"]),
            "source_duration_seconds": float(sample.get("source_duration_seconds", 0.0)),
            "processed_duration_seconds": float(sample.get("processed_duration_seconds", 0.0)),
            "duration_seconds": float(sample.get("duration_seconds", 0.0)),
            "was_cropped": bool(sample.get("was_cropped", False)),
            "estimated_dense_ctc_steps": int(sample.get("estimated_dense_ctc_steps", sample["feature_length"])),
            "estimated_ctc_steps": int(
                sample.get("estimated_ctc_steps", estimate_subsampled_output_length(sample["feature_length"]))
            ),
        }

    def _preload(self, device: str | None) -> None:
        self._samples = [self._load_record(record, device=device) for record in self.records]

    def _try_preload(self, device: str) -> bool:
        try:
            self._preload(device=device)
            return True
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            self._samples = None
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                "Cached feature preload to GPU failed due to insufficient VRAM. Falling back to CPU-backed loading.",
                flush=True,
            )
            return False

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._samples is not None:
            return self._samples[index]
        return self._load_record(self.records[index], device=None)


def estimate_feature_frames_from_seconds(
    duration_seconds: float,
    *,
    sample_rate: int,
    hop_length: int,
    win_length: int,
    max_audio_seconds: float = 0.0,
) -> int:
    seconds = max(0.0, float(duration_seconds))
    if max_audio_seconds > 0:
        seconds = min(seconds, float(max_audio_seconds))
    num_samples = max(1, int(round(seconds * sample_rate)))
    if num_samples <= win_length:
        return 1
    return 1 + max(0, (num_samples - win_length) // max(1, hop_length))


def resolve_dataset_length_hints(dataset, args) -> list[int]:
    max_audio_seconds = float(getattr(args, "max_audio_seconds", 0.0))
    sample_rate = int(getattr(args, "sample_rate", 1))
    hop_length = int(getattr(args, "hop_length", 1))
    win_length = int(getattr(args, "win_length", hop_length))

    cached_samples = getattr(dataset, "_samples", None)
    if cached_samples is not None:
        return [max(1, int(sample.get("feature_length", 1))) for sample in cached_samples]

    records = getattr(dataset, "records", None)
    if records is None:
        return [1] * len(dataset)

    lengths: list[int] = []
    for record in records:
        length = 0
        for key in ("feature_length", "estimated_dense_ctc_steps", "estimated_ctc_steps", "input_length"):
            raw_value = record.get(key)
            if raw_value is None:
                continue
            length = int(raw_value)
            if length > 0:
                break
        if length <= 0:
            seconds = float(
                record.get(
                    "processed_duration_seconds",
                    record.get("duration_seconds", record.get("source_duration_seconds", 0.0)),
                )
            )
            length = estimate_feature_frames_from_seconds(
                seconds,
                sample_rate=sample_rate,
                hop_length=hop_length,
                win_length=win_length,
                max_audio_seconds=max_audio_seconds,
            )
        lengths.append(max(1, length))
    return lengths


class DynamicBatchSampler(Sampler):
    """Length-aware batch sampler with optional distributed batch partitioning."""

    def __init__(
        self,
        lengths: Sequence[int],
        max_tokens: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        drop_last: bool = False,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be > 0, got {max_tokens}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}")
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.lengths = [max(1, int(length)) for length in lengths]
        self.max_tokens = int(max_tokens)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._global_batches = self._build_global_batches()

    def _build_global_batches(self) -> list[list[int]]:
        if not self.lengths:
            return []
        ordered_indices = sorted(range(len(self.lengths)), key=lambda idx: (self.lengths[idx], idx))
        batches: list[list[int]] = []
        current_batch: list[int] = []
        current_tokens = 0
        for idx in ordered_indices:
            sample_tokens = self.lengths[idx]
            exceeds_limit = current_batch and (current_tokens + sample_tokens > self.max_tokens)
            if exceeds_limit:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(idx)
            current_tokens += sample_tokens
        if current_batch:
            batches.append(current_batch)
        return batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _get_rank_batches(self) -> list[list[int]]:
        batches = [list(batch) for batch in self._global_batches]
        if self.shuffle and len(batches) > 1:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)
        if self.num_replicas == 1:
            return batches
        if not batches:
            return []
        remainder = len(batches) % self.num_replicas
        if remainder != 0:
            if self.drop_last:
                batches = batches[: len(batches) - remainder]
            else:
                pad = self.num_replicas - remainder
                for idx in range(pad):
                    batches.append(list(batches[idx % len(batches)]))
        return batches[self.rank::self.num_replicas]

    def __iter__(self):
        yield from self._get_rank_batches()

    def __len__(self) -> int:
        if self.num_replicas == 1:
            return len(self._global_batches)
        if self.drop_last:
            return len(self._global_batches) // self.num_replicas
        return (len(self._global_batches) + self.num_replicas - 1) // self.num_replicas


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        self.decay = float(decay)
        self.num_updates = 0

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            self.num_updates += 1
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)
            for ema_buffer, model_buffer in zip(self.ema_model.buffers(), model.buffers()):
                ema_buffer.copy_(model_buffer)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_state": self.ema_model.state_dict(),
            "decay": self.decay,
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.ema_model.load_state_dict(state_dict["model_state"])
        self.decay = float(state_dict.get("decay", self.decay))
        self.num_updates = int(state_dict.get("num_updates", 0))

    @property
    def ready(self) -> bool:
        return self.num_updates > 0

def build_raw_collate_fn(args, tokenizer: CharTokenizer):
    extractor = LogMelExtractor(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
    )
    max_samples = 0 if float(args.max_audio_seconds) <= 0 else int(args.max_audio_seconds * args.sample_rate)

    def collate_fn(batch: list[dict]) -> dict:
        features = []
        feature_lengths = []
        targets = []
        target_lengths = []
        texts = []
        ids = []
        domains = []

        for sample in batch:
            prepared = prepare_feature_sample(
                sample,
                extractor=extractor,
                expected_sample_rate=args.sample_rate,
                max_samples=max_samples,
                tokenizer=tokenizer,
            )
            features.append(prepared["features"])
            feature_lengths.append(prepared["feature_length"])
            targets.append(prepared["token_ids"])
            target_lengths.append(prepared["target_length"])
            texts.append(prepared["text"])
            ids.append(prepared["id"])
            domains.append(prepared["domain"])

        return {
            "inputs": pad_sequence(features, batch_first=True),
            "input_lengths": torch.tensor(feature_lengths, dtype=torch.long),
            "targets": torch.cat(targets),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
            "texts": texts,
            "ids": ids,
            "domains": domains,
        }

    return collate_fn


def _cached_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    features = [sample["features"] for sample in batch]
    targets = [sample["token_ids"] for sample in batch]
    return {
        "inputs": pad_sequence(features, batch_first=True),
        "input_lengths": torch.tensor([sample["feature_length"] for sample in batch], dtype=torch.long),
        "targets": torch.cat(targets),
        "target_lengths": torch.tensor([sample["target_length"] for sample in batch], dtype=torch.long),
        "texts": [sample["text"] for sample in batch],
        "ids": [sample["id"] for sample in batch],
        "domains": [sample["domain"] for sample in batch],
    }


def build_cached_collate_fn():
    return _cached_collate_fn


def build_collate_fn(args, tokenizer: CharTokenizer):
    if getattr(args, "data_mode", "raw") == "cached":
        return build_cached_collate_fn()
    return build_raw_collate_fn(args, tokenizer)


def lengths_to_mask(lengths, max_len: int):
    t = torch.arange(max_len, device=lengths.device)
    return t.unsqueeze(0) < lengths.unsqueeze(1)


if TORCH_IMPORT_ERROR is None:
    class DenseProjector(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout),
            )

        def forward(self, hidden_states, mask):
            return self.net(hidden_states), None


    class TokenMoEProjector(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_experts: int,
            temperature: float,
            dropout: float,
        ):
            super().__init__()
            self.temperature = temperature
            self.router = nn.Linear(input_dim, num_experts)
            self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
            self.norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, hidden_states, mask):
            gates = torch.softmax(self.router(hidden_states) / self.temperature, dim=-1)
            expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts], dim=2)
            mixed = torch.sum(expert_outputs * gates.unsqueeze(-1), dim=2)
            mixed = self.dropout(self.norm(F.gelu(mixed)))

            masked = gates * mask.unsqueeze(-1)
            avg_gates = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return mixed, avg_gates


    class SmearProjector(nn.Module):
        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_experts: int,
            temperature: float,
            dropout: float,
        ):
            super().__init__()
            self.temperature = temperature
            self.router = nn.Linear(input_dim, num_experts)
            self.expert_weight = nn.Parameter(torch.empty(num_experts, output_dim, input_dim))
            self.expert_bias = nn.Parameter(torch.zeros(num_experts, output_dim))
            nn.init.xavier_uniform_(self.expert_weight)
            self.norm = nn.LayerNorm(output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, hidden_states, mask):
            token_gates = torch.softmax(self.router(hidden_states) / self.temperature, dim=-1)
            masked = token_gates * mask.unsqueeze(-1)
            avg_gates = masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

            mixed_weight = torch.einsum("be,eoi->boi", avg_gates, self.expert_weight)
            mixed_bias = torch.einsum("be,eo->bo", avg_gates, self.expert_bias)
            projected = torch.einsum("bti,boi->bto", hidden_states, mixed_weight)
            projected = projected + mixed_bias.unsqueeze(1)
            projected = self.dropout(self.norm(F.gelu(projected)))
            return projected, avg_gates


    class AcousticCTCModel(nn.Module):
        def __init__(self, args, vocab_size: int):
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(args.n_mels, args.encoder_dim),
                nn.LayerNorm(args.encoder_dim),
                nn.Dropout(args.dropout),
            )
            self.encoder = nn.GRU(
                input_size=args.encoder_dim,
                hidden_size=args.encoder_dim // 2,
                num_layers=args.encoder_layers,
                batch_first=True,
                dropout=args.dropout if args.encoder_layers > 1 else 0.0,
                bidirectional=True,
            )
            if args.model_type == "dense":
                self.projector = DenseProjector(args.encoder_dim, args.projector_dim, args.dropout)
            elif args.model_type == "token_moe":
                self.projector = TokenMoEProjector(
                    args.encoder_dim,
                    args.projector_dim,
                    args.num_experts,
                    args.router_temperature,
                    args.dropout,
                )
            else:
                self.projector = SmearProjector(
                    args.encoder_dim,
                    args.projector_dim,
                    args.num_experts,
                    args.router_temperature,
                    args.dropout,
                )
            self.ctc_head = nn.Linear(args.projector_dim, vocab_size)

        def forward(self, inputs, input_lengths):
            hidden = self.input_proj(inputs)
            packed = pack_padded_sequence(hidden, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
            encoded, _ = self.encoder(packed)
            encoded, output_lengths = pad_packed_sequence(encoded, batch_first=True)
            mask = lengths_to_mask(output_lengths.to(encoded.device), encoded.size(1)).float()
            projected, routing = self.projector(encoded, mask)
            logits = self.ctc_head(projected)
            log_probs = F.log_softmax(logits, dim=-1)
            return log_probs, output_lengths.cpu(), routing


else:
    DenseProjector = None
    TokenMoEProjector = None
    SmearProjector = None
    AcousticCTCModel = None


def build_dataset_for_mode(
    records: list[dict],
    *,
    tokenizer: CharTokenizer,
    sample_rate: int,
    args,
    manifest_path: str | None,
    device: str | None = None,
):
    if getattr(args, "data_mode", "raw") == "cached":
        preload_device = None
        if getattr(args, "preload_to_gpu", False) and device and device.startswith("cuda"):
            preload_device = device
        return CachedFeatureDataset(
            records,
            manifest_path=manifest_path,
            cache_dir=getattr(args, "cache_dir", None),
            preload_cache=getattr(args, "preload_cache", False),
            preload_device=preload_device,
        )
    return SpeechSimulationDataset(records, tokenizer, sample_rate)


def dataset_storage_device(dataset) -> str:
    return getattr(dataset, "storage_device", "raw")


def is_memory_resident_dataset(dataset) -> bool:
    return bool(getattr(dataset, "_samples", None) is not None)


def resolve_loader_kwargs(
    args,
    *,
    is_cuda: bool,
    data_on_device: bool = False,
    memory_resident: bool = False,
) -> dict[str, Any]:
    force_single_process = data_on_device or memory_resident
    num_workers = 0 if force_single_process else max(0, int(args.num_workers))
    pin_memory = False if data_on_device else resolve_mode(
        getattr(args, "pin_memory", "auto"),
        is_cuda,
        default_on_cuda=True,
    )
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        use_persistent = resolve_mode(
            getattr(args, "persistent_workers", "auto"),
            is_cuda,
            default_on_cuda=True,
        )
        if use_persistent:
            kwargs["persistent_workers"] = True
        if int(args.prefetch_factor) > 0:
            kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


def synchronize_for_timing(device: str, enabled: bool) -> None:
    if enabled and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def move_batch_to_device(
    batch: Any,
    device: str,
    non_blocking: bool = False,
    *,
    keep_cpu_keys: set[str] | None = None,
):
    keep_cpu_keys = {"input_lengths", "target_lengths"} if keep_cpu_keys is None else keep_cpu_keys
    if torch is not None and torch.is_tensor(batch):
        target_device = torch.device(device)
        if batch.device == target_device:
            return batch
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {
            key: value if key in keep_cpu_keys else move_batch_to_device(
                value,
                device,
                non_blocking=non_blocking,
                keep_cpu_keys=keep_cpu_keys,
            )
            for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [
            move_batch_to_device(item, device, non_blocking=non_blocking, keep_cpu_keys=keep_cpu_keys)
            for item in batch
        ]
    if isinstance(batch, tuple):
        return tuple(
            move_batch_to_device(item, device, non_blocking=non_blocking, keep_cpu_keys=keep_cpu_keys)
            for item in batch
        )
    return batch


def routing_regularizer(avg_gates, num_experts: int):
    if avg_gates is None:
        return torch.tensor(0.0, dtype=torch.float32)
    expected = torch.full(
        (num_experts,),
        1.0 / num_experts,
        device=avg_gates.device,
        dtype=avg_gates.dtype,
    )
    return F.mse_loss(avg_gates.mean(dim=0), expected)


def decode_batch(log_probs, output_lengths, tokenizer: CharTokenizer) -> list[str]:
    greedy = log_probs.argmax(dim=-1).cpu()
    outputs: list[str] = []
    for i, length in enumerate(output_lengths.tolist()):
        outputs.append(tokenizer.decode(greedy[i, :length].tolist()))
    return outputs


def _log_add_exp(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def beam_search_decode(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
    tokenizer: CharTokenizer,
    beam_width: int = 10,
) -> list[str]:
    """CTC prefix beam search in log-space."""
    if beam_width <= 1:
        return decode_batch(log_probs, output_lengths, tokenizer)

    blank_id = tokenizer.blank_id
    results: list[str] = []

    for i in range(log_probs.size(0)):
        T = int(output_lengths[i].item())
        frame_log_probs = log_probs[i, :T].cpu()

        beams: dict[tuple[int, ...], list[float]] = {(): [0.0, float("-inf")]}

        for t in range(T):
            frame = frame_log_probs[t]
            top_k = min(beam_width + 1, frame.size(0))
            topk_log_probs, topk_ids = frame.topk(top_k)
            blank_log_prob = float(frame[blank_id].item())

            new_beams: dict[tuple[int, ...], list[float]] = defaultdict(
                lambda: [float("-inf"), float("-inf")]
            )

            for prefix, (pb, pnb) in beams.items():
                p_total = _log_add_exp(pb, pnb)
                new_beams[prefix][0] = _log_add_exp(new_beams[prefix][0], p_total + blank_log_prob)

                for k in range(top_k):
                    c = int(topk_ids[k].item())
                    if c == blank_id:
                        continue
                    token_log_prob = float(topk_log_probs[k].item())
                    if prefix and prefix[-1] == c:
                        new_beams[prefix][1] = _log_add_exp(new_beams[prefix][1], pnb + token_log_prob)
                        ext = prefix + (c,)
                        new_beams[ext][1] = _log_add_exp(new_beams[ext][1], pb + token_log_prob)
                    else:
                        ext = prefix + (c,)
                        new_beams[ext][1] = _log_add_exp(new_beams[ext][1], p_total + token_log_prob)

            sorted_beams = sorted(
                new_beams.items(),
                key=lambda x: _log_add_exp(x[1][0], x[1][1]),
                reverse=True,
            )[:beam_width]
            beams = {bk: bv for bk, bv in sorted_beams}

        best_prefix = max(beams, key=lambda bk: _log_add_exp(beams[bk][0], beams[bk][1]))
        results.append(tokenizer.decode_tokens(list(best_prefix)))

    return results


def select_hypotheses(log_probs, output_lengths, tokenizer: CharTokenizer, args) -> list[str]:
    decode_mode = str(getattr(args, "decode_mode", "greedy"))
    if decode_mode == "beam":
        return beam_search_decode(
            log_probs,
            output_lengths,
            tokenizer,
            beam_width=max(1, int(getattr(args, "beam_width", 1))),
        )
    return decode_batch(log_probs, output_lengths, tokenizer)


def summarize_routing(storage: dict[str, list[torch.Tensor]]) -> dict[str, list[float]]:
    summary: dict[str, list[float]] = {}
    for domain, values in storage.items():
        if not values:
            continue
        mean = torch.stack(values).mean(dim=0)
        summary[domain] = [round(float(v), 6) for v in mean.tolist()]
    return summary


def flatten_routing_metrics(prefix: str, routing_map: dict[str, list[float]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for domain, values in routing_map.items():
        safe_domain = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(domain)).strip("_") or "unknown"
        for idx, value in enumerate(values):
            flat[f"{prefix}/routing/{safe_domain}/expert_{idx}"] = float(value)
    return flat


def init_wandb_run(args, output_dir: Path, config: dict):
    project = getattr(args, "wandb_project", None)
    mode = getattr(args, "wandb_mode", "disabled")
    if not project or mode == "disabled":
        return None

    try:
        import wandb
    except ImportError:
        print("wandb not installed, skip wandb logging.", flush=True)
        return None

    return wandb.init(
        project=project,
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_run_name", None),
        config=config,
        mode=mode,
        dir=str(output_dir),
    )


def log_wandb_metrics(wandb_run, metrics: dict) -> None:
    if wandb_run is None:
        return
    if not metrics:
        return
    step = metrics.get("global_step")
    if step is not None:
        wandb_run.log(metrics, step=int(step))
    else:
        wandb_run.log(metrics)


def finish_wandb_run(wandb_run) -> None:
    if wandb_run is not None:
        wandb_run.finish()


def build_progress(iterable, *, total: int, desc: str, leave: bool):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=leave, dynamic_ncols=True)


def build_lr_scheduler(optimizer, args, steps_per_epoch: int):
    scheduler_type = getattr(args, "scheduler", "none")
    if scheduler_type == "none":
        return None
    total_steps = max(1, int(args.epochs) * max(1, steps_per_epoch))
    warmup_steps = int(getattr(args, "warmup_steps", 0))
    if warmup_steps <= 0:
        warmup_steps = int(round(total_steps * float(getattr(args, "warmup_ratio", 0.05))))
    warmup_steps = min(max(0, warmup_steps), max(0, total_steps - 1))
    min_lr_scale = float(getattr(args, "min_lr_scale", 0.1))

    def lr_lambda(current_step: int) -> float:
        step = current_step + 1
        if warmup_steps > 0 and step <= warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def evaluate(
    model,
    loader,
    tokenizer: CharTokenizer,
    ctc_loss,
    args,
    device: str,
    stage: str = "eval",
    use_amp: bool = False,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_mean_cer = 0.0
    total_mean_wer = 0.0
    total_char_edits = 0
    total_char_length = 0
    total_word_edits = 0
    total_word_length = 0
    samples = 0
    routing_by_domain: dict[str, list[torch.Tensor]] = defaultdict(list)
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16

    with torch.no_grad():
        iterator = build_progress(loader, total=len(loader), desc=stage, leave=False)
        for step, batch in enumerate(iterator, start=1):
            batch = move_batch_to_device(batch, device, non_blocking=device.startswith("cuda"))
            with torch.autocast(
                device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp
            ):
                log_probs, output_lengths, routing = model(batch["inputs"], batch["input_lengths"])
                loss = ctc_loss(
                    log_probs.transpose(0, 1),
                    batch["targets"],
                    output_lengths,
                    batch["target_lengths"],
                )
                if routing is not None:
                    loss = loss + args.load_balance_weight * routing_regularizer(
                        routing, args.num_experts
                    )
            total_loss += loss.item()

            hypotheses = select_hypotheses(log_probs, output_lengths, tokenizer, args)
            for idx, (ref, hyp) in enumerate(zip(batch["texts"], hypotheses)):
                normalized_ref = normalize_eval_text(ref, args)
                normalized_hyp = normalize_eval_text(hyp, args)
                total_mean_cer += compute_cer(normalized_ref, normalized_hyp)
                total_mean_wer += compute_wer(normalized_ref, normalized_hyp)
                error_totals = compute_text_error_totals(normalized_ref, normalized_hyp)
                total_char_edits += error_totals["char_edits"]
                total_char_length += error_totals["char_length"]
                total_word_edits += error_totals["word_edits"]
                total_word_length += error_totals["word_length"]
                if routing is not None:
                    routing_by_domain[batch["domains"][idx]].append(routing[idx].detach().cpu())
                samples += 1

            if tqdm is not None:
                iterator.set_postfix(
                    loss=f"{total_loss / max(1, step):.4f}",
                    cer=f"{total_mean_cer / max(1, samples):.4f}",
                )

    corpus_cer = total_char_edits / max(1, total_char_length)
    corpus_wer = total_word_edits / max(1, total_word_length)
    return {
        "loss": total_loss / max(1, len(loader)),
        "cer": corpus_cer,
        "wer": corpus_wer,
        "mean_cer": total_mean_cer / max(1, samples),
        "mean_wer": total_mean_wer / max(1, samples),
        "corpus_cer": corpus_cer,
        "corpus_wer": corpus_wer,
        "routing": summarize_routing(routing_by_domain),
    }


def train_one_epoch(
    model,
    loader,
    tokenizer: CharTokenizer,
    optimizer,
    ctc_loss,
    args,
    device: str,
    epoch: int,
    wandb_run=None,
    scaler=None,
    use_amp: bool = False,
    scheduler=None,
):
    del tokenizer
    model.train()
    running_loss = 0.0
    accum_steps = max(1, int(getattr(args, "grad_accum_steps", 1)))
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16
    iterator = build_progress(loader, total=len(loader), desc=f"train e{epoch}", leave=False)
    profile_enabled = bool(getattr(args, "profile_performance", False))
    timing_sums = {
        "data": 0.0,
        "transfer": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }
    loop_end_time = time.perf_counter()

    for step, batch in enumerate(iterator, start=1):
        timing_sums["data"] += time.perf_counter() - loop_end_time

        transfer_start = time.perf_counter()
        batch = move_batch_to_device(batch, device, non_blocking=device.startswith("cuda"))
        synchronize_for_timing(device, profile_enabled)
        timing_sums["transfer"] += time.perf_counter() - transfer_start

        if (step - 1) % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        if getattr(args, "spec_augment", False):
            batch["inputs"] = spec_augment(
                batch["inputs"],
                freq_mask_param=int(getattr(args, "freq_mask_param", 27)),
                time_mask_param=int(getattr(args, "time_mask_param", 100)),
                num_freq_masks=int(getattr(args, "num_freq_masks", 2)),
                num_time_masks=int(getattr(args, "num_time_masks", 2)),
            )

        forward_start = time.perf_counter()
        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            log_probs, output_lengths, routing = model(batch["inputs"], batch["input_lengths"])
            loss = ctc_loss(
                log_probs.transpose(0, 1),
                batch["targets"],
                output_lengths,
                batch["target_lengths"],
            )
            if routing is not None:
                loss = loss + args.load_balance_weight * routing_regularizer(
                    routing, args.num_experts
                )
        synchronize_for_timing(device, profile_enabled)
        timing_sums["forward"] += time.perf_counter() - forward_start

        scaled_loss = loss / accum_steps
        backward_start = time.perf_counter()
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        synchronize_for_timing(device, profile_enabled)
        timing_sums["backward"] += time.perf_counter() - backward_start

        optimizer_start = time.perf_counter()
        if step % accum_steps == 0 or step == len(loader):
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
        synchronize_for_timing(device, profile_enabled)
        timing_sums["optimizer"] += time.perf_counter() - optimizer_start

        running_loss += loss.item()
        avg_loss = running_loss / step
        if tqdm is not None:
            iterator.set_postfix(loss=f"{avg_loss:.4f}")

        if step % args.log_interval == 0 or step == len(loader):
            print(f"epoch={epoch} step={step}/{len(loader)} train_loss={avg_loss:.4f}", flush=True)
            log_wandb_metrics(
                wandb_run,
                {
                    "global_step": (epoch - 1) * len(loader) + step,
                    "epoch": epoch,
                    "train/loss_step": float(avg_loss),
                    "train/lr": float(optimizer.param_groups[0]["lr"]),
                },
            )

        if profile_enabled and (step % max(1, args.log_timing_every) == 0 or step == len(loader)):
            avg_timing = {key: value / step for key, value in timing_sums.items()}
            print(
                f"timing epoch={epoch} step={step}/{len(loader)} "
                f"data={avg_timing['data']:.4f}s transfer={avg_timing['transfer']:.4f}s "
                f"forward={avg_timing['forward']:.4f}s backward={avg_timing['backward']:.4f}s "
                f"optimizer={avg_timing['optimizer']:.4f}s",
                flush=True,
            )
            log_wandb_metrics(
                wandb_run,
                {
                    "global_step": (epoch - 1) * len(loader) + step,
                    "epoch": epoch,
                    "timing/data": avg_timing["data"],
                    "timing/transfer": avg_timing["transfer"],
                    "timing/forward": avg_timing["forward"],
                    "timing/backward": avg_timing["backward"],
                    "timing/optimizer": avg_timing["optimizer"],
                },
            )
        loop_end_time = time.perf_counter()

    return running_loss / max(1, len(loader))


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    args = parse_args()
    ensure_torch()
    set_seed(args.seed)

    output_dir = prepare_output_dir(
        Path(args.output_dir).resolve(),
        allow_existing=bool(getattr(args, "allow_existing_output_dir", False)),
    )
    save_json(output_dir / "config.json", vars(args))
    wandb_run = init_wandb_run(args, output_dir, vars(args))

    try:
        train_records = load_jsonl(args.train_manifest)
        valid_records = load_jsonl(args.valid_manifest)
        test_records = load_jsonl(args.test_manifest) if args.test_manifest else None

        tokenizer = resolve_training_tokenizer(
            train_records,
            args=args,
            train_manifest=args.train_manifest,
            output_dir=output_dir,
        )
        tokenizer.save(output_dir / "vocab.json")

        device = choose_device(args.device)
        print(f"Using device: {device}", flush=True)
        use_amp, use_tf32 = configure_runtime(device, args)
        print(f"Runtime options: amp={use_amp} tf32={use_tf32}", flush=True)
        scaler = create_grad_scaler(use_amp=use_amp, is_cuda=device.startswith("cuda"))

        train_dataset = build_dataset_for_mode(
            train_records,
            tokenizer=tokenizer,
            sample_rate=args.sample_rate,
            args=args,
            manifest_path=args.train_manifest,
            device=device,
        )
        valid_dataset = build_dataset_for_mode(
            valid_records,
            tokenizer=tokenizer,
            sample_rate=args.sample_rate,
            args=args,
            manifest_path=args.valid_manifest,
            device=device,
        )
        test_dataset = (
            build_dataset_for_mode(
                test_records,
                tokenizer=tokenizer,
                sample_rate=args.sample_rate,
                args=args,
                manifest_path=args.test_manifest,
                device=device,
            )
            if test_records
            else None
        )
        collate_fn = build_collate_fn(args, tokenizer)
        is_cuda = device.startswith("cuda")
        data_on_device = dataset_storage_device(train_dataset).startswith("cuda")
        memory_resident = is_memory_resident_dataset(train_dataset)
        loader_kwargs = resolve_loader_kwargs(
            args,
            is_cuda=is_cuda,
            data_on_device=data_on_device,
            memory_resident=memory_resident,
        )
        print(
            f"Data mode: {args.data_mode} "
            f"train_storage={dataset_storage_device(train_dataset)} "
            f"memory_resident={memory_resident} "
            f"loader={loader_kwargs}",
            flush=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        test_loader = (
            DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                **loader_kwargs,
            )
            if test_dataset
            else None
        )

        model = AcousticCTCModel(args, vocab_size=len(tokenizer.id_to_token)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_lr_scheduler(optimizer, args, len(train_loader))
        ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)
        if scheduler is not None:
            print(
                f"Scheduler: {args.scheduler} warmup_steps="
                f"{int(args.warmup_steps) if args.warmup_steps > 0 else int(round(args.warmup_ratio * args.epochs * len(train_loader)))} "
                f"min_lr_scale={args.min_lr_scale}",
                flush=True,
            )

        best_valid_cer = float("inf")
        no_improve_rounds = 0
        eval_every = max(1, int(args.eval_every_epochs))
        patience = max(0, int(args.early_stop_patience))
        history: list[dict] = []
        for epoch in range(1, args.epochs + 1):
            print(f"Starting epoch {epoch}/{args.epochs}", flush=True)
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                tokenizer=tokenizer,
                optimizer=optimizer,
                ctc_loss=ctc_loss,
                args=args,
                device=device,
                epoch=epoch,
                wandb_run=wandb_run,
                scaler=scaler,
                use_amp=use_amp,
                scheduler=scheduler,
            )
            should_eval = (epoch % eval_every == 0) or (epoch == args.epochs)
            if not should_eval:
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": round(train_loss, 6),
                        "valid_loss": None,
                        "valid_cer": None,
                        "valid_wer": None,
                        "valid_routing": {},
                    }
                )
                log_wandb_metrics(
                    wandb_run,
                    {
                        "global_step": epoch * len(train_loader),
                        "epoch": epoch,
                        "train/loss_epoch": train_loss,
                    },
                )
                continue

            valid_metrics = evaluate(
                model=model,
                loader=valid_loader,
                tokenizer=tokenizer,
                ctc_loss=ctc_loss,
                args=args,
                device=device,
                stage=f"valid e{epoch}",
                use_amp=use_amp,
            )

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "valid_loss": round(valid_metrics["loss"], 6),
                "valid_cer": round(valid_metrics["cer"], 6),
                "valid_wer": round(valid_metrics["wer"], 6),
                "valid_mean_cer": round(valid_metrics["mean_cer"], 6),
                "valid_mean_wer": round(valid_metrics["mean_wer"], 6),
                "valid_corpus_cer": round(valid_metrics["corpus_cer"], 6),
                "valid_corpus_wer": round(valid_metrics["corpus_wer"], 6),
                "valid_routing": valid_metrics["routing"],
            }
            history.append(epoch_metrics)
            print(
                f"epoch={epoch} valid_loss={valid_metrics['loss']:.4f} "
                f"valid_cer={valid_metrics['cer']:.4f} valid_wer={valid_metrics['wer']:.4f} "
                f"mean_cer={valid_metrics['mean_cer']:.4f} mean_wer={valid_metrics['mean_wer']:.4f}",
                flush=True,
            )

            global_step = epoch * len(train_loader)
            is_best = valid_metrics["cer"] < best_valid_cer
            log_wandb_metrics(
                wandb_run,
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "train/loss_epoch": train_loss,
                    "valid/loss": valid_metrics["loss"],
                    "valid/cer": valid_metrics["cer"],
                    "valid/wer": valid_metrics["wer"],
                    "valid/mean_cer": valid_metrics["mean_cer"],
                    "valid/mean_wer": valid_metrics["mean_wer"],
                    "valid/corpus_cer": valid_metrics["corpus_cer"],
                    "valid/corpus_wer": valid_metrics["corpus_wer"],
                    "valid/is_best": int(is_best),
                    **flatten_routing_metrics("valid", valid_metrics["routing"]),
                },
            )

            if is_best:
                best_valid_cer = valid_metrics["cer"]
                no_improve_rounds = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": vars(args),
                        "vocab": tokenizer.id_to_token,
                        "best_valid_cer": best_valid_cer,
                    },
                    output_dir / "best.pt",
                )
                save_json(output_dir / "best_valid_metrics.json", valid_metrics)
                log_wandb_metrics(
                    wandb_run,
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "valid/best_cer": best_valid_cer,
                    },
                )
            else:
                no_improve_rounds += 1
                if patience > 0 and no_improve_rounds >= patience:
                    print(
                        f"Early stopping at epoch {epoch}: no validation CER improvement in "
                        f"{no_improve_rounds} eval rounds.",
                        flush=True,
                    )
                    break

        save_json(output_dir / "train_history.json", {"epochs": history})

        if test_loader is not None:
            checkpoint = torch.load(output_dir / "best.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            test_metrics = evaluate(
                model=model,
                loader=test_loader,
                tokenizer=tokenizer,
                ctc_loss=ctc_loss,
                args=args,
                device=device,
                stage="test",
                use_amp=use_amp,
            )
            save_json(output_dir / "test_metrics.json", test_metrics)
            print(
                f"test_loss={test_metrics['loss']:.4f} test_cer={test_metrics['cer']:.4f} "
                f"test_wer={test_metrics['wer']:.4f} "
                f"mean_cer={test_metrics['mean_cer']:.4f} mean_wer={test_metrics['mean_wer']:.4f}",
                flush=True,
            )
            log_wandb_metrics(
                wandb_run,
                {
                    "global_step": args.epochs * len(train_loader),
                    "epoch": args.epochs,
                    "test/loss": test_metrics["loss"],
                    "test/cer": test_metrics["cer"],
                    "test/wer": test_metrics["wer"],
                    "test/mean_cer": test_metrics["mean_cer"],
                    "test/mean_wer": test_metrics["mean_wer"],
                    "test/corpus_cer": test_metrics["corpus_cer"],
                    "test/corpus_wer": test_metrics["corpus_wer"],
                    **flatten_routing_metrics("test", test_metrics["routing"]),
                },
            )
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
