from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    from torch.utils.data import DataLoader, Dataset

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
    TORCH_IMPORT_ERROR = exc


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact ASR simulation with dense/token-MoE/SMEAR projector variants."
    )
    parser.add_argument("--train-manifest", required=True, help="Training JSONL manifest.")
    parser.add_argument("--valid-manifest", required=True, help="Validation JSONL manifest.")
    parser.add_argument("--test-manifest", default=None, help="Optional test JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
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
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clip value.")
    parser.add_argument("--encoder-dim", type=int, default=256, help="Encoder hidden size.")
    parser.add_argument("--encoder-layers", type=int, default=3, help="BiGRU layers.")
    parser.add_argument("--projector-dim", type=int, default=256, help="Projector output size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Expected sample rate.")
    parser.add_argument("--n-fft", type=int, default=512, help="FFT size.")
    parser.add_argument("--hop-length", type=int, default=160, help="STFT hop length.")
    parser.add_argument("--win-length", type=int, default=400, help="STFT window length.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins.")
    parser.add_argument("--max-audio-seconds", type=float, default=12.0, help="Max audio duration.")
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
) -> CharTokenizer:
    vocab_path = Path(args.vocab_json).resolve() if getattr(args, "vocab_json", None) else None
    if vocab_path is None:
        vocab_path = infer_vocab_path(train_manifest, getattr(args, "cache_dir", None))
    if vocab_path is not None and vocab_path.exists():
        return CharTokenizer.load(vocab_path)
    return CharTokenizer.from_records(train_records)


class CharTokenizer:
    def __init__(self, vocab: list[str]):
        self.id_to_token = list(vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}
        self.blank_id = self.token_to_id["<blank>"]
        self.unk_id = self.token_to_id["<unk>"]

    @classmethod
    def from_records(cls, records: list[dict]):
        chars = sorted({char for item in records for char in item.get("text", "")})
        return cls(["<blank>", "<unk>"] + chars)

    @classmethod
    def load(cls, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        if not isinstance(vocab, list):
            raise ValueError(f"Vocabulary file must contain a JSON list: {path}")
        return cls([str(token) for token in vocab])

    def encode(self, text: str) -> list[int]:
        ids = [self.token_to_id.get(ch, self.unk_id) for ch in text]
        return ids if ids else [self.unk_id]

    def decode(self, token_ids: list[int]) -> str:
        chars: list[str] = []
        prev = None
        for idx in token_ids:
            if idx == self.blank_id or idx == prev:
                prev = idx
                continue
            if 0 <= idx < len(self.id_to_token):
                token = self.id_to_token[idx]
                if token not in {"<blank>", "<unk>"}:
                    chars.append(token)
            prev = idx
        return "".join(chars)

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.id_to_token, handle, ensure_ascii=False, indent=2)


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
        return torch.log(mel + 1e-6).transpose(0, 1)


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
    if waveform.numel() > max_samples:
        waveform = waveform[:max_samples]

    features = extractor(waveform).contiguous()
    token_ids = sample.get("token_ids")
    if token_ids is None:
        if tokenizer is None:
            raise ValueError("tokenizer is required when sample does not already carry token_ids.")
        token_ids = torch.tensor(tokenizer.encode(sample.get("text", "")), dtype=torch.long)
    elif not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor(token_ids, dtype=torch.long)

    return {
        "id": sample["id"],
        "audio_filepath": sample["audio_filepath"],
        "text": sample.get("text", ""),
        "domain": domain,
        "features": features,
        "feature_length": int(features.size(0)),
        "token_ids": token_ids.to(dtype=torch.long).contiguous(),
        "target_length": int(token_ids.numel()),
    }


class SpeechSimulationDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer: CharTokenizer, sample_rate: int):
        self.records = []
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        for item in records:
            text = item.get("text", "")
            token_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
            self.records.append(
                {
                    "id": item["id"],
                    "audio_filepath": item["audio_filepath"],
                    "text": text,
                    "domain": item.get("simulation_domain", "clean"),
                    "token_ids": token_ids,
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
        features = sample["features"].contiguous()
        if device is not None:
            features = features.to(device)
            token_ids = token_ids.to(device)
        return {
            "id": sample["id"],
            "text": sample["text"],
            "domain": sample["domain"],
            "features": features,
            "feature_length": int(sample["feature_length"]),
            "token_ids": token_ids,
            "target_length": int(sample["target_length"]),
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


def build_raw_collate_fn(args, tokenizer: CharTokenizer):
    extractor = LogMelExtractor(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
    )
    max_samples = int(args.max_audio_seconds * args.sample_rate)

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


def build_cached_collate_fn():
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
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

    return collate_fn


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
    total_cer = 0.0
    total_wer = 0.0
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

            hypotheses = decode_batch(log_probs, output_lengths, tokenizer)
            for idx, (ref, hyp) in enumerate(zip(batch["texts"], hypotheses)):
                total_cer += compute_cer(ref, hyp)
                total_wer += compute_wer(ref, hyp)
                if routing is not None:
                    routing_by_domain[batch["domains"][idx]].append(routing[idx].detach().cpu())
                samples += 1

            if tqdm is not None:
                iterator.set_postfix(
                    loss=f"{total_loss / max(1, step):.4f}",
                    cer=f"{total_cer / max(1, samples):.4f}",
                )

    return {
        "loss": total_loss / max(1, len(loader)),
        "cer": total_cer / max(1, samples),
        "wer": total_wer / max(1, samples),
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
):
    del tokenizer
    model.train()
    running_loss = 0.0
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
        optimizer.zero_grad(set_to_none=True)

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

        backward_start = time.perf_counter()
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        synchronize_for_timing(device, profile_enabled)
        timing_sums["backward"] += time.perf_counter() - backward_start

        optimizer_start = time.perf_counter()
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
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

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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
        ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

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
                "valid_routing": valid_metrics["routing"],
            }
            history.append(epoch_metrics)
            print(
                f"epoch={epoch} valid_loss={valid_metrics['loss']:.4f} "
                f"valid_cer={valid_metrics['cer']:.4f} valid_wer={valid_metrics['wer']:.4f}",
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
                f"test_wer={test_metrics['wer']:.4f}",
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
                    **flatten_routing_metrics("test", test_metrics["routing"]),
                },
            )
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
