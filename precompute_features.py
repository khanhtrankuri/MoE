from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from train_dme_sim import (
    CharTokenizer,
    LogMelExtractor,
    build_progress,
    ensure_torch,
    load_jsonl,
    prepare_feature_sample,
    save_json,
    set_seed,
)

try:
    import torch
except ImportError:
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute log-mel features and targets into on-disk cache files.")
    parser.add_argument("--manifest", required=True, help="Input raw-audio JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory where cached feature files will be written.")
    parser.add_argument(
        "--tokenizer-source-manifest",
        default=None,
        help="Optional manifest used to build the shared vocabulary. Defaults to --manifest.",
    )
    parser.add_argument(
        "--vocab-json",
        default=None,
        help="Optional existing vocabulary JSON. Use the train split vocabulary for valid/test cache generation.",
    )
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate.")
    parser.add_argument("--n-fft", type=int, default=512, help="FFT size for log-mel extraction.")
    parser.add_argument("--hop-length", type=int, default=160, help="STFT hop length.")
    parser.add_argument("--win-length", type=int, default=400, help="STFT window length.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins.")
    parser.add_argument("--max-audio-seconds", type=float, default=12.0, help="Crop longer audio before feature extraction.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic cache creation.")
    return parser.parse_args()


def resolve_tokenizer(args: argparse.Namespace, records: list[dict]) -> CharTokenizer:
    if args.vocab_json:
        return CharTokenizer.load(args.vocab_json)
    source_manifest = args.tokenizer_source_manifest or args.manifest
    source_records = records if source_manifest == args.manifest else load_jsonl(source_manifest)
    return CharTokenizer.from_records(source_records)


def safe_file_stem(raw_value: str, index: int) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw_value).strip("_")
    if not cleaned:
        cleaned = f"sample_{index:08d}"
    return cleaned[:120]


def main() -> None:
    args = parse_args()
    ensure_torch()
    set_seed(args.seed)

    records = load_jsonl(args.manifest)
    tokenizer = resolve_tokenizer(args, records)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save(output_dir / "vocab.json")
    extractor = LogMelExtractor(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mels=args.n_mels,
    )
    max_samples = int(args.max_audio_seconds * args.sample_rate)

    cached_manifest_path = output_dir / "manifest.jsonl"
    total_frames = 0
    total_tokens = 0

    with cached_manifest_path.open("w", encoding="utf-8") as manifest_file:
        iterator = build_progress(records, total=len(records), desc="precompute", leave=False)
        for index, sample in enumerate(iterator, start=1):
            prepared = prepare_feature_sample(
                sample,
                extractor=extractor,
                expected_sample_rate=args.sample_rate,
                max_samples=max_samples,
                tokenizer=tokenizer,
            )
            filename = f"{index:08d}_{safe_file_stem(prepared['id'], index)}.pt"
            relative_path = Path("features") / filename
            torch.save(
                {
                    "id": prepared["id"],
                    "text": prepared["text"],
                    "domain": prepared["domain"],
                    "features": prepared["features"].cpu(),
                    "feature_length": prepared["feature_length"],
                    "target_ids": prepared["token_ids"].cpu(),
                    "target_length": prepared["target_length"],
                },
                output_dir / relative_path,
            )

            total_frames += int(prepared["feature_length"])
            total_tokens += int(prepared["target_length"])
            manifest_file.write(
                json.dumps(
                    {
                        "id": prepared["id"],
                        "text": prepared["text"],
                        "domain": prepared["domain"],
                        "feature_path": str(relative_path),
                        "feature_length": prepared["feature_length"],
                        "target_length": prepared["target_length"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    save_json(
        output_dir / "summary.json",
        {
            "source_manifest": str(Path(args.manifest).resolve()),
            "num_samples": len(records),
            "sample_rate": args.sample_rate,
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "win_length": args.win_length,
            "max_audio_seconds": args.max_audio_seconds,
            "total_feature_frames": total_frames,
            "total_target_tokens": total_tokens,
            "cached_manifest": str(cached_manifest_path),
        },
    )
    print(f"Cached {len(records)} samples to {output_dir}", flush=True)
    print(f"Cached manifest: {cached_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
