from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from train_dme_sim import (
    CharTokenizer,
    LogMelExtractor,
    add_tokenizer_args,
    build_progress,
    ensure_torch,
    load_jsonl,
    prepare_feature_sample,
    resolve_training_tokenizer,
    save_json,
    set_seed,
)
from text_utils import preview_text

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
    add_tokenizer_args(parser)
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate.")
    parser.add_argument("--n-fft", type=int, default=512, help="FFT size for log-mel extraction.")
    parser.add_argument("--hop-length", type=int, default=160, help="STFT hop length.")
    parser.add_argument("--win-length", type=int, default=400, help="STFT window length.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins.")
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=0.0,
        help="Crop longer audio before feature extraction. 0 disables cropping.",
    )
    parser.add_argument(
        "--fail-on-ctc-impossible",
        action="store_true",
        help="Raise an error if any sample has target_length > estimated_ctc_steps after subsampling.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for deterministic cache creation.")
    return parser.parse_args()


def resolve_tokenizer(args: argparse.Namespace, records: list[dict], output_dir: Path) -> CharTokenizer:
    if args.vocab_json:
        return CharTokenizer.load(args.vocab_json)
    source_manifest = args.tokenizer_source_manifest or args.manifest
    source_records = (
        records
        if Path(source_manifest).resolve() == Path(args.manifest).resolve()
        else load_jsonl(source_manifest)
    )
    return resolve_training_tokenizer(
        source_records,
        args=args,
        train_manifest=source_manifest,
        output_dir=output_dir,
    )


def infer_split_name(records: list[dict], manifest_path: str) -> str:
    for record in records:
        split_name = str(record.get("split", "")).strip().lower()
        if split_name:
            return split_name
    return Path(manifest_path).stem.lower()


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

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = resolve_tokenizer(args, records, output_dir)
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
    max_samples = 0 if float(args.max_audio_seconds) <= 0 else int(args.max_audio_seconds * args.sample_rate)
    split_name = infer_split_name(records, args.manifest)
    is_eval_split = split_name in {"validation", "valid", "test", "eval"}

    cached_manifest_path = output_dir / "manifest.jsonl"
    total_frames = 0
    total_tokens = 0
    kept_samples = 0
    cropped_count = 0
    longest_samples: list[dict] = []
    rejected_ctc_impossible: list[dict] = []
    cropped_eval_samples: list[dict] = []

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
            snapshot = {
                "id": prepared["id"],
                "domain": prepared["domain"],
                "source_duration_seconds": prepared["source_duration_seconds"],
                "processed_duration_seconds": prepared["processed_duration_seconds"],
                "feature_length": prepared["feature_length"],
                "target_length": prepared["target_length"],
                "estimated_ctc_steps": prepared["estimated_ctc_steps"],
                "was_cropped": prepared["was_cropped"],
                "text_preview": preview_text(prepared["text"], limit=160),
            }
            longest_samples.append(snapshot)
            longest_samples.sort(key=lambda item: item["source_duration_seconds"], reverse=True)
            longest_samples = longest_samples[:20]
            if prepared["was_cropped"]:
                cropped_count += 1
                if is_eval_split:
                    cropped_eval_samples.append(snapshot)
            if prepared["target_length"] > prepared["estimated_ctc_steps"]:
                rejected_ctc_impossible.append(snapshot)
                continue

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
                    "source_duration_seconds": prepared["source_duration_seconds"],
                    "processed_duration_seconds": prepared["processed_duration_seconds"],
                    "duration_seconds": prepared["duration_seconds"],
                    "was_cropped": prepared["was_cropped"],
                    "estimated_dense_ctc_steps": prepared["estimated_dense_ctc_steps"],
                    "estimated_ctc_steps": prepared["estimated_ctc_steps"],
                },
                output_dir / relative_path,
            )

            total_frames += int(prepared["feature_length"])
            total_tokens += int(prepared["target_length"])
            kept_samples += 1
            manifest_file.write(
                json.dumps(
                    {
                        "id": prepared["id"],
                        "text": prepared["text"],
                        "domain": prepared["domain"],
                        "feature_path": str(relative_path),
                        "feature_length": prepared["feature_length"],
                        "target_length": prepared["target_length"],
                        "source_duration_seconds": prepared["source_duration_seconds"],
                        "processed_duration_seconds": prepared["processed_duration_seconds"],
                        "duration_seconds": prepared["duration_seconds"],
                        "was_cropped": prepared["was_cropped"],
                        "estimated_dense_ctc_steps": prepared["estimated_dense_ctc_steps"],
                        "estimated_ctc_steps": prepared["estimated_ctc_steps"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    if rejected_ctc_impossible:
        save_json(output_dir / "ctc_impossible_samples.json", {"samples": rejected_ctc_impossible})
    save_json(output_dir / "longest_samples.json", {"samples": longest_samples})

    save_json(
        output_dir / "summary.json",
        {
            "source_manifest": str(Path(args.manifest).resolve()),
            "num_source_samples": len(records),
            "num_samples": kept_samples,
            "split": split_name,
            "sample_rate": args.sample_rate,
            "n_mels": args.n_mels,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "win_length": args.win_length,
            "max_audio_seconds": args.max_audio_seconds,
            "total_feature_frames": total_frames,
            "total_target_tokens": total_tokens,
            "cropped_samples": cropped_count,
            "ctc_impossible_rejected_samples": len(rejected_ctc_impossible),
            "cached_manifest": str(cached_manifest_path),
        },
    )
    print(f"Cached {kept_samples}/{len(records)} samples to {output_dir}", flush=True)
    print(f"Cached manifest: {cached_manifest_path}", flush=True)
    if rejected_ctc_impossible:
        print(
            f"Rejected {len(rejected_ctc_impossible)} CTC-impossible samples. "
            f"Report: {output_dir / 'ctc_impossible_samples.json'}",
            flush=True,
        )
    if is_eval_split and cropped_eval_samples:
        save_json(output_dir / "cropped_eval_samples.json", {"samples": cropped_eval_samples})
        raise ValueError(
            f"Eval split '{split_name}' contains {len(cropped_eval_samples)} cropped samples. "
            "Rebuild with --max-audio-seconds 0."
        )
    if args.fail_on_ctc_impossible and rejected_ctc_impossible:
        raise ValueError(
            f"Found {len(rejected_ctc_impossible)} samples with target_length > estimated_ctc_steps. "
            "See ctc_impossible_samples.json for details."
        )


if __name__ == "__main__":
    main()
