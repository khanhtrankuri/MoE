from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from text_utils import normalize_transcript


DEFAULT_TRAIN_SUBSETS = ("train-clean-100",)
DEFAULT_VALID_SUBSETS = ("dev-clean", "dev-other")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Process a local LibriSpeech tree into WAV files and JSONL manifests."
    )
    parser.add_argument(
        "--libri-root",
        default=str((repo_root / "LibriSpeech_download" / "LibriSpeech").resolve()),
        help="Path to the extracted LibriSpeech root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str((repo_root / "processed_data_librispeech").resolve()),
        help="Directory where converted WAV files and manifests will be written.",
    )
    parser.add_argument(
        "--train-subsets",
        nargs="+",
        default=list(DEFAULT_TRAIN_SUBSETS),
        help="Subset directories assigned to the training manifest.",
    )
    parser.add_argument(
        "--valid-subsets",
        nargs="+",
        default=list(DEFAULT_VALID_SUBSETS),
        help="Subset directories assigned to the validation manifest.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target WAV sample rate. Use 0 to preserve the original rate.",
    )
    parser.add_argument(
        "--source-sample-rate",
        type=int,
        default=16000,
        help="Source sample rate recorded in the manifest. LibriSpeech is normally 16 kHz.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language tag written into the manifest.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default="ffmpeg",
        help="Path to the ffmpeg executable used for FLAC -> WAV conversion.",
    )
    parser.add_argument(
        "--ffmpeg-loglevel",
        default="error",
        help="ffmpeg log level.",
    )
    parser.add_argument(
        "--text-normalization",
        choices=("none", "nfc", "nfkc"),
        default="nfc",
        help="Unicode normalization applied to transcripts.",
    )
    parser.add_argument(
        "--max-utterances-per-subset",
        type=int,
        default=None,
        help="Optional cap used for quick smoke tests.",
    )
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Re-encode WAV files even if they already exist.",
    )
    return parser.parse_args()


def is_usable_ffmpeg(candidate: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [candidate, "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return False, f"launch failed: {exc}"

    if result.returncode == 0:
        return True, ""

    details = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    return False, details


def iter_ffmpeg_candidates(ffmpeg_path: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str | Path | None) -> None:
        if not candidate:
            return
        candidate_path = Path(candidate)
        if not candidate_path.exists():
            return
        resolved = str(candidate_path.resolve())
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    direct_path = Path(ffmpeg_path)
    if direct_path.exists():
        add(direct_path)

    resolved_requested = shutil.which(ffmpeg_path)
    if resolved_requested:
        add(resolved_requested)

    resolved_default = shutil.which("ffmpeg")
    if resolved_default:
        add(resolved_default)

    conda_prefixes: list[Path] = []
    for raw_prefix in (
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
    ):
        if raw_prefix:
            conda_prefixes.append(Path(raw_prefix))
    if direct_path.exists():
        conda_prefixes.append(direct_path.resolve().parent)

    extra_roots: list[Path] = []
    for prefix in conda_prefixes:
        extra_roots.append(prefix)
        if prefix.name.lower() == "bin" and prefix.parent.name.lower() == "library":
            extra_roots.append(prefix.parent.parent)
        if prefix.name.lower() == "library":
            extra_roots.append(prefix.parent)
        if prefix.parent.name.lower() == "envs":
            extra_roots.append(prefix.parent.parent)

    for root in extra_roots:
        add(root / "Library" / "bin" / "ffmpeg.exe")
        add(root / "bin" / "ffmpeg")
        pkgs_dir = root / "pkgs"
        if pkgs_dir.exists():
            for packaged in sorted(pkgs_dir.glob("ffmpeg-*")):
                add(packaged / "Library" / "bin" / "ffmpeg.exe")

    return candidates


def require_ffmpeg(ffmpeg_path: str) -> str:
    requested_resolved: str | None = None
    requested_path = Path(ffmpeg_path)
    if requested_path.exists():
        requested_resolved = str(requested_path.resolve())

    attempted: list[str] = []
    for candidate in iter_ffmpeg_candidates(ffmpeg_path):
        usable, reason = is_usable_ffmpeg(candidate)
        if usable:
            if requested_resolved is not None and str(Path(candidate).resolve()) != requested_resolved:
                print(
                    f"Requested ffmpeg '{ffmpeg_path}' is not usable. Falling back to: {candidate}",
                    flush=True,
                )
            return candidate
        attempted.append(f"{candidate} ({reason})")

    raise FileNotFoundError(
        "Unable to find a working ffmpeg executable. "
        f"Requested value: {ffmpeg_path!r}. Attempted: {attempted or ['none']}"
    )


def detect_subset_dirs(libri_root: Path) -> list[str]:
    subset_names: list[str] = []
    for child in sorted(libri_root.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith(("train-", "dev-", "test-")):
            continue
        subset_names.append(child.name)
    if not subset_names:
        raise FileNotFoundError(
            f"No LibriSpeech subset directories were found under {libri_root}."
        )
    return subset_names


def resolve_split_mapping(libri_root: Path, train_subsets: list[str], valid_subsets: list[str]) -> dict[str, list[str]]:
    available_subsets = detect_subset_dirs(libri_root)
    available_set = set(available_subsets)

    missing_train = sorted(set(train_subsets) - available_set)
    missing_valid = sorted(set(valid_subsets) - available_set)
    if missing_train:
        raise FileNotFoundError(f"Train subsets not found: {missing_train}")
    if missing_valid:
        raise FileNotFoundError(f"Validation subsets not found: {missing_valid}")

    train_unique = []
    valid_unique = []
    seen: set[str] = set()
    for subset in train_subsets:
        if subset not in seen:
            train_unique.append(subset)
            seen.add(subset)
    for subset in valid_subsets:
        if subset not in seen:
            valid_unique.append(subset)
            seen.add(subset)

    test_subsets = [subset for subset in available_subsets if subset not in seen]
    if not test_subsets:
        raise ValueError(
            "No test subsets remain after assigning train and validation subsets. "
            "Adjust --train-subsets/--valid-subsets."
        )

    return {
        "train": train_unique,
        "validation": valid_unique,
        "test": test_subsets,
    }


def parse_speakers_metadata(libri_root: Path) -> dict[str, dict[str, str]]:
    speakers_path = libri_root / "SPEAKERS.TXT"
    if not speakers_path.exists():
        return {}

    metadata: dict[str, dict[str, str]] = {}
    with speakers_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(";") or line.startswith("ID"):
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) < 5:
                continue
            speaker_id = parts[0]
            metadata[speaker_id] = {
                "gender": parts[1],
                "corpus_subset": parts[2],
                "minutes": parts[3],
                "reader_name": parts[4],
            }
    return metadata


def parse_transcript_file(transcript_path: Path, *, text_normalization: str) -> dict[str, str]:
    unicode_form = None if text_normalization == "none" else text_normalization.upper()
    utterances: dict[str, str] = {}
    with transcript_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                utterance_id, transcript = line.split(" ", 1)
            except ValueError as exc:
                raise ValueError(f"Invalid transcript line in {transcript_path}: {line!r}") from exc
            utterances[utterance_id] = normalize_transcript(transcript, unicode_form=unicode_form)
    return utterances


def convert_flac_to_wav(
    input_path: Path,
    output_path: Path,
    *,
    ffmpeg_path: str,
    ffmpeg_loglevel: str,
    sample_rate: int,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return

    command = [
        ffmpeg_path,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        ffmpeg_loglevel,
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-ac",
        "1",
    ]
    if sample_rate > 0:
        command.extend(["-ar", str(sample_rate)])
    command.append(str(output_path))

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise RuntimeError(
            f"ffmpeg failed for {input_path} using '{ffmpeg_path}': {message}"
        )


def read_wav_info(wav_path: Path) -> tuple[int, int, float]:
    with wave.open(str(wav_path), "rb") as handle:
        sample_rate = int(handle.getframerate())
        num_frames = int(handle.getnframes())
    duration_seconds = round(num_frames / sample_rate, 6) if sample_rate > 0 else 0.0
    return sample_rate, num_frames, duration_seconds


def build_record(
    *,
    split_name: str,
    source_subset: str,
    utterance_id: str,
    transcript: str,
    wav_path: Path,
    output_dir: Path,
    speaker_metadata: dict[str, dict[str, str]],
    default_language: str,
    source_sample_rate: int,
) -> dict[str, object]:
    speaker_id = utterance_id.split("-", 1)[0]
    speaker_info = speaker_metadata.get(speaker_id, {})
    sample_rate, num_samples, duration_seconds = read_wav_info(wav_path)
    return {
        "id": utterance_id,
        "split": split_name,
        "source_subset": source_subset,
        "audio_filepath": str(wav_path.resolve()),
        "audio_relpath": str(wav_path.relative_to(output_dir)),
        "text": transcript,
        "speaker_id": speaker_id,
        "gender": speaker_info.get("gender", ""),
        "language": default_language,
        "sample_rate": sample_rate,
        "source_sample_rate": int(source_sample_rate),
        "num_samples": num_samples,
        "duration_seconds": duration_seconds,
    }


def write_manifests(split_name: str, records: list[dict[str, object]], output_dir: Path) -> None:
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = manifests_dir / f"{split_name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = manifests_dir / f"{split_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        if records:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)


def export_split(
    *,
    libri_root: Path,
    output_dir: Path,
    split_name: str,
    source_subsets: list[str],
    speaker_metadata: dict[str, dict[str, str]],
    default_language: str,
    text_normalization: str,
    max_utterances_per_subset: int | None,
    ffmpeg_path: str,
    ffmpeg_loglevel: str,
    sample_rate: int,
    overwrite_audio: bool,
    source_sample_rate: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    records: list[dict[str, object]] = []
    split_summary: dict[str, dict[str, object]] = {}

    for subset_name in source_subsets:
        subset_dir = libri_root / subset_name
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory does not exist: {subset_dir}")

        subset_records: list[dict[str, object]] = []
        transcript_files = sorted(subset_dir.rglob("*.trans.txt"))
        if not transcript_files:
            split_summary[subset_name] = {
                "status": "skipped_missing_transcripts",
                "num_rows": 0,
                "total_duration_seconds": 0.0,
            }
            print(
                f"[{split_name}:{subset_name}] skipped because no *.trans.txt files were found",
                flush=True,
            )
            continue

        processed = 0
        for transcript_file in transcript_files:
            utterances = parse_transcript_file(
                transcript_file,
                text_normalization=text_normalization,
            )
            chapter_dir = transcript_file.parent
            for utterance_id, transcript in utterances.items():
                flac_path = chapter_dir / f"{utterance_id}.flac"
                if not flac_path.exists():
                    raise FileNotFoundError(f"Missing FLAC file for {utterance_id}: {flac_path}")

                wav_path = output_dir / "audio" / split_name / subset_name / f"{utterance_id}.wav"
                convert_flac_to_wav(
                    input_path=flac_path,
                    output_path=wav_path,
                    ffmpeg_path=ffmpeg_path,
                    ffmpeg_loglevel=ffmpeg_loglevel,
                    sample_rate=sample_rate,
                    overwrite=overwrite_audio,
                )
                subset_records.append(
                    build_record(
                        split_name=split_name,
                        source_subset=subset_name,
                        utterance_id=utterance_id,
                        transcript=transcript,
                        wav_path=wav_path,
                        output_dir=output_dir,
                        speaker_metadata=speaker_metadata,
                        default_language=default_language,
                        source_sample_rate=source_sample_rate,
                    )
                )
                processed += 1
                if processed % 100 == 0:
                    print(f"[{split_name}:{subset_name}] processed {processed} utterances", flush=True)
                if max_utterances_per_subset is not None and processed >= max_utterances_per_subset:
                    break
            if max_utterances_per_subset is not None and processed >= max_utterances_per_subset:
                break

        records.extend(subset_records)
        split_summary[subset_name] = {
            "status": "ok",
            "num_rows": len(subset_records),
            "total_duration_seconds": round(
                sum(float(item["duration_seconds"]) for item in subset_records),
                6,
            ),
        }
        print(
            f"[{split_name}:{subset_name}] completed {len(subset_records)} utterances",
            flush=True,
        )

    return records, split_summary


def main() -> None:
    args = parse_args()
    libri_root = Path(args.libri_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not libri_root.exists():
        raise FileNotFoundError(f"LibriSpeech root does not exist: {libri_root}")

    ffmpeg_path = require_ffmpeg(args.ffmpeg_path)
    split_mapping = resolve_split_mapping(
        libri_root,
        train_subsets=list(args.train_subsets),
        valid_subsets=list(args.valid_subsets),
    )
    speaker_metadata = parse_speakers_metadata(libri_root)

    print(f"Using LibriSpeech root: {libri_root}")
    print(f"Writing output to: {output_dir}")
    print(f"Resolved split mapping: {split_mapping}")
    print(f"Using ffmpeg: {ffmpeg_path}")

    summary: dict[str, dict[str, object]] = {}
    all_records: list[dict[str, object]] = []
    for split_name in ("train", "validation", "test"):
        print(f"Processing split '{split_name}'", flush=True)
        records, subset_summary = export_split(
            libri_root=libri_root,
            output_dir=output_dir,
            split_name=split_name,
            source_subsets=split_mapping[split_name],
            speaker_metadata=speaker_metadata,
            default_language=args.language,
            text_normalization=args.text_normalization,
            max_utterances_per_subset=args.max_utterances_per_subset,
            ffmpeg_path=ffmpeg_path,
            ffmpeg_loglevel=args.ffmpeg_loglevel,
            sample_rate=int(args.sample_rate),
            overwrite_audio=bool(args.overwrite_audio),
            source_sample_rate=int(args.source_sample_rate),
        )
        if not records:
            raise ValueError(f"No records were produced for split '{split_name}'.")
        records.sort(key=lambda item: (str(item["source_subset"]), str(item["id"])))
        write_manifests(split_name, records, output_dir)
        all_records.extend(records)
        summary[split_name] = {
            "num_rows": len(records),
            "total_duration_seconds": round(
                sum(float(item["duration_seconds"]) for item in records),
                6,
            ),
            "source_subsets": split_mapping[split_name],
            "subsets": subset_summary,
        }

    summary_path = output_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": "LibriSpeech_local",
                "libri_root": str(libri_root),
                "output_dir": str(output_dir),
                "sample_rate": int(args.sample_rate),
                "source_sample_rate": int(args.source_sample_rate),
                "language": args.language,
                "text_normalization": args.text_normalization,
                "ffmpeg_path": ffmpeg_path,
                "splits": summary,
                "total_rows": len(all_records),
                "total_duration_seconds": round(
                    sum(float(item["duration_seconds"]) for item in all_records),
                    6,
                ),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Finished. Files written to: {output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
