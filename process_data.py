import argparse
import csv
import io
import json
import re
import wave
from collections import Counter
from pathlib import Path

from text_utils import collect_out_of_script_chars, normalize_transcript, preview_text

AUTO = "auto"
AUDIO_COLUMN_CANDIDATES = ("audio", "speech", "wav", "sound")
TEXT_COLUMN_CANDIDATES = (
    "text",
    "transcript",
    "transcription",
    "sentence",
    "normalized_text",
    "utt_text",
)
ID_COLUMN_CANDIDATES = ("id", "segment_id", "utt_id", "utterance_id", "audio_id")
SPEAKER_COLUMN_CANDIDATES = ("speaker_id", "speaker", "spk_id", "client_id")
GENDER_COLUMN_CANDIDATES = ("gender", "sex")
LANGUAGE_COLUMN_CANDIDATES = ("language", "lang", "locale")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and process a Hugging Face speech dataset into WAV files and manifests."
    )
    parser.add_argument(
        "--dataset",
        default="SPRINGLab/IndicTTS-Hindi",
        help="Hugging Face dataset name or local dataset path.",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        help="Optional Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--output-dir",
        default="processed_data",
        help="Directory where audio files and metadata will be written.",
    )
    parser.add_argument(
        "--audio-column",
        default=AUTO,
        help="Dataset column containing audio samples. Use 'auto' to infer.",
    )
    parser.add_argument(
        "--text-column",
        default=AUTO,
        help="Dataset column containing transcripts. Use 'auto' to infer.",
    )
    parser.add_argument(
        "--id-column",
        default=AUTO,
        help="Optional dataset column used for stable file names. Use 'auto' to infer.",
    )
    parser.add_argument(
        "--speaker-column",
        default=AUTO,
        help="Optional dataset column containing speaker IDs. Use 'auto' to infer.",
    )
    parser.add_argument(
        "--gender-column",
        default=AUTO,
        help="Optional dataset column containing speaker gender. Use 'auto' to infer.",
    )
    parser.add_argument(
        "--language-column",
        default=AUTO,
        help="Optional dataset column containing language tags. Use 'auto' to infer.",
    )
    parser.add_argument(
        "--default-language",
        default="",
        help="Fallback language tag used when no language column is present or value is empty.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Target sample rate for exported WAV files.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.0,
        help="Fraction of the train split to reserve for validation if no validation split exists.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.0,
        help="Fraction of the train split to reserve for test if no test split exists.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when creating validation/test splits.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only the first N samples per split.",
    )
    parser.add_argument(
        "--repo-cache",
        default=None,
        help="Optional cache directory passed to load_dataset.",
    )
    parser.add_argument(
        "--text-normalization",
        choices=("none", "nfc", "nfkc"),
        default="nfc",
        help="Unicode normalization applied to exported transcripts.",
    )
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._") or "sample"


def coerce_string(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def optional_value(row: dict, column_name: str | None) -> str:
    if not column_name:
        return ""
    return coerce_string(row.get(column_name))


def _pick_candidate(available_columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in available_columns:
            return name
    return None


def resolve_required_column(
    requested: str | None,
    available_columns: list[str],
    candidates: tuple[str, ...],
    column_label: str,
) -> str:
    if requested and requested != AUTO:
        if requested not in available_columns:
            raise KeyError(
                f"{column_label} column '{requested}' not found. Available columns: {available_columns}"
            )
        return requested

    inferred = _pick_candidate(available_columns, candidates)
    if inferred:
        print(f"Auto-detected {column_label} column: '{inferred}'")
        return inferred

    raise KeyError(
        f"Could not infer {column_label} column. Available columns: {available_columns}. "
        f"Pass --{column_label}-column explicitly."
    )


def resolve_optional_column(
    requested: str | None,
    available_columns: list[str],
    candidates: tuple[str, ...],
    column_label: str,
) -> str | None:
    if not requested or requested == AUTO:
        inferred = _pick_candidate(available_columns, candidates)
        if inferred:
            print(f"Auto-detected {column_label} column: '{inferred}'")
            return inferred
        print(f"No {column_label} column found. Writing empty '{column_label}' values.")
        return None

    if requested not in available_columns:
        print(f"{column_label} column '{requested}' not found. Writing empty '{column_label}' values.")
        return None
    return requested


def require_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "The 'numpy' package is required. Install it with 'pip install numpy'."
        ) from exc
    return np


def as_dataset_dict(dataset):
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with 'pip install datasets'."
        ) from exc

    if isinstance(dataset, DatasetDict):
        return dataset
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    raise TypeError(f"Unsupported dataset type: {type(dataset)!r}")


def maybe_create_splits(dataset_dict, val_size: float, test_size: float, seed: int):
    try:
        from datasets import DatasetDict
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with 'pip install datasets'."
        ) from exc

    if not 0.0 <= val_size < 1.0:
        raise ValueError("--val-size must be in [0, 1).")
    if not 0.0 <= test_size < 1.0:
        raise ValueError("--test-size must be in [0, 1).")
    if val_size + test_size >= 1.0:
        raise ValueError("--val-size + --test-size must be less than 1.")

    has_train = "train" in dataset_dict
    has_validation = "validation" in dataset_dict
    has_test = "test" in dataset_dict

    if has_validation and has_test:
        return dataset_dict

    if not has_train:
        raise ValueError("Dataset has no train split to divide into validation/test splits.")

    if val_size == 0.0 and test_size == 0.0:
        return dataset_dict

    train_split = dataset_dict["train"]

    # No validation/test exists: create them from train according to requested fractions.
    if not has_validation and not has_test:
        if val_size > 0.0 and test_size > 0.0:
            held_out_fraction = val_size + test_size
            split_once = train_split.train_test_split(test_size=held_out_fraction, seed=seed)
            held_out = split_once["test"]
            test_ratio_inside_held_out = test_size / held_out_fraction
            split_twice = held_out.train_test_split(test_size=test_ratio_inside_held_out, seed=seed)
            return DatasetDict(
                {
                    "train": split_once["train"],
                    "validation": split_twice["train"],
                    "test": split_twice["test"],
                }
            )

        split_once = train_split.train_test_split(test_size=val_size or test_size, seed=seed)
        split_name = "validation" if val_size > 0.0 else "test"
        return DatasetDict({"train": split_once["train"], split_name: split_once["test"]})

    # Has test but no validation (e.g., MUCS-Hinglish): optionally create validation from train.
    if has_test and not has_validation:
        if test_size > 0.0:
            print("Note: existing 'test' split found; --test-size is ignored.")
        if val_size == 0.0:
            return dataset_dict
        split_once = train_split.train_test_split(test_size=val_size, seed=seed)
        return DatasetDict(
            {
                "train": split_once["train"],
                "validation": split_once["test"],
                "test": dataset_dict["test"],
            }
        )

    # Has validation but no test: optionally create test from train.
    if has_validation and not has_test:
        if val_size > 0.0:
            print("Note: existing 'validation' split found; --val-size is ignored.")
        if test_size == 0.0:
            return dataset_dict
        split_once = train_split.train_test_split(test_size=test_size, seed=seed)
        return DatasetDict(
            {
                "train": split_once["train"],
                "validation": dataset_dict["validation"],
                "test": split_once["test"],
            }
        )

    return dataset_dict


def write_wav(file_path: Path, audio_array, sample_rate: int) -> None:
    np = require_numpy()

    file_path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(audio_array)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    elif array.ndim != 2:
        raise ValueError(f"Expected mono or multi-channel audio, got shape {array.shape}.")

    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, -1.0, 1.0)
        pcm = (array * 32767.0).astype(np.int16)
    elif np.issubdtype(array.dtype, np.integer):
        pcm = array.astype(np.int16)
    else:
        raise ValueError(f"Unsupported audio dtype: {array.dtype}")

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def load_wav_with_wave(audio_path: str | None, audio_bytes: bytes | None):
    np = require_numpy()
    source = io.BytesIO(audio_bytes) if audio_bytes is not None else str(audio_path)
    with wave.open(source, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        array = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        array = (array - 128.0) / 128.0
    elif sample_width == 2:
        array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        array = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes.")

    if num_channels > 1:
        array = array.reshape(-1, num_channels)
    return array, sample_rate


def load_audio_array(audio_value):
    audio_path = audio_value.get("path") if isinstance(audio_value, dict) else None
    audio_bytes = audio_value.get("bytes") if isinstance(audio_value, dict) else None
    last_error = None

    try:
        import soundfile as sf

        source = io.BytesIO(audio_bytes) if audio_bytes is not None else audio_path
        if source is None:
            raise ValueError("Audio entry does not contain a path or bytes payload.")
        array, sample_rate = sf.read(source, dtype="float32", always_2d=False)
        return array, int(sample_rate)
    except Exception as exc:
        last_error = exc

    try:
        return load_wav_with_wave(audio_path=audio_path, audio_bytes=audio_bytes)
    except Exception as exc:
        details = f"soundfile error: {last_error}; wave error: {exc}"
        raise RuntimeError(
            "Unable to decode audio without Hugging Face's built-in decoder. "
            "Install 'soundfile' or provide WAV inputs. "
            f"Details: {details}"
        ) from exc


def resample_audio(audio_array, source_rate: int, target_rate: int):
    np = require_numpy()
    if source_rate == target_rate:
        return audio_array

    array = np.asarray(audio_array, dtype=np.float32)
    squeeze = False
    if array.ndim == 1:
        array = array[:, np.newaxis]
        squeeze = True
    elif array.ndim != 2:
        raise ValueError(f"Expected mono or multi-channel audio, got shape {array.shape}.")

    source_length = array.shape[0]
    target_length = max(1, int(round(source_length * target_rate / source_rate)))
    source_positions = np.arange(source_length, dtype=np.float32)
    target_positions = np.linspace(0, max(source_length - 1, 0), num=target_length, dtype=np.float32)
    channels = [
        np.interp(target_positions, source_positions, array[:, channel]).astype(np.float32)
        for channel in range(array.shape[1])
    ]
    resampled = np.stack(channels, axis=1)
    if squeeze:
        return resampled[:, 0]
    return resampled


def infer_sample_id(row: dict, audio: dict, split_name: str, index: int, id_column: str | None) -> str:
    sample_id = optional_value(row, id_column)
    if sample_id:
        return sample_id

    audio_path = audio.get("path") if isinstance(audio, dict) else None
    if audio_path:
        audio_stem = sanitize_name(Path(audio_path).stem)
        if audio_stem:
            return audio_stem

    return f"{split_name}_{index:05d}"


def export_split(
    split_name: str,
    split_dataset,
    output_dir: Path,
    audio_column: str,
    text_column: str,
    id_column: str | None,
    speaker_column: str | None,
    gender_column: str | None,
    language_column: str | None,
    default_language: str,
    target_sample_rate: int,
    text_normalization: str,
) -> tuple[list[dict], dict]:
    split_audio_dir = output_dir / "audio" / split_name
    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    out_of_script_counts: Counter[str] = Counter()
    out_of_script_examples: list[dict] = []
    bom_removed_samples = 0
    total_rows = len(split_dataset)
    for index, row in enumerate(split_dataset):
        audio = row[audio_column]
        audio_array, source_sample_rate = load_audio_array(audio)
        export_sample_rate = target_sample_rate or source_sample_rate
        export_array = (
            resample_audio(audio_array, source_sample_rate, export_sample_rate)
            if export_sample_rate != source_sample_rate
            else audio_array
        )
        sample_id = infer_sample_id(row, audio, split_name, index, id_column)
        file_stem = sanitize_name(sample_id)
        audio_path = split_audio_dir / f"{file_stem}.wav"
        write_wav(audio_path, export_array, export_sample_rate)

        duration_seconds = float(len(export_array) / export_sample_rate)
        raw_text = optional_value(row, text_column)
        bom_removed_samples += raw_text.count("\ufeff")
        unicode_form = None if text_normalization == "none" else text_normalization.upper()
        normalized_text = normalize_transcript(raw_text, unicode_form=unicode_form)
        flagged_chars = set(collect_out_of_script_chars(normalized_text))
        for ch in normalized_text:
            if ch in flagged_chars:
                out_of_script_counts[ch] += 1
        if flagged_chars and len(out_of_script_examples) < 20:
            out_of_script_examples.append(
                {
                    "id": sample_id,
                    "chars": [ch.encode("unicode_escape").decode("ascii") for ch in sorted(flagged_chars)],
                    "text_preview": preview_text(normalized_text, limit=160).encode("unicode_escape").decode("ascii"),
                }
            )
        record = {
            "id": sample_id,
            "split": split_name,
            "audio_filepath": str(audio_path.resolve()),
            "audio_relpath": str(audio_path.relative_to(output_dir)),
            "text": normalized_text,
            "speaker_id": optional_value(row, speaker_column),
            "gender": optional_value(row, gender_column),
            "language": optional_value(row, language_column) or default_language,
            "sample_rate": int(export_sample_rate),
            "source_sample_rate": int(source_sample_rate),
            "num_samples": int(len(export_array)),
            "duration_seconds": round(duration_seconds, 6),
        }
        records.append(record)

        if (index + 1) % 25 == 0 or index + 1 == total_rows:
            print(f"[{split_name}] exported {index + 1}/{total_rows}")

    jsonl_path = manifests_dir / f"{split_name}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    csv_path = manifests_dir / f"{split_name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)

    audit = {
        "text_normalization": text_normalization,
        "bom_removed_samples": bom_removed_samples,
        "out_of_script_char_counts": {
            key.encode("unicode_escape").decode("ascii"): value
            for key, value in sorted(out_of_script_counts.items(), key=lambda item: item[0])
        },
        "out_of_script_examples": out_of_script_examples,
    }
    return records, audit


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with 'pip install datasets'."
        ) from exc

    print(f"Loading dataset: {args.dataset}")
    if args.dataset_config:
        print(f"Using dataset config: {args.dataset_config}")
    dataset = load_dataset(args.dataset, name=args.dataset_config, cache_dir=args.repo_cache)
    dataset_dict = as_dataset_dict(dataset)
    print(f"Original splits: {list(dataset_dict.keys())}")
    dataset_dict = maybe_create_splits(dataset_dict, args.val_size, args.test_size, args.seed)
    print(f"Final splits: {list(dataset_dict.keys())}")

    available_columns = dataset_dict[next(iter(dataset_dict))].column_names
    args.audio_column = resolve_required_column(
        requested=args.audio_column,
        available_columns=available_columns,
        candidates=AUDIO_COLUMN_CANDIDATES,
        column_label="audio",
    )
    args.text_column = resolve_required_column(
        requested=args.text_column,
        available_columns=available_columns,
        candidates=TEXT_COLUMN_CANDIDATES,
        column_label="text",
    )
    args.id_column = resolve_optional_column(
        requested=args.id_column,
        available_columns=available_columns,
        candidates=ID_COLUMN_CANDIDATES,
        column_label="id",
    )
    args.speaker_column = resolve_optional_column(
        requested=args.speaker_column,
        available_columns=available_columns,
        candidates=SPEAKER_COLUMN_CANDIDATES,
        column_label="speaker",
    )
    args.gender_column = resolve_optional_column(
        requested=args.gender_column,
        available_columns=available_columns,
        candidates=GENDER_COLUMN_CANDIDATES,
        column_label="gender",
    )
    args.language_column = resolve_optional_column(
        requested=args.language_column,
        available_columns=available_columns,
        candidates=LANGUAGE_COLUMN_CANDIDATES,
        column_label="language",
    )

    print(f"Reading '{args.audio_column}' without Hugging Face audio decoding")
    for split_name in list(dataset_dict.keys()):
        dataset_dict[split_name] = dataset_dict[split_name].cast_column(
            args.audio_column, Audio(decode=False)
        )

    if args.sample_rate:
        print(f"Resampling exported audio to {args.sample_rate} Hz")
    else:
        print("Preserving source sample rates")

    summary: dict[str, dict] = {}
    text_audit_by_split: dict[str, dict] = {}
    all_records: list[dict] = []
    for split_name, split_dataset in dataset_dict.items():
        if args.max_samples is not None:
            limit = min(args.max_samples, len(split_dataset))
            split_dataset = split_dataset.select(range(limit))

        print(f"Processing split '{split_name}' with {len(split_dataset)} rows")
        records, text_audit = export_split(
            split_name=split_name,
            split_dataset=split_dataset,
            output_dir=output_dir,
            audio_column=args.audio_column,
            text_column=args.text_column,
            id_column=args.id_column,
            speaker_column=args.speaker_column,
            gender_column=args.gender_column,
            language_column=args.language_column,
            default_language=args.default_language,
            target_sample_rate=args.sample_rate,
            text_normalization=args.text_normalization,
        )
        all_records.extend(records)
        summary[split_name] = {
            "num_rows": len(records),
            "total_duration_seconds": round(sum(item["duration_seconds"] for item in records), 6),
        }
        text_audit_by_split[split_name] = text_audit

    summary_path = output_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "output_dir": str(output_dir),
                "sample_rate": args.sample_rate,
                "columns": {
                    "audio": args.audio_column,
                    "text": args.text_column,
                    "id": args.id_column,
                    "speaker": args.speaker_column,
                    "gender": args.gender_column,
                    "language": args.language_column,
                },
                "default_language": args.default_language,
                "text_normalization": args.text_normalization,
                "splits": summary,
                "text_audit": text_audit_by_split,
                "total_rows": len(all_records),
                "total_duration_seconds": round(
                    sum(item["duration_seconds"] for item in all_records), 6
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
