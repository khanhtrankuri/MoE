"""train_libri.py – Single-GPU MoE ASR training on LibriSpeech.

Trains on train-clean-100 (by default), validates on dev-clean,
and evaluates individually on dev-clean, dev-other, test-clean, test-other.

Usage
-----
    python train_libri.py \
        --train-manifest processed_data_librispeech/manifests/train.jsonl \
        --valid-manifest processed_data_librispeech/manifests/validation.jsonl \
        --test-manifest  processed_data_librispeech/manifests/test.jsonl \
        --output-dir runs/libri_moe
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    nn = None
    F = None
    TORCH_IMPORT_ERROR = exc

from train_dme_sim import (
    CharTokenizer,
    add_data_pipeline_args,
    add_profiling_args,
    add_tokenizer_args,
    build_collate_fn,
    build_dataset_for_mode,
    build_progress,
    choose_device,
    compute_cer,
    compute_text_error_totals,
    compute_wer,
    configure_runtime,
    create_grad_scaler,
    dataset_storage_device,
    DynamicBatchSampler,
    ensure_torch,
    finish_wandb_run,
    flatten_routing_metrics,
    is_memory_resident_dataset,
    init_wandb_run,
    lengths_to_mask,
    load_jsonl,
    log_wandb_metrics,
    move_batch_to_device,
    ModelEMA,
    normalize_eval_text,
    prepare_output_dir,
    resolve_loader_kwargs,
    resolve_dataset_length_hints,
    resolve_training_tokenizer,
    save_json,
    select_hypotheses,
    set_seed,
    spec_augment,
    summarize_routing,
    synchronize_for_timing,
)

# Re-use model architecture and helpers from train_arch_moe
from train_arch_moe import (
    EncoderMoECTCModel,
    append_vector_metrics,
    build_lr_scheduler,
    competition_targets,
    compute_expert_scores,
    compute_per_sample_ctc_losses,
    flatten_scalar_metrics,
    get_annealed_temperature,
    get_effective_competition_weight,
    resolve_autocast_dtype,
    routing_alignment_loss,
    routing_entropy,
    routing_regularizer,
    should_compute_competition_metrics,
    should_run_expert_evolution,
    evolve_experts,
    train_one_epoch,
    evaluate,
    parse_args as _base_parse_args,
)


# ===========================================================================
# Argument parsing – extends the base with extra eval manifests
# ===========================================================================

def parse_args() -> argparse.Namespace:
    # Start from the base parser's result and add extra args
    args = _base_parse_args()

    # Add extra eval manifest paths (parsed manually so we don't duplicate the base parser)
    import sys
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--eval-manifests", nargs="*", default=None,
                              help="Additional JSONL manifests to evaluate individually (e.g. dev-clean, dev-other, etc.).")
    extra_parser.add_argument("--eval-names", nargs="*", default=None,
                              help="Names corresponding to --eval-manifests for logging.")
    extra_args, _ = extra_parser.parse_known_args()
    args.eval_manifests = extra_args.eval_manifests
    args.eval_names = extra_args.eval_names
    return args


# ===========================================================================
# Per-subset evaluation
# ===========================================================================

def evaluate_subsets(
    model,
    records_by_name: dict[str, list[dict]],
    tokenizer: CharTokenizer,
    ctc_loss,
    args: argparse.Namespace,
    device: str,
    use_amp: bool,
    epoch: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate model on multiple named subsets independently."""
    all_metrics = {}
    for name, records in records_by_name.items():
        dataset = build_dataset_for_mode(
            records,
            tokenizer=tokenizer,
            sample_rate=args.sample_rate,
            args=args,
            manifest_path=None,
            device=device,
        )
        collate_fn = build_collate_fn(args, tokenizer)
        is_cuda = device.startswith("cuda")
        data_on_device = dataset_storage_device(dataset).startswith("cuda")
        memory_resident = is_memory_resident_dataset(dataset)
        loader_kwargs = resolve_loader_kwargs(
            args,
            is_cuda=is_cuda,
            data_on_device=data_on_device,
            memory_resident=memory_resident,
        )
        max_tokens = max(0, int(getattr(args, "max_tokens_per_batch", 0)))
        if max_tokens > 0:
            batch_sampler = DynamicBatchSampler(
                resolve_dataset_length_hints(dataset, args),
                max_tokens,
                shuffle=False,
                seed=args.seed,
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate_fn,
                **loader_kwargs,
            )
        else:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                **loader_kwargs,
            )

        metrics = evaluate(
            model=model,
            loader=loader,
            tokenizer=tokenizer,
            ctc_loss=ctc_loss,
            args=args,
            device=device,
            stage=name,
            use_amp=use_amp,
            epoch=epoch,
        )
        all_metrics[name] = metrics
        print(
            f"  [{name}] loss={metrics['loss']:.4f} cer={metrics['cer']:.4f} "
            f"wer={metrics['wer']:.4f} mean_cer={metrics['mean_cer']:.4f} "
            f"mean_wer={metrics['mean_wer']:.4f}",
            flush=True,
        )
    return all_metrics


def split_records_by_source_subset(records: list[dict]) -> dict[str, list[dict]]:
    """Split a list of JSONL records by their 'source_subset' field."""
    by_subset: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        subset_name = rec.get("source_subset", "unknown")
        by_subset[subset_name].append(rec)
    return dict(by_subset)


# ===========================================================================
# Main
# ===========================================================================

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

        # Load extra eval manifests if provided
        extra_eval_records: dict[str, list[dict]] = {}
        if args.eval_manifests:
            names = args.eval_names or [Path(p).stem for p in args.eval_manifests]
            for name, path in zip(names, args.eval_manifests):
                extra_eval_records[name] = load_jsonl(path)

        # Pre-split records by source_subset for per-epoch evaluation
        valid_by_subset = split_records_by_source_subset(valid_records) if valid_records else {}
        test_by_subset = split_records_by_source_subset(test_records) if test_records else {}

        tokenizer = resolve_training_tokenizer(
            train_records,
            args=args,
            train_manifest=args.train_manifest,
            output_dir=output_dir,
        )
        tokenizer.save(output_dir / "vocab.json")

        device = choose_device(args.device)
        print(f"Using device: {device}", flush=True)
        n_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs with DataParallel.", flush=True)
        use_amp, use_tf32 = configure_runtime(device, args)
        print(f"Runtime options: amp={use_amp} tf32={use_tf32}", flush=True)
        scaler = create_grad_scaler(use_amp=use_amp, is_cuda=device.startswith("cuda"))

        train_dataset = build_dataset_for_mode(
            train_records, tokenizer=tokenizer, sample_rate=args.sample_rate,
            args=args, manifest_path=args.train_manifest, device=device,
        )
        valid_dataset = build_dataset_for_mode(
            valid_records, tokenizer=tokenizer, sample_rate=args.sample_rate,
            args=args, manifest_path=args.valid_manifest, device=device,
        )
        test_dataset = (
            build_dataset_for_mode(
                test_records, tokenizer=tokenizer, sample_rate=args.sample_rate,
                args=args, manifest_path=args.test_manifest, device=device,
            )
            if test_records else None
        )

        collate_fn = build_collate_fn(args, tokenizer)
        is_cuda = device.startswith("cuda")
        data_on_device = dataset_storage_device(train_dataset).startswith("cuda")
        memory_resident = is_memory_resident_dataset(train_dataset)
        loader_kwargs = resolve_loader_kwargs(
            args, is_cuda=is_cuda, data_on_device=data_on_device, memory_resident=memory_resident,
        )

        max_tokens_per_batch = max(0, int(getattr(args, "max_tokens_per_batch", 0)))
        if max_tokens_per_batch > 0:
            train_batch_sampler = DynamicBatchSampler(
                resolve_dataset_length_hints(train_dataset, args), max_tokens_per_batch,
                shuffle=True, seed=args.seed,
            )
            valid_batch_sampler = DynamicBatchSampler(
                resolve_dataset_length_hints(valid_dataset, args), max_tokens_per_batch,
                shuffle=False, seed=args.seed,
            )
            test_batch_sampler = (
                DynamicBatchSampler(
                    resolve_dataset_length_hints(test_dataset, args), max_tokens_per_batch,
                    shuffle=False, seed=args.seed,
                ) if test_dataset else None
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_sampler=train_batch_sampler,
                collate_fn=collate_fn, **loader_kwargs,
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_sampler=valid_batch_sampler,
                collate_fn=collate_fn, **loader_kwargs,
            )
            test_loader = (
                torch.utils.data.DataLoader(
                    test_dataset, batch_sampler=test_batch_sampler,
                    collate_fn=collate_fn, **loader_kwargs,
                ) if test_dataset and test_batch_sampler else None
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_fn, **loader_kwargs,
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collate_fn, **loader_kwargs,
            )
            test_loader = (
                torch.utils.data.DataLoader(
                    test_dataset, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_fn, **loader_kwargs,
                ) if test_dataset else None
            )

        model = EncoderMoECTCModel(args, vocab_size=len(tokenizer.id_to_token)).to(device)
        raw_model = model
        if n_gpus > 1:
            model = nn.DataParallel(model)
        ema = ModelEMA(raw_model, decay=args.ema_decay) if float(getattr(args, "ema_decay", 0.0)) > 0.0 else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_lr_scheduler(optimizer, args, len(train_loader))
        ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True, reduction="mean")

        best_valid_cer = float("inf")
        no_improve_rounds = 0
        eval_every = max(1, int(args.eval_every_epochs))
        patience = max(0, int(args.early_stop_patience))
        history: list[dict[str, Any]] = []

        # ── Training loop ─────────────────────────────────────────────────
        for epoch in range(1, args.epochs + 1):
            print(f"Starting epoch {epoch}/{args.epochs}", flush=True)
            train_metrics = train_one_epoch(
                model=model, loader=train_loader, tokenizer=tokenizer,
                optimizer=optimizer, ctc_loss=ctc_loss, args=args, device=device,
                epoch=epoch, wandb_run=wandb_run, scaler=scaler, use_amp=use_amp,
                scheduler=scheduler, ema=ema, ema_source_model=raw_model,
            )

            if int(getattr(args, "temperature_anneal_epochs", 0)) > 0:
                new_temp = get_annealed_temperature(args, epoch)
                for moe_module in raw_model.get_moe_modules():
                    moe_module.temperature = new_temp

            evolve_logs: list[dict[str, Any]] = []
            if should_run_expert_evolution(args, epoch):
                evolve_logs = evolve_experts(raw_model, valid_loader, ctc_loss, args, device, use_amp=use_amp)
                if evolve_logs:
                    save_json(output_dir / f"expert_evolution_epoch_{epoch}.json", {"events": evolve_logs})

            should_eval = (epoch % eval_every == 0) or (epoch == args.epochs)
            if not should_eval:
                history.append({"epoch": epoch, "train_loss": round(train_metrics["loss"], 6)})
                log_wandb_metrics(wandb_run, {
                    "global_step": epoch * len(train_loader), "epoch": epoch,
                    "train/loss_epoch": train_metrics["loss"],
                    "train/lr": train_metrics["lr"],
                })
                continue

            eval_model = ema.ema_model if ema is not None and ema.ready else model
            valid_metrics = evaluate(
                model=eval_model, loader=valid_loader, tokenizer=tokenizer,
                ctc_loss=ctc_loss, args=args, device=device,
                stage=f"valid e{epoch}", use_amp=use_amp, epoch=epoch,
            )
            print(
                f"epoch={epoch} valid_loss={valid_metrics['loss']:.4f} "
                f"valid_cer={valid_metrics['cer']:.4f} valid_wer={valid_metrics['wer']:.4f}",
                flush=True,
            )

            epoch_rec = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "valid_loss": round(valid_metrics["loss"], 6),
                "valid_cer": round(valid_metrics["cer"], 6),
                "valid_wer": round(valid_metrics["wer"], 6),
            }
            history.append(epoch_rec)

            # ── Per-subset evaluation each eval epoch ─────────────────
            epoch_subset_metrics: dict[str, dict[str, Any]] = {}
            if valid_by_subset:
                epoch_subset_metrics.update(evaluate_subsets(
                    eval_model, valid_by_subset, tokenizer, ctc_loss, args,
                    device, use_amp, epoch=epoch,
                ))
            if test_by_subset:
                epoch_subset_metrics.update(evaluate_subsets(
                    eval_model, test_by_subset, tokenizer, ctc_loss, args,
                    device, use_amp, epoch=epoch,
                ))

            # Print per-subset WER summary for this epoch
            if epoch_subset_metrics:
                print(f"  ── Epoch {epoch} per-subset WER ──", flush=True)
                for sname in sorted(epoch_subset_metrics.keys()):
                    sm = epoch_subset_metrics[sname]
                    print(f"    {sname:<20} WER={sm['wer']:.4f}  CER={sm['cer']:.4f}", flush=True)

            is_best = valid_metrics["cer"] < best_valid_cer
            log_payload = {
                "global_step": epoch * len(train_loader), "epoch": epoch,
                "train/loss_epoch": train_metrics["loss"],
                "train/lr": train_metrics["lr"],
                "valid/loss": valid_metrics["loss"],
                "valid/cer": valid_metrics["cer"],
                "valid/wer": valid_metrics["wer"],
                "valid/is_best": int(is_best),
            }
            # Log per-subset metrics to wandb
            for sname, sm in epoch_subset_metrics.items():
                safe = sname.replace("-", "_")
                log_payload[f"eval/{safe}/wer"] = sm["wer"]
                log_payload[f"eval/{safe}/cer"] = sm["cer"]
                log_payload[f"eval/{safe}/mean_wer"] = sm["mean_wer"]
                log_payload[f"eval/{safe}/mean_cer"] = sm["mean_cer"]
                log_payload[f"eval/{safe}/loss"] = sm["loss"]
            log_wandb_metrics(wandb_run, log_payload)

            if is_best:
                best_valid_cer = valid_metrics["cer"]
                no_improve_rounds = 0
                torch.save({
                    "model_state": raw_model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "config": vars(args),
                    "vocab": tokenizer.id_to_token,
                    "best_valid_cer": best_valid_cer,
                    "ema_model_state": ema.state_dict() if ema is not None else None,
                }, output_dir / "best.pt")
                save_json(output_dir / "best_valid_metrics.json", valid_metrics)
            else:
                no_improve_rounds += 1
                if patience > 0 and no_improve_rounds >= patience:
                    print(f"Early stopping at epoch {epoch}.", flush=True)
                    break

        save_json(output_dir / "train_history.json", {"epochs": history})

        # ── Per-subset evaluation with best checkpoint ────────────────────
        if (output_dir / "best.pt").exists():
            checkpoint = torch.load(output_dir / "best.pt", map_location=device)
            raw_model.load_state_dict(checkpoint["model_state"])
            if ema is not None and checkpoint.get("ema_model_state") is not None:
                ema.load_state_dict(checkpoint["ema_model_state"])
            final_model = ema.ema_model if ema is not None and ema.ready else model
            final_epoch = history[-1]["epoch"] if history else args.epochs

            # Standard test set evaluation
            if test_loader is not None:
                test_metrics = evaluate(
                    model=final_model, loader=test_loader, tokenizer=tokenizer,
                    ctc_loss=ctc_loss, args=args, device=device,
                    stage="test", use_amp=use_amp, epoch=final_epoch,
                )
                save_json(output_dir / "test_metrics.json", test_metrics)
                print(
                    f"test_cer={test_metrics['cer']:.4f} test_wer={test_metrics['wer']:.4f}",
                    flush=True,
                )

            # Per-subset evaluation: split valid and test records by source_subset
            print("\n=== Per-subset evaluation ===", flush=True)
            all_subset_metrics: dict[str, dict[str, Any]] = {}

            # Split validation records by source_subset (dev-clean, dev-other)
            if valid_records:
                valid_by_subset = split_records_by_source_subset(valid_records)
                print(f"Validation subsets: {list(valid_by_subset.keys())}", flush=True)
                valid_subset_metrics = evaluate_subsets(
                    final_model, valid_by_subset, tokenizer, ctc_loss, args,
                    device, use_amp, epoch=final_epoch,
                )
                all_subset_metrics.update(valid_subset_metrics)

            # Split test records by source_subset (test-clean, test-other)
            if test_records:
                test_by_subset = split_records_by_source_subset(test_records)
                print(f"Test subsets: {list(test_by_subset.keys())}", flush=True)
                test_subset_metrics = evaluate_subsets(
                    final_model, test_by_subset, tokenizer, ctc_loss, args,
                    device, use_amp, epoch=final_epoch,
                )
                all_subset_metrics.update(test_subset_metrics)

            # Evaluate extra manifests if provided
            if extra_eval_records:
                print(f"Extra eval sets: {list(extra_eval_records.keys())}", flush=True)
                extra_metrics = evaluate_subsets(
                    final_model, extra_eval_records, tokenizer, ctc_loss, args,
                    device, use_amp, epoch=final_epoch,
                )
                all_subset_metrics.update(extra_metrics)

            # Save all per-subset metrics
            save_json(output_dir / "subset_metrics.json", {
                name: {
                    "cer": round(m["cer"], 6),
                    "wer": round(m["wer"], 6),
                    "mean_cer": round(m["mean_cer"], 6),
                    "mean_wer": round(m["mean_wer"], 6),
                    "corpus_cer": round(m["corpus_cer"], 6),
                    "corpus_wer": round(m["corpus_wer"], 6),
                    "loss": round(m["loss"], 6),
                }
                for name, m in all_subset_metrics.items()
            })

            # Print summary table
            print("\n=== LibriSpeech Evaluation Summary ===", flush=True)
            print(f"{'Subset':<20} {'CER':>8} {'WER':>8}", flush=True)
            print("-" * 38, flush=True)
            for name in sorted(all_subset_metrics.keys()):
                m = all_subset_metrics[name]
                print(f"{name:<20} {m['cer']:>8.4f} {m['wer']:>8.4f}", flush=True)
            print("-" * 38, flush=True)

            # Log to wandb
            for name, m in all_subset_metrics.items():
                safe_name = name.replace("-", "_")
                log_wandb_metrics(wandb_run, {
                    f"eval/{safe_name}/cer": m["cer"],
                    f"eval/{safe_name}/wer": m["wer"],
                    f"eval/{safe_name}/mean_cer": m["mean_cer"],
                    f"eval/{safe_name}/mean_wer": m["mean_wer"],
                })

    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
