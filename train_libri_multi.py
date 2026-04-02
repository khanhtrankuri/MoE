"""train_libri_multi.py – Multi-GPU (Expert Parallel) MoE ASR training on LibriSpeech.

Trains on train-clean-100 (by default), validates on dev-clean,
and evaluates individually on dev-clean, dev-other, test-clean, test-other.

Usage
-----
Single GPU:
    python train_libri_multi.py --train-manifest ... --output-dir ...

Multi-GPU Expert Parallel:
    torchrun --nproc_per_node=4 train_libri_multi.py --expert-parallel \
        --train-manifest ... --output-dir ...

Multi-GPU Data Parallel:
    torchrun --nproc_per_node=4 train_libri_multi.py --no-expert-parallel \
        --train-manifest ... --output-dir ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    dist = None
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

# Re-use model architecture and helpers from train_arch_moe_mutil
from train_arch_moe_mutil import (
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
    precompute_shared_params,
    _EP_EXPERT_SUBMODULES,
    _dist_active,
    parse_args as _base_parse_args,
)


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    args = _base_parse_args()
    import sys
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--eval-manifests", nargs="*", default=None,
                              help="Additional JSONL manifests to evaluate individually.")
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
    world_size: int = 1,
    rank: int = 0,
) -> dict[str, dict[str, Any]]:
    """Evaluate model on multiple named subsets independently."""
    all_metrics = {}
    for name, records in records_by_name.items():
        dataset = build_dataset_for_mode(
            records, tokenizer=tokenizer, sample_rate=args.sample_rate,
            args=args, manifest_path=None, device=device,
        )
        collate_fn = build_collate_fn(args, tokenizer)
        is_cuda = device.startswith("cuda")
        data_on_device = dataset_storage_device(dataset).startswith("cuda")
        memory_resident = is_memory_resident_dataset(dataset)
        loader_kw = resolve_loader_kwargs(
            args, is_cuda=is_cuda, data_on_device=data_on_device, memory_resident=memory_resident,
        )

        max_tokens = max(0, int(getattr(args, "max_tokens_per_batch", 0)))
        if max_tokens > 0:
            batch_sampler = DynamicBatchSampler(
                resolve_dataset_length_hints(dataset, args), max_tokens,
                shuffle=False, seed=args.seed,
                num_replicas=world_size, rank=rank,
            )
            loader = torch.utils.data.DataLoader(
                dataset, batch_sampler=batch_sampler,
                collate_fn=collate_fn, **loader_kw,
            )
        else:
            if world_size > 1:
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
            else:
                sampler = None
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size,
                shuffle=False, sampler=sampler,
                collate_fn=collate_fn, **loader_kw,
            )

        metrics = evaluate(
            model=model, loader=loader, tokenizer=tokenizer,
            ctc_loss=ctc_loss, args=args, device=device,
            stage=name, use_amp=use_amp, epoch=epoch,
            world_size=world_size, rank=rank,
        )
        all_metrics[name] = metrics
        if rank == 0:
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

    # ── Distributed / EP setup ─────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = rank == 0
    use_ep = bool(getattr(args, "expert_parallel", True)) and world_size > 1

    if torch.cuda.is_available() and "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if world_size > 1:
        _backend_arg = getattr(args, "dist_backend", "auto")
        _backend = ("nccl" if torch.cuda.is_available() else "gloo") if _backend_arg == "auto" else _backend_arg
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        else:
            device = "cpu"
        if _backend == "nccl":
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
        import datetime
        dist.init_process_group(backend=_backend, timeout=datetime.timedelta(seconds=1800))
        if is_main:
            print(f"Distributed backend: {_backend}, world_size={world_size}", flush=True)
        if use_ep:
            if args.ffn_type == "shared_adapter_moe" and args.num_experts % world_size != 0:
                raise ValueError(
                    f"--num-experts {args.num_experts} must be divisible by world_size {world_size}."
                )
            if float(getattr(args, "layer_drop", 0.0)) > 0:
                args.layer_drop = 0.0
    else:
        device = choose_device(args.device)

    set_seed(args.seed)

    # ── Output dir ─────────────────────────────────────────────────────────
    if is_main:
        output_dir = prepare_output_dir(
            Path(args.output_dir).resolve(),
            allow_existing=bool(getattr(args, "allow_existing_output_dir", False)),
        )
        save_json(output_dir / "config.json", vars(args))
    else:
        output_dir = Path(args.output_dir).resolve()
    if world_size > 1:
        dist.barrier()

    wandb_run = init_wandb_run(args, output_dir, vars(args)) if is_main else None

    try:
        train_records = load_jsonl(args.train_manifest)
        valid_records = load_jsonl(args.valid_manifest)
        test_records = load_jsonl(args.test_manifest) if args.test_manifest else None

        extra_eval_records: dict[str, list[dict]] = {}
        if args.eval_manifests:
            names = args.eval_names or [Path(p).stem for p in args.eval_manifests]
            for name, path in zip(names, args.eval_manifests):
                extra_eval_records[name] = load_jsonl(path)

        tokenizer = resolve_training_tokenizer(
            train_records, args=args,
            train_manifest=args.train_manifest, output_dir=output_dir,
        )
        if is_main:
            tokenizer.save(output_dir / "vocab.json")

        use_amp, use_tf32 = configure_runtime(device, args)
        scaler = create_grad_scaler(use_amp=use_amp, is_cuda=device.startswith("cuda"))

        # ── Datasets & loaders ────────────────────────────────────────────
        def _make_ds(records, manifest_path):
            return build_dataset_for_mode(
                records, tokenizer=tokenizer, sample_rate=args.sample_rate,
                args=args, manifest_path=manifest_path, device=device,
            )

        train_ds = _make_ds(train_records, args.train_manifest)
        valid_ds = _make_ds(valid_records, args.valid_manifest)
        test_ds = _make_ds(test_records, args.test_manifest) if test_records else None

        collate_fn = build_collate_fn(args, tokenizer)
        is_cuda = device.startswith("cuda")
        data_on_device = dataset_storage_device(train_ds).startswith("cuda")
        mem_resident = is_memory_resident_dataset(train_ds)
        loader_kw = resolve_loader_kwargs(args, is_cuda=is_cuda,
                                          data_on_device=data_on_device, memory_resident=mem_resident)

        max_tokens_per_batch = max(0, int(getattr(args, "max_tokens_per_batch", 0)))
        if max_tokens_per_batch > 0:
            train_sampler = DynamicBatchSampler(
                resolve_dataset_length_hints(train_ds, args), max_tokens_per_batch,
                shuffle=True, seed=args.seed, num_replicas=world_size, rank=rank,
            )
            valid_sampler = DynamicBatchSampler(
                resolve_dataset_length_hints(valid_ds, args), max_tokens_per_batch,
                shuffle=False, seed=args.seed, num_replicas=world_size, rank=rank,
            )
            test_sampler = (
                DynamicBatchSampler(
                    resolve_dataset_length_hints(test_ds, args), max_tokens_per_batch,
                    shuffle=False, seed=args.seed, num_replicas=world_size, rank=rank,
                ) if test_ds else None
            )
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_sampler=train_sampler, collate_fn=collate_fn, **loader_kw)
            valid_loader = torch.utils.data.DataLoader(
                valid_ds, batch_sampler=valid_sampler, collate_fn=collate_fn, **loader_kw)
            test_loader = (
                torch.utils.data.DataLoader(test_ds, batch_sampler=test_sampler, collate_fn=collate_fn, **loader_kw)
                if test_ds and test_sampler else None
            )
        else:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) \
                if world_size > 1 else None
            valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False) \
                if world_size > 1 else None
            test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False) \
                if world_size > 1 and test_ds else None

            def _make_loader(ds, sampler, shuffle_flag):
                return torch.utils.data.DataLoader(
                    ds, batch_size=args.batch_size,
                    shuffle=(sampler is None and shuffle_flag),
                    sampler=sampler, collate_fn=collate_fn, **loader_kw,
                )

            train_loader = _make_loader(train_ds, train_sampler, True)
            valid_loader = _make_loader(valid_ds, valid_sampler, False)
            test_loader = _make_loader(test_ds, test_sampler, False) if test_ds else None

        # ── Model ─────────────────────────────────────────────────────────
        model = EncoderMoECTCModel(
            args, vocab_size=len(tokenizer.id_to_token),
            rank=rank, world_size=world_size if use_ep else 1,
        ).to(device)
        raw_model = model

        if world_size > 1 and not use_ep:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            raw_model = model.module

        ema = ModelEMA(raw_model, decay=args.ema_decay) if float(getattr(args, "ema_decay", 0.0)) > 0.0 else None
        _shared_params = precompute_shared_params(raw_model) if use_ep else None

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_lr_scheduler(optimizer, args, len(train_loader))
        ctc_loss_fn = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True, reduction="mean")

        # ── Resume ────────────────────────────────────────────────────────
        start_epoch = 1
        best_cer = float("inf")
        no_imp = 0
        if args.resume:
            ckpt = torch.load(Path(args.resume), map_location=device)
            raw_model.load_state_dict(ckpt["model_state"], strict=False)
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None and scheduler is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if ema is not None and ckpt.get("ema_model_state") is not None:
                ema.load_state_dict(ckpt["ema_model_state"])
            if "best_valid_cer" in ckpt:
                best_cer = ckpt["best_valid_cer"]
            if "epoch" in ckpt:
                start_epoch = ckpt["epoch"] + 1
            if is_main:
                print(f"Resumed from {args.resume} (epoch {start_epoch - 1}).", flush=True)
            if world_size > 1:
                dist.barrier()

        # ── Training loop ─────────────────────────────────────────────────
        eval_every = max(1, int(args.eval_every_epochs))
        patience = max(0, int(args.early_stop_patience))
        history: list = []

        # ── Prepare results.txt ──────────────────────────────────────────
        results_path = output_dir / "results.txt"
        if is_main:
            write_header = not results_path.exists() or start_epoch == 1
            if write_header:
                with open(results_path, "w") as rf:
                    rf.write(f"{'Epoch':>6}  {'train_loss':>11}  {'valid_loss':>11}  "
                             f"{'valid_CER':>10}  {'valid_WER':>10}  |  "
                             f"{'dev-clean CER':>14}  {'dev-clean WER':>14}  "
                             f"{'dev-other CER':>14}  {'dev-other WER':>14}  "
                             f"{'test-clean CER':>15}  {'test-clean WER':>15}  "
                             f"{'test-other CER':>15}  {'test-other WER':>15}\n")
                    rf.write("-" * 200 + "\n")

        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None and hasattr(valid_sampler, "set_epoch"):
                valid_sampler.set_epoch(epoch)

            if is_main:
                print(f"Starting epoch {epoch}/{args.epochs}", flush=True)

            train_m = train_one_epoch(
                model=model, loader=train_loader, tokenizer=tokenizer,
                optimizer=optimizer, ctc_loss=ctc_loss_fn, args=args, device=device,
                epoch=epoch, wandb_run=wandb_run, scaler=scaler, use_amp=use_amp,
                scheduler=scheduler, world_size=world_size, rank=rank,
                shared_params=_shared_params, ema=ema, ema_source_model=raw_model,
            )

            if int(getattr(args, "temperature_anneal_epochs", 0)) > 0:
                new_t = get_annealed_temperature(args, epoch)
                for m in raw_model.get_moe_modules():
                    m.temperature = new_t

            evolve_logs: list = []
            if should_run_expert_evolution(args, epoch):
                evolve_logs = evolve_experts(raw_model, valid_loader, ctc_loss_fn, args, device, use_amp)
                if evolve_logs and is_main:
                    save_json(output_dir / f"expert_evolution_epoch_{epoch}.json", {"events": evolve_logs})
                if world_size > 1:
                    dist.barrier()

            should_eval = (epoch % eval_every == 0) or (epoch == args.epochs)
            if not should_eval:
                if is_main:
                    history.append({"epoch": epoch, "train_loss": round(train_m["loss"], 6)})
                    log_wandb_metrics(wandb_run, {
                        "global_step": epoch * len(train_loader), "epoch": epoch,
                        "train/loss_epoch": train_m["loss"], "train/lr": train_m["lr"],
                    })
                continue

            eval_model = ema.ema_model if ema is not None and ema.ready else model
            valid_m = evaluate(
                model=eval_model, loader=valid_loader, tokenizer=tokenizer,
                ctc_loss=ctc_loss_fn, args=args, device=device,
                stage=f"valid e{epoch}", use_amp=use_amp, epoch=epoch,
                world_size=world_size, rank=rank,
            )

            if is_main:
                print(
                    f"epoch={epoch} valid_loss={valid_m['loss']:.4f} "
                    f"valid_cer={valid_m['cer']:.4f} valid_wer={valid_m['wer']:.4f}",
                    flush=True,
                )

            # ── Per-subset evaluation each eval epoch ─────────────────
            valid_by_subset = split_records_by_source_subset(valid_records) if valid_records else {}
            test_by_subset = split_records_by_source_subset(test_records) if test_records else {}

            epoch_subset_metrics: dict[str, dict[str, Any]] = {}
            if valid_by_subset:
                epoch_subset_metrics.update(evaluate_subsets(
                    eval_model, valid_by_subset, tokenizer, ctc_loss_fn, args,
                    device, use_amp, epoch=epoch,
                    world_size=world_size, rank=rank,
                ))
            if test_by_subset:
                epoch_subset_metrics.update(evaluate_subsets(
                    eval_model, test_by_subset, tokenizer, ctc_loss_fn, args,
                    device, use_amp, epoch=epoch,
                    world_size=world_size, rank=rank,
                ))

            # Print per-subset WER summary for this epoch
            if is_main and epoch_subset_metrics:
                print(f"  ── Epoch {epoch} per-subset WER ──", flush=True)
                for sname in sorted(epoch_subset_metrics.keys()):
                    sm = epoch_subset_metrics[sname]
                    print(f"    {sname:<20} WER={sm['wer']:.4f}  CER={sm['cer']:.4f}", flush=True)

            # ── Write epoch results to results.txt ───────────────────
            if is_main:
                subset_order = ["dev-clean", "dev-other", "test-clean", "test-other"]
                with open(results_path, "a") as rf:
                    line = (f"{epoch:>6}  {train_m['loss']:>11.6f}  {valid_m['loss']:>11.6f}  "
                            f"{valid_m['cer']:>10.4f}  {valid_m['wer']:>10.4f}  |")
                    for sname in subset_order:
                        sm = epoch_subset_metrics.get(sname, {})
                        cer_val = sm.get("cer", float("nan"))
                        wer_val = sm.get("wer", float("nan"))
                        line += f"  {cer_val:>14.4f}  {wer_val:>14.4f}"
                    rf.write(line + "\n")

            is_best = valid_m["cer"] < best_cer
            if is_main:
                history.append({
                    "epoch": epoch,
                    "train_loss": round(train_m["loss"], 6),
                    "valid_loss": round(valid_m["loss"], 6),
                    "valid_cer": round(valid_m["cer"], 6),
                    "valid_wer": round(valid_m["wer"], 6),
                })
                log_payload = {
                    "global_step": epoch * len(train_loader), "epoch": epoch,
                    "train/loss_epoch": train_m["loss"], "train/lr": train_m["lr"],
                    "valid/loss": valid_m["loss"], "valid/cer": valid_m["cer"],
                    "valid/wer": valid_m["wer"], "valid/is_best": int(is_best),
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
                best_cer = valid_m["cer"]
                no_imp = 0
                if is_main:
                    torch.save({
                        "epoch": epoch,
                        "model_state": raw_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler else None,
                        "scaler_state": scaler.state_dict() if scaler is not None else None,
                        "config": vars(args),
                        "vocab": tokenizer.id_to_token,
                        "best_valid_cer": best_cer,
                        "expert_parallel": use_ep,
                        "world_size": world_size,
                        "ema_model_state": ema.state_dict() if ema is not None else None,
                    }, output_dir / "best.pt")
                    save_json(output_dir / "best_valid_metrics.json", valid_m)
            else:
                no_imp += 1
                if patience > 0 and no_imp >= patience:
                    if is_main:
                        print(f"Early stopping at epoch {epoch}.", flush=True)
                    break

        if is_main:
            save_json(output_dir / "train_history.json", {"epochs": history})

        # ── Per-subset evaluation with best checkpoint ────────────────────
        if (output_dir / "best.pt").exists():
            ckpt = torch.load(output_dir / "best.pt", map_location=device)
            raw_model.load_state_dict(ckpt["model_state"], strict=False)
            if ema is not None and ckpt.get("ema_model_state") is not None:
                ema.load_state_dict(ckpt["ema_model_state"])
            final_model = ema.ema_model if ema is not None and ema.ready else model
            final_epoch = history[-1]["epoch"] if history else args.epochs

            # Standard test evaluation
            if test_loader is not None:
                test_m = evaluate(
                    model=final_model, loader=test_loader, tokenizer=tokenizer,
                    ctc_loss=ctc_loss_fn, args=args, device=device, stage="test",
                    use_amp=use_amp, epoch=final_epoch,
                    world_size=world_size, rank=rank,
                )
                if is_main:
                    save_json(output_dir / "test_metrics.json", test_m)
                    print(f"test_cer={test_m['cer']:.4f} test_wer={test_m['wer']:.4f}", flush=True)

            # Per-subset evaluation
            if is_main:
                print("\n=== Per-subset evaluation ===", flush=True)
            all_subset_metrics: dict[str, dict[str, Any]] = {}

            if valid_records:
                valid_by_subset = split_records_by_source_subset(valid_records)
                if is_main:
                    print(f"Validation subsets: {list(valid_by_subset.keys())}", flush=True)
                valid_subset_metrics = evaluate_subsets(
                    final_model, valid_by_subset, tokenizer, ctc_loss_fn, args,
                    device, use_amp, epoch=final_epoch,
                    world_size=world_size, rank=rank,
                )
                all_subset_metrics.update(valid_subset_metrics)

            if test_records:
                test_by_subset = split_records_by_source_subset(test_records)
                if is_main:
                    print(f"Test subsets: {list(test_by_subset.keys())}", flush=True)
                test_subset_metrics = evaluate_subsets(
                    final_model, test_by_subset, tokenizer, ctc_loss_fn, args,
                    device, use_amp, epoch=final_epoch,
                    world_size=world_size, rank=rank,
                )
                all_subset_metrics.update(test_subset_metrics)

            if extra_eval_records:
                if is_main:
                    print(f"Extra eval sets: {list(extra_eval_records.keys())}", flush=True)
                extra_metrics = evaluate_subsets(
                    final_model, extra_eval_records, tokenizer, ctc_loss_fn, args,
                    device, use_amp, epoch=final_epoch,
                    world_size=world_size, rank=rank,
                )
                all_subset_metrics.update(extra_metrics)

            if is_main:
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

                print("\n=== LibriSpeech Evaluation Summary ===", flush=True)
                print(f"{'Subset':<20} {'CER':>8} {'WER':>8}", flush=True)
                print("-" * 38, flush=True)
                for name in sorted(all_subset_metrics.keys()):
                    m = all_subset_metrics[name]
                    print(f"{name:<20} {m['cer']:>8.4f} {m['wer']:>8.4f}", flush=True)
                print("-" * 38, flush=True)

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
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
