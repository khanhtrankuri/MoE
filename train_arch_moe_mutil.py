"""train_arch_moe_ep.py  – Expert-Parallel MoE ASR training.

Usage
-----
Single GPU (unchanged behaviour):
    python train_arch_moe_ep.py --train-manifest ... --output-dir ...

Data Parallel only (all experts replicated on every GPU):
    torchrun --nproc_per_node=4 train_arch_moe_ep.py --no-expert-parallel ...

Expert Parallel (experts sharded across GPUs – the "true MoE" mode):
    torchrun --nproc_per_node=4 train_arch_moe_ep.py --expert-parallel ...
    # num_experts MUST be divisible by nproc_per_node.

Expert Parallel, multi-node:
    torchrun --nnodes=2 --nproc_per_node=4 \
             --rdzv_backend=c10d --rdzv_endpoint=host0:29400 \
             train_arch_moe_ep.py --expert-parallel ...
"""

from __future__ import annotations

import argparse
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
    ensure_torch,
    finish_wandb_run,
    flatten_routing_metrics,
    is_memory_resident_dataset,
    init_wandb_run,
    lengths_to_mask,
    load_jsonl,
    log_wandb_metrics,
    move_batch_to_device,
    normalize_eval_text,
    prepare_output_dir,
    resolve_loader_kwargs,
    resolve_training_tokenizer,
    save_json,
    select_hypotheses,
    set_seed,
    spec_augment,
    summarize_routing,
    synchronize_for_timing,
)


# ===========================================================================
# Argument parsing
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Transformer/Conformer ASR model with Competitive-Attractive "
                    "SharedAdapterMoE – Expert Parallel edition."
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--valid-manifest", required=True)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-existing-output-dir", action="store_true")
    parser.add_argument("--encoder-type", choices=("transformer", "conformer"), default="transformer")
    parser.add_argument("--ffn-type", choices=("dense", "shared_adapter_moe"), default="shared_adapter_moe")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2, help="Number of experts activated per frame (Top-K routing)")
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--load-balance-weight", type=float, default=0.01)
    parser.add_argument("--competition-weight", type=float, default=0.05)
    parser.add_argument("--competition-score", choices=("exp_neg_loss", "inverse_loss"), default="exp_neg_loss")
    parser.add_argument("--competition-epsilon", type=float, default=1e-6)
    parser.add_argument("--competition-batches", type=int, default=0)
    parser.add_argument("--competition-on-valid", action="store_true")
    parser.add_argument("--competition-interval-steps", type=int, default=1)
    parser.add_argument("--competition-warmup-epochs", type=int, default=0)
    parser.add_argument("--competition-ramp-epochs", type=int, default=3)
    parser.add_argument("--expert-evolve-every-epochs", type=int, default=0)
    parser.add_argument("--expert-evolve-start-epoch", type=int, default=3)
    parser.add_argument("--expert-merge-alpha", type=float, default=0.5)
    parser.add_argument("--expert-merge-split-ratio", type=float, default=0.5)
    parser.add_argument("--expert-merge-replace", choices=("worst", "random", "redundant"), default="worst")
    parser.add_argument("--expert-merge-blocks", type=int, default=0)
    # ── Expert Parallelism ────────────────────────────────────────────────
    parser.add_argument(
        "--expert-parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Expert Parallelism: shard experts across GPUs. "
             "num_experts must be divisible by world_size. "
             "Use --no-expert-parallel to fall back to Data Parallel.",
    )
    parser.add_argument(
        "--dist-backend",
        choices=("nccl", "gloo", "auto"),
        default="auto",
        help="Distributed backend. 'auto' uses nccl when CUDA is available, "
             "gloo otherwise. Use 'gloo' on Kaggle/environments with NCCL issues.",
    )
    # ── Training ─────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    add_data_pipeline_args(parser)
    add_tokenizer_args(parser)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("none", "warmup_cosine"), default="warmup_cosine")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--entropy-bonus-weight", type=float, default=0.0)
    parser.add_argument("--temperature-anneal-start", type=float, default=None)
    parser.add_argument("--temperature-anneal-end", type=float, default=None)
    parser.add_argument("--temperature-anneal-epochs", type=int, default=0)
    parser.add_argument("--spec-augment", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--freq-mask-param", type=int, default=27)
    parser.add_argument("--time-mask-param", type=int, default=100)
    parser.add_argument("--num-freq-masks", type=int, default=2)
    parser.add_argument("--num-time-masks", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--decode-mode", choices=("greedy", "beam"), default="greedy")
    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-dim", type=int, default=1024)
    parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--projector-dim", type=int, default=256)
    parser.add_argument("--conv-kernel-size", type=int, default=15)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=160)
    parser.add_argument("--win-length", type=int, default=400)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--max-audio-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--amp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--tf32", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--wandb-project", default="moe-asr")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default="arch-casamoe-ep")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--normalize-eval-text", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--layer-drop", type=float, default=0.1)
    parser.add_argument("--intermediate-ctc-weight", type=float, default=0.3)
    parser.add_argument("--intermediate-ctc-layer", type=int, default=0)
    parser.set_defaults(tokenizer_type="grapheme")
    add_profiling_args(parser)
    return parser.parse_args()


# ===========================================================================
# Distributed helpers
# ===========================================================================

def _dist_active() -> bool:
    return dist is not None and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _all_reduce_scalar(value: float, device: str) -> float:
    if not _dist_active():
        return value
    t = torch.tensor(value, dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _all_reduce_tensor(t: torch.Tensor, device: str) -> torch.Tensor:
    if not _dist_active():
        return t
    buf = t.to(device)
    dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    return buf.cpu()


def _sync_batch_time_dims(batch_size: int, time_steps: int, device: str | torch.device) -> tuple[int, int]:
    """Return the max batch/time dimensions across the process group."""
    if not _dist_active():
        return batch_size, time_steps
    dims = torch.tensor([batch_size, time_steps], dtype=torch.long, device=device)
    dist.all_reduce(dims, op=dist.ReduceOp.MAX)
    return int(dims[0].item()), int(dims[1].item())


def _pad_batch_time_dims(
    tensor: torch.Tensor,
    target_batch: int,
    target_time: int,
) -> torch.Tensor:
    """Pad a (B, T, D) or (E, B, T, D) tensor up to a shared batch/time shape."""
    if tensor.dim() == 3:
        batch_size, time_steps = int(tensor.shape[0]), int(tensor.shape[1])
    elif tensor.dim() == 4:
        batch_size, time_steps = int(tensor.shape[1]), int(tensor.shape[2])
    else:
        raise ValueError(f"Expected a 3-D or 4-D tensor, got shape {tuple(tensor.shape)}")

    if target_batch < batch_size or target_time < time_steps:
        raise ValueError(
            f"Cannot pad tensor with shape {tuple(tensor.shape)} to smaller target "
            f"({target_batch}, {target_time})."
        )

    if target_time > time_steps:
        tensor = F.pad(tensor, (0, 0, 0, target_time - time_steps))
    if target_batch > batch_size:
        tensor = F.pad(tensor, (0, 0, 0, 0, 0, target_batch - batch_size))
    return tensor.contiguous()


# Names of parameter sub-modules that belong to experts (not shared).
# Used to skip all_reduce for these params during EP gradient reduction.
_EP_EXPERT_SUBMODULES = {"trunks", "share_down", "adapter_up", "adapter_down"}


def _reduce_non_expert_gradients(model: nn.Module, world_size: int) -> None:
    """In Expert Parallel mode we do NOT use DDP.
    Instead, manually all_reduce gradients of every param that is NOT
    an expert weight (router, encoder, CTC head, etc.).
    Expert weights live on exactly one GPU → no reduction needed.
    """
    if world_size <= 1:
        return
    for name, param in model.named_parameters():
        # e.g. "blocks.2.ffn.trunks.0.0.weight" → parts contain "trunks"
        if any(part in _EP_EXPERT_SUBMODULES for part in name.split(".")):
            continue  # expert param – stays local
        grad_present = torch.tensor(
            0 if param.grad is None else 1,
            dtype=torch.int64,
            device=param.device,
        )
        dist.all_reduce(grad_present, op=dist.ReduceOp.SUM)
        if int(grad_present.item()) == 0:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def _sync_batchnorm_buffers(model: nn.Module, world_size: int) -> None:
    """Keep BatchNorm running stats aligned in EP mode without SyncBatchNorm."""
    if world_size <= 1:
        return
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        if not module.track_running_stats:
            continue
        if module.running_mean is not None:
            dist.all_reduce(module.running_mean, op=dist.ReduceOp.SUM)
            module.running_mean.div_(world_size)
        if module.running_var is not None:
            dist.all_reduce(module.running_var, op=dist.ReduceOp.SUM)
            module.running_var.div_(world_size)
        if module.num_batches_tracked is not None:
            dist.all_reduce(module.num_batches_tracked, op=dist.ReduceOp.MAX)


# ===========================================================================
# Autograd-safe all_gather for expert outputs
# ===========================================================================

class _AllGatherGrad(torch.autograd.Function):
    """Differentiable all_gather across the expert-parallel group.

    Forward : concatenates local_tensor (local_E, B, T, D) from every rank
              into (num_experts, B, T, D) on each rank.
    Backward: slices the local-expert gradient, pads it back to the gathered
              shape, and all_reduces it so each expert receives contributions
              from every rank's local loss.
    """

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:  # type: ignore[override]
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.local_size = local_tensor.shape[0]
        ctx.input_batch = local_tensor.shape[1]
        ctx.input_time = local_tensor.shape[2]
        gather_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gather_list, local_tensor.contiguous())
        return torch.cat(gather_list, dim=0)  # (num_experts, B, T, D)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        rank = ctx.rank
        local_size = ctx.local_size
        local_grad = grad_output[rank * local_size:(rank + 1) * local_size].contiguous()
        local_grad = _pad_batch_time_dims(local_grad, ctx.input_batch, ctx.input_time)
        if ctx.world_size > 1:
            dist.all_reduce(local_grad, op=dist.ReduceOp.SUM)
        return local_grad, None, None


# ===========================================================================
# Expert-state broadcast utility (used in evolution)
# ===========================================================================

def _pack_expert_state(state: dict) -> tuple[torch.Tensor, list]:
    """Flatten an expert state-dict into a single 1-D float32 tensor."""
    tensors, meta = [], []
    for group_name in ("trunk", "share_down", "adapter_up", "adapter_down"):
        for k, v in state[group_name].items():
            tensors.append(v.float().reshape(-1))
            meta.append((group_name, k, v.shape, v.dtype))
    return torch.cat(tensors), meta


def _unpack_expert_state(flat: torch.Tensor, meta: list) -> dict:
    state: dict = {"trunk": {}, "share_down": {}, "adapter_up": {}, "adapter_down": {}}
    offset = 0
    for group_name, k, shape, dtype in meta:
        numel = 1
        for s in shape:
            numel *= s
        state[group_name][k] = flat[offset:offset + numel].view(shape).to(dtype)
        offset += numel
    return state


def _broadcast_expert_state(
    state: dict,
    owning_rank: int,
    device: str,
    meta_ref: list | None = None,
) -> dict:
    """Broadcast an expert's state from *owning_rank* to all other ranks.

    All ranks must call this with the same *owning_rank*.
    Non-owning ranks pass an empty (zeros) state packed with the correct *meta_ref*.
    """
    if not _dist_active():
        return state

    flat, meta = _pack_expert_state(state)
    flat = flat.to(device)
    dist.broadcast(flat, src=owning_rank)
    # Use meta from the owning rank (same shapes across all ranks due to same model arch)
    return _unpack_expert_state(flat.cpu(), meta)


# ===========================================================================
# Model architecture
# ===========================================================================

if TORCH_IMPORT_ERROR is None:

    class SinusoidalPositionalEncoding(nn.Module):
        def __init__(self, model_dim: int, max_len: int = 10000):
            super().__init__()
            positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim)
            )
            encoding = torch.zeros(max_len, model_dim, dtype=torch.float32)
            encoding[:, 0::2] = torch.sin(positions * div_term)
            encoding[:, 1::2] = torch.cos(positions * div_term)
            self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.encoding[:, : x.size(1)]

    class Conv2dSubsampling(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, dropout: float):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, output_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.GELU(),
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.GELU(),
            )
            reduced_freq = ((input_dim + 1) // 2 + 1) // 2
            self.out = nn.Sequential(nn.Linear(output_dim * reduced_freq, output_dim), nn.Dropout(dropout))

        def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
            return ((lengths + 1) // 2 + 1) // 2

        def forward(self, features: torch.Tensor, input_lengths: torch.Tensor):
            h = self.conv(features.unsqueeze(1))
            B, C, T, F = h.shape
            h = h.transpose(1, 2).contiguous().view(B, T, C * F)
            return self.out(h), self.output_lengths(input_lengths)

    class RelativePositionalEncoding(nn.Module):
        def __init__(self, model_dim: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.model_dim = model_dim
            self.dropout = nn.Dropout(dropout)
            pe = torch.zeros(max_len, model_dim)
            pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div = torch.exp(torch.arange(0, model_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / model_dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe, persistent=False)

        def forward(self, x: torch.Tensor):
            T = x.size(1)
            pe_pos = self.pe[:T]
            pe_neg = self.pe[1:T].clone()
            pe_neg[:, 0::2] = -pe_neg[:, 0::2]
            pos_enc = torch.cat([torch.flip(pe_pos, [0]), pe_neg], dim=0).unsqueeze(0)
            return self.dropout(x), self.dropout(pos_enc)

    class RelativeMultiHeadAttention(nn.Module):
        def __init__(self, model_dim: int, num_heads: int, dropout: float):
            super().__init__()
            assert model_dim % num_heads == 0
            self.num_heads, self.head_dim = num_heads, model_dim // num_heads
            self.model_dim = model_dim
            self.scale = self.head_dim ** -0.5
            self.w_q = nn.Linear(model_dim, model_dim)
            self.w_k = nn.Linear(model_dim, model_dim)
            self.w_v = nn.Linear(model_dim, model_dim)
            self.w_pos = nn.Linear(model_dim, model_dim, bias=False)
            self.w_out = nn.Linear(model_dim, model_dim)
            self.pos_bias_u = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
            self.pos_bias_v = nn.Parameter(torch.Tensor(num_heads, self.head_dim))
            nn.init.xavier_uniform_(self.pos_bias_u.unsqueeze(0))
            nn.init.xavier_uniform_(self.pos_bias_v.unsqueeze(0))
            self.attn_dropout = nn.Dropout(dropout)

        @staticmethod
        def _rel_shift(x: torch.Tensor) -> torch.Tensor:
            B, H, T, L = x.shape
            x = F.pad(x, (1, 0)).contiguous().view(B, H, L + 1, T)
            return x[:, :, 1:].contiguous().view(B, H, T, L)[:, :, :, :T]

        def forward(self, query, key, value, pos_enc, key_padding_mask=None):
            B, T, _ = query.shape
            q = self.w_q(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.w_k(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.w_v(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            p = self.w_pos(pos_enc).view(-1, pos_enc.size(1), self.num_heads, self.head_dim).transpose(1, 2)
            content_score = torch.matmul(q + self.pos_bias_u[None, :, None, :], k.transpose(-2, -1))
            pos_score = self._rel_shift(torch.matmul(q + self.pos_bias_v[None, :, None, :], p.transpose(-2, -1)))
            scores = (content_score + pos_score) * self.scale
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
            attn = self.attn_dropout(torch.softmax(scores, dim=-1))
            out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, self.model_dim)
            return self.w_out(out)

    class DenseFFN(nn.Module):
        def __init__(self, model_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(model_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, model_dim), nn.Dropout(dropout),
            )

        def forward(self, x, mask, forced_expert=None, return_all_experts=False):
            del mask, forced_expert, return_all_experts
            return self.net(x), None, None

    # -----------------------------------------------------------------------
    # Original single-GPU SharedAdapterMoEFFN (kept for DDP / single-GPU mode)
    # -----------------------------------------------------------------------
    class SharedAdapterMoEFFN(nn.Module):
        """All experts on one GPU. Used for single-GPU and DDP (data-parallel) modes.

        Improvements over the original:
        - Frame-level routing: each frame picks its own expert mix (not sentence-level).
        - Top-K sparse routing: only top_k experts are activated per frame, saving FLOPs.
        - Auxiliary load-balancing loss via z_loss on router logits (Switch Transformer style).
        """

        def __init__(self, model_dim, hidden_dim, adapter_hidden_dim, num_experts, temperature, dropout, top_k=2):
            super().__init__()
            self.temperature = float(temperature)
            self.num_experts = int(num_experts)
            self.top_k = min(int(top_k), self.num_experts)
            self.router = nn.Linear(model_dim, num_experts)
            self.trunks = nn.ModuleList([
                nn.Sequential(nn.Linear(model_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
                for _ in range(num_experts)
            ])
            self.share_down = nn.ModuleList([nn.Linear(hidden_dim, model_dim) for _ in range(num_experts)])
            self.adapter_up = nn.ModuleList([nn.Linear(hidden_dim, adapter_hidden_dim) for _ in range(num_experts)])
            self.adapter_down = nn.ModuleList([nn.Linear(adapter_hidden_dim, model_dim) for _ in range(num_experts)])
            self.dropout = nn.Dropout(dropout)

        def _pooled_hidden(self, h, mask):
            return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)

        def _expert_forward(self, h, idx):
            eh = self.trunks[idx](h)
            return self.dropout(self.share_down[idx](eh) + self.adapter_down[idx](self.dropout(F.gelu(self.adapter_up[idx](eh)))))

        def forward(self, h, mask, forced_expert=None, return_all_experts=False):
            B, T, D = h.shape
            # Frame-level routing: (B, T, E) instead of (B, E)
            router_logits = self.router(h) / self.temperature  # (B, T, E)

            if forced_expert is not None:
                out = self._expert_forward(h, forced_expert)
                # Return sentence-level gates for compatibility with competition scoring
                pooled = self._pooled_hidden(h, mask)
                gates_sent = torch.softmax(self.router(pooled) / self.temperature, dim=-1)
                fg = torch.zeros_like(gates_sent); fg[:, forced_expert] = 1.0
                return out, fg, {"pooled": pooled, "router_gates": gates_sent}

            # Top-K sparse gating per frame
            if self.top_k < self.num_experts:
                topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)  # (B,T,K)
                topk_gates = torch.softmax(topk_vals, dim=-1)  # (B,T,K)
                # Scatter back to full expert dim for aggregation
                full_gates = torch.zeros(B, T, self.num_experts, device=h.device, dtype=h.dtype)
                full_gates.scatter_(-1, topk_idx, topk_gates)
            else:
                full_gates = torch.softmax(router_logits, dim=-1)  # (B,T,E)

            # Only compute experts that have non-zero gates
            active_experts = set()
            if self.top_k < self.num_experts:
                active_experts = set(topk_idx.unique().tolist())
            else:
                active_experts = set(range(self.num_experts))

            merged = torch.zeros_like(h)
            stacked_list = []
            for i in range(self.num_experts):
                if i in active_experts:
                    expert_out = self._expert_forward(h, i)
                else:
                    expert_out = torch.zeros_like(h)
                if return_all_experts:
                    stacked_list.append(expert_out)
                merged = merged + full_gates[:, :, i].unsqueeze(-1) * expert_out

            # Sentence-level gates for competition/logging compatibility
            gate_mean = (full_gates * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)  # (B, E)

            aux = {"pooled": self._pooled_hidden(h, mask), "router_gates": gate_mean}
            if return_all_experts:
                aux["all_expert_outputs"] = torch.stack(stacked_list, dim=2)  # (B, T, E, D)
            return merged, gate_mean, aux

        # --- Expert state helpers (used by evolution) ---
        def get_expert_state(self, expert_idx: int) -> dict:
            return {
                "trunk":       {k: v.detach().clone() for k, v in self.trunks[expert_idx].state_dict().items()},
                "share_down":  {k: v.detach().clone() for k, v in self.share_down[expert_idx].state_dict().items()},
                "adapter_up":  {k: v.detach().clone() for k, v in self.adapter_up[expert_idx].state_dict().items()},
                "adapter_down":{k: v.detach().clone() for k, v in self.adapter_down[expert_idx].state_dict().items()},
            }

        def set_expert_state(self, expert_idx: int, state_dict_like: dict) -> None:
            self.trunks[expert_idx].load_state_dict(state_dict_like["trunk"])
            self.share_down[expert_idx].load_state_dict(state_dict_like["share_down"])
            self.adapter_up[expert_idx].load_state_dict(state_dict_like["adapter_up"])
            self.adapter_down[expert_idx].load_state_dict(state_dict_like["adapter_down"])

        @staticmethod
        def _flatten_state_dict_tensors(s):
            return [(gn, tn, t) for gn, g in s.items() for tn, t in g.items()]

        @staticmethod
        def _merge_tensor(ta, tb, *, alpha, split_index, tensor_offset):
            fa, fb = ta.reshape(-1), tb.to(ta.device, dtype=ta.dtype).reshape(-1)
            out = torch.empty_like(fa)
            ls = max(0, min(fa.numel(), split_index - tensor_offset))
            if ls > 0: out[:ls] = fa[:ls] * (1 - alpha) + fb[:ls] * alpha
            if ls < fa.numel(): out[ls:] = fa[ls:] * alpha + fb[ls:] * (1 - alpha)
            return out.view_as(ta)

        def merge_experts(self, parent_a, parent_b, child_idx=None, alpha=0.5, split_ratio=0.5, mode="split_linear"):
            sa, sb = self.get_expert_state(parent_a), self.get_expert_state(parent_b)
            flat_a = self._flatten_state_dict_tensors(sa)
            flat_b = self._flatten_state_dict_tensors(sb)
            total = sum(t.numel() for _, _, t in flat_a)
            si = int(round(max(0., min(1., split_ratio)) * total))
            ms: dict = {"trunk": {}, "share_down": {}, "adapter_up": {}, "adapter_down": {}}
            off = 0
            for (gn, tn, ta), (_, _, tb) in zip(flat_a, flat_b):
                ms[gn][tn] = self._merge_tensor(ta, tb, alpha=alpha, split_index=si, tensor_offset=off)
                off += ta.numel()
            if child_idx is not None:
                self.set_expert_state(child_idx, ms)
            return ms

    # -----------------------------------------------------------------------
    # Expert-Parallel SharedAdapterMoEFFN  ← KEY NEW CLASS
    # -----------------------------------------------------------------------
    class ExpertParallelSharedAdapterMoEFFN(nn.Module):
        """Expert Parallel MoE FFN.

        Each GPU stores exactly ``num_experts // world_size`` experts.
        The router is replicated on every GPU.

        Forward pass
        ~~~~~~~~~~~~
        1. Compute gates from the replicated router (same result on every GPU
           since the input is replicated via DistributedSampler).
        2. Compute local expert outputs (this GPU's experts only).
        3. ``_AllGatherGrad.apply`` collects all expert outputs across GPUs.
        4. Weighted sum with gates.

        The all_gather is differentiable: gradients flow back only to the
        local expert parameters, keeping the parameter-gradient mapping correct.

        ``forced_expert`` mode (used by ``compute_expert_scores``)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The GPU that owns the requested expert computes it; the result is
        broadcast to all GPUs via ``dist.all_reduce(SUM)``, with non-owning
        GPUs contributing zeros.
        """

        def __init__(
            self,
            model_dim: int,
            hidden_dim: int,
            adapter_hidden_dim: int,
            num_experts: int,
            temperature: float,
            dropout: float,
            rank: int,
            world_size: int,
            top_k: int = 2,
        ):
            super().__init__()
            if num_experts % world_size != 0:
                raise ValueError(
                    f"num_experts ({num_experts}) must be divisible by world_size ({world_size}) "
                    "for Expert Parallelism."
                )
            self.temperature = float(temperature)
            self.num_experts = int(num_experts)
            self.top_k = min(int(top_k), self.num_experts)
            self.world_size = world_size
            self.rank = rank
            self.local_num_experts = num_experts // world_size
            self.expert_offset = rank * self.local_num_experts  # first global expert index on this GPU

            # Router is identical on every GPU (replicated); its gradients are
            # all_reduced with the other non-expert params.
            self.router = nn.Linear(model_dim, num_experts)

            # Only local experts are instantiated on this GPU.
            self.trunks = nn.ModuleList([
                nn.Sequential(nn.Linear(model_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
                for _ in range(self.local_num_experts)
            ])
            self.share_down  = nn.ModuleList([nn.Linear(hidden_dim, model_dim) for _ in range(self.local_num_experts)])
            self.adapter_up  = nn.ModuleList([nn.Linear(hidden_dim, adapter_hidden_dim) for _ in range(self.local_num_experts)])
            self.adapter_down= nn.ModuleList([nn.Linear(adapter_hidden_dim, model_dim) for _ in range(self.local_num_experts)])
            self.dropout = nn.Dropout(dropout)

        # --- internal helpers ---

        def _pooled(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)

        def _local_expert_fwd(self, h: torch.Tensor, local_idx: int) -> torch.Tensor:
            eh = self.trunks[local_idx](h)
            return self.dropout(
                self.share_down[local_idx](eh)
                + self.adapter_down[local_idx](self.dropout(F.gelu(self.adapter_up[local_idx](eh))))
            )

        # --- forward ---

        def forward(
            self,
            hidden_states: torch.Tensor,
            mask: torch.Tensor,
            forced_expert: int | None = None,
            return_all_experts: bool = False,
        ):
            B, T, D = hidden_states.shape

            # ── forced_expert path (used by competition scoring) ──
            if forced_expert is not None:
                pooled = self._pooled(hidden_states, mask)
                gates_sent = torch.softmax(self.router(pooled) / self.temperature, dim=-1)
                local_idx = forced_expert - self.expert_offset
                if 0 <= local_idx < self.local_num_experts:
                    output = self._local_expert_fwd(hidden_states, local_idx)
                else:
                    output = torch.zeros_like(hidden_states)
                if self.world_size > 1:
                    target_B, target_T = _sync_batch_time_dims(B, T, hidden_states.device)
                    output = _pad_batch_time_dims(output, target_B, target_T)
                    dist.all_reduce(output, op=dist.ReduceOp.SUM)
                    output = output[:B, :T, :]  # trim back
                fg = torch.zeros_like(gates_sent); fg[:, forced_expert] = 1.0
                return output, fg, {"pooled": pooled, "router_gates": gates_sent}

            # ── normal path with frame-level Top-K routing ──
            router_logits = self.router(hidden_states) / self.temperature  # (B, T, E)

            if self.top_k < self.num_experts:
                topk_vals, topk_idx = torch.topk(router_logits, self.top_k, dim=-1)
                topk_gates = torch.softmax(topk_vals, dim=-1)
                full_gates = torch.zeros(B, T, self.num_experts, device=hidden_states.device, dtype=hidden_states.dtype)
                full_gates.scatter_(-1, topk_idx, topk_gates)
            else:
                full_gates = torch.softmax(router_logits, dim=-1)

            # Step 1: compute this GPU's local expert outputs → (local_E, B, T, D)
            local_outputs = torch.stack(
                [self._local_expert_fwd(hidden_states, i) for i in range(self.local_num_experts)],
                dim=0,
            )

            # Step 2: differentiable all_gather → (num_experts, B, T, D)
            # Each rank may have different batch/time shapes. Synchronize the
            # max shape across ranks, pad before gather, and trim after.
            if self.world_size > 1:
                target_B, target_T = _sync_batch_time_dims(B, T, hidden_states.device)
                local_outputs = _pad_batch_time_dims(local_outputs, target_B, target_T)
                all_outputs = _AllGatherGrad.apply(local_outputs, self.world_size, self.rank)
                all_outputs = all_outputs[:, :B, :T, :]  # trim back to local batch/time
            else:
                all_outputs = local_outputs

            # Step 3: frame-level weighted sum  full_gates:(B,T,E)  all_outputs:(E,B,T,D)
            # Rearrange all_outputs to (B,T,E,D) for element-wise multiply
            all_out_btde = all_outputs.permute(1, 2, 0, 3)  # (B,T,E,D)
            merged = (all_out_btde * full_gates.unsqueeze(-1)).sum(dim=2)  # (B,T,D)

            # Sentence-level gates for competition/logging compatibility
            gate_mean = (full_gates * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)

            aux = {"pooled": self._pooled(hidden_states, mask), "router_gates": gate_mean}
            if return_all_experts:
                aux["all_expert_outputs"] = all_out_btde
            return merged, gate_mean, aux

        # --- expert state helpers (global index aware) ---

        def global_to_local(self, global_idx: int) -> int:
            local = global_idx - self.expert_offset
            if not (0 <= local < self.local_num_experts):
                raise ValueError(
                    f"Expert {global_idx} is not on rank {self.rank} "
                    f"(local experts: {self.expert_offset}–{self.expert_offset + self.local_num_experts - 1})"
                )
            return local

        def owns_expert(self, global_idx: int) -> bool:
            return self.expert_offset <= global_idx < self.expert_offset + self.local_num_experts

        def get_expert_state(self, global_idx: int) -> dict:
            li = self.global_to_local(global_idx)
            return {
                "trunk":       {k: v.detach().clone() for k, v in self.trunks[li].state_dict().items()},
                "share_down":  {k: v.detach().clone() for k, v in self.share_down[li].state_dict().items()},
                "adapter_up":  {k: v.detach().clone() for k, v in self.adapter_up[li].state_dict().items()},
                "adapter_down":{k: v.detach().clone() for k, v in self.adapter_down[li].state_dict().items()},
            }

        def set_expert_state(self, global_idx: int, state: dict) -> None:
            li = self.global_to_local(global_idx)
            self.trunks[li].load_state_dict(state["trunk"])
            self.share_down[li].load_state_dict(state["share_down"])
            self.adapter_up[li].load_state_dict(state["adapter_up"])
            self.adapter_down[li].load_state_dict(state["adapter_down"])

        def get_expert_state_distributed(self, global_idx: int, device: str) -> dict:
            """Return expert state on every rank, fetching from the owning rank."""
            owning_rank = global_idx // self.local_num_experts
            if self.owns_expert(global_idx):
                state = self.get_expert_state(global_idx)
            else:
                # Create a dummy state filled with zeros (correct shape, wrong values –
                # will be overwritten by the broadcast)
                li_ref = 0  # use local expert 0 as shape reference
                state = {
                    "trunk":        {k: torch.zeros_like(v) for k, v in self.trunks[li_ref].state_dict().items()},
                    "share_down":   {k: torch.zeros_like(v) for k, v in self.share_down[li_ref].state_dict().items()},
                    "adapter_up":   {k: torch.zeros_like(v) for k, v in self.adapter_up[li_ref].state_dict().items()},
                    "adapter_down": {k: torch.zeros_like(v) for k, v in self.adapter_down[li_ref].state_dict().items()},
                }
            return _broadcast_expert_state(state, owning_rank=owning_rank, device=device)

        @staticmethod
        def _flatten_state(s: dict):
            return [(gn, tn, t) for gn, g in s.items() for tn, t in g.items()]

        @staticmethod
        def _merge_tensor(ta, tb, *, alpha, split_index, tensor_offset):
            fa = ta.reshape(-1)
            fb = tb.to(ta.device, dtype=ta.dtype).reshape(-1)
            out = torch.empty_like(fa)
            ls = max(0, min(fa.numel(), split_index - tensor_offset))
            if ls > 0: out[:ls] = fa[:ls] * (1 - alpha) + fb[:ls] * alpha
            if ls < fa.numel(): out[ls:] = fa[ls:] * alpha + fb[ls:] * (1 - alpha)
            return out.view_as(ta)

        def merge_experts(
            self,
            parent_a_state: dict,
            parent_b_state: dict,
            child_global_idx: int,
            alpha: float = 0.5,
            split_ratio: float = 0.5,
        ) -> None:
            """Merge two (already-gathered) expert states into a local expert."""
            flat_a = self._flatten_state(parent_a_state)
            flat_b = self._flatten_state(parent_b_state)
            total = sum(t.numel() for _, _, t in flat_a)
            si = int(round(max(0., min(1., split_ratio)) * total))
            ms: dict = {"trunk": {}, "share_down": {}, "adapter_up": {}, "adapter_down": {}}
            off = 0
            for (gn, tn, ta), (_, _, tb) in zip(flat_a, flat_b):
                ms[gn][tn] = self._merge_tensor(ta, tb, alpha=alpha, split_index=si, tensor_offset=off)
                off += ta.numel()
            self.set_expert_state(child_global_idx, ms)

    # -----------------------------------------------------------------------
    # Encoder blocks
    # -----------------------------------------------------------------------
    class TransformerMoEBlock(nn.Module):
        def __init__(self, args: argparse.Namespace, rank: int = 0, world_size: int = 1):
            super().__init__()
            self.layer_drop = float(getattr(args, "layer_drop", 0.0))
            self.self_attn_norm = nn.LayerNorm(args.encoder_dim)
            self.self_attn = RelativeMultiHeadAttention(args.encoder_dim, args.num_heads, args.dropout)
            self.dropout = nn.Dropout(args.dropout)
            self.ffn_norm = nn.LayerNorm(args.encoder_dim)
            top_k = int(getattr(args, "top_k", 2))
            use_ep = world_size > 1 and getattr(args, "expert_parallel", True)
            if args.ffn_type == "dense":
                self.ffn = DenseFFN(args.encoder_dim, args.ffn_hidden_dim, args.dropout)
            elif use_ep:
                self.ffn = ExpertParallelSharedAdapterMoEFFN(
                    model_dim=args.encoder_dim, hidden_dim=args.ffn_hidden_dim,
                    adapter_hidden_dim=args.adapter_hidden_dim, num_experts=args.num_experts,
                    temperature=args.router_temperature, dropout=args.dropout,
                    rank=rank, world_size=world_size, top_k=top_k,
                )
            else:
                self.ffn = SharedAdapterMoEFFN(
                    model_dim=args.encoder_dim, hidden_dim=args.ffn_hidden_dim,
                    adapter_hidden_dim=args.adapter_hidden_dim, num_experts=args.num_experts,
                    temperature=args.router_temperature, dropout=args.dropout, top_k=top_k,
                )

        def forward(self, h, mask, pos_enc, forced_expert=None, return_all_experts=False):
            if self.training and self.layer_drop > 0 and random.random() < self.layer_drop:
                return h, None, None
            kpm = ~mask.bool()
            attn_in = self.self_attn_norm(h)
            h = h + self.dropout(self.self_attn(attn_in, attn_in, attn_in, pos_enc=pos_enc, key_padding_mask=kpm))
            ffn_out, routing, aux = self.ffn(self.ffn_norm(h), mask.float(),
                                              forced_expert=forced_expert, return_all_experts=return_all_experts)
            return h + self.dropout(ffn_out), routing, aux

    class ConformerConvModule(nn.Module):
        def __init__(self, model_dim: int, kernel_size: int, dropout: float):
            super().__init__()
            self.layer_norm = nn.LayerNorm(model_dim)
            self.pointwise_in = nn.Conv1d(model_dim, 2 * model_dim, kernel_size=1)
            self.depthwise = nn.Conv1d(model_dim, model_dim, kernel_size=kernel_size,
                                       padding=kernel_size // 2, groups=model_dim)
            self.batch_norm = nn.BatchNorm1d(model_dim)
            self.pointwise_out = nn.Conv1d(model_dim, model_dim, kernel_size=1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            x = F.glu(self.pointwise_in(self.layer_norm(h).transpose(1, 2)), dim=1)
            x = F.silu(self.batch_norm(self.depthwise(x)))
            return self.dropout(self.pointwise_out(x).transpose(1, 2))

    class ConformerMoEBlock(nn.Module):
        def __init__(self, args: argparse.Namespace, rank: int = 0, world_size: int = 1):
            super().__init__()
            self.layer_drop = float(getattr(args, "layer_drop", 0.0))
            self.macaron_norm = nn.LayerNorm(args.encoder_dim)
            self.macaron_ffn = DenseFFN(args.encoder_dim, args.ffn_hidden_dim, args.dropout)
            self.self_attn_norm = nn.LayerNorm(args.encoder_dim)
            self.self_attn = RelativeMultiHeadAttention(args.encoder_dim, args.num_heads, args.dropout)
            self.conv_module = ConformerConvModule(args.encoder_dim, args.conv_kernel_size, args.dropout)
            self.ffn_norm = nn.LayerNorm(args.encoder_dim)
            top_k = int(getattr(args, "top_k", 2))
            use_ep = world_size > 1 and getattr(args, "expert_parallel", True)
            if args.ffn_type == "dense":
                self.ffn = DenseFFN(args.encoder_dim, args.ffn_hidden_dim, args.dropout)
            elif use_ep:
                self.ffn = ExpertParallelSharedAdapterMoEFFN(
                    model_dim=args.encoder_dim, hidden_dim=args.ffn_hidden_dim,
                    adapter_hidden_dim=args.adapter_hidden_dim, num_experts=args.num_experts,
                    temperature=args.router_temperature, dropout=args.dropout,
                    rank=rank, world_size=world_size, top_k=top_k,
                )
            else:
                self.ffn = SharedAdapterMoEFFN(
                    model_dim=args.encoder_dim, hidden_dim=args.ffn_hidden_dim,
                    adapter_hidden_dim=args.adapter_hidden_dim, num_experts=args.num_experts,
                    temperature=args.router_temperature, dropout=args.dropout, top_k=top_k,
                )
            self.final_norm = nn.LayerNorm(args.encoder_dim)
            self.dropout = nn.Dropout(args.dropout)

        def forward(self, h, mask, pos_enc, forced_expert=None, return_all_experts=False):
            if self.training and self.layer_drop > 0 and random.random() < self.layer_drop:
                return h, None, None
            mac_out, _, _ = self.macaron_ffn(self.macaron_norm(h), mask.float())
            h = h + 0.5 * self.dropout(mac_out)
            kpm = ~mask.bool()
            attn_in = self.self_attn_norm(h)
            h = h + self.dropout(self.self_attn(attn_in, attn_in, attn_in, pos_enc=pos_enc, key_padding_mask=kpm))
            h = h + self.conv_module(h)
            ffn_out, routing, aux = self.ffn(self.ffn_norm(h), mask.float(),
                                              forced_expert=forced_expert, return_all_experts=return_all_experts)
            return self.final_norm(h + 0.5 * self.dropout(ffn_out)), routing, aux

    class EncoderMoECTCModel(nn.Module):
        def __init__(self, args: argparse.Namespace, vocab_size: int, rank: int = 0, world_size: int = 1):
            super().__init__()
            self.num_experts = int(args.num_experts)
            self.ffn_type = args.ffn_type
            self.subsampling = Conv2dSubsampling(args.n_mels, args.encoder_dim, args.dropout)
            self.position = RelativePositionalEncoding(args.encoder_dim, dropout=args.dropout)
            block_cls = TransformerMoEBlock if args.encoder_type == "transformer" else ConformerMoEBlock
            self.blocks = nn.ModuleList([block_cls(args, rank=rank, world_size=world_size) for _ in range(args.encoder_layers)])
            self.output_norm = nn.LayerNorm(args.encoder_dim)
            self.projector = nn.Sequential(
                nn.Linear(args.encoder_dim, args.projector_dim), nn.GELU(), nn.Dropout(args.dropout)
            )
            self.ctc_head = nn.Linear(args.projector_dim, vocab_size)
            inter_weight = float(getattr(args, "intermediate_ctc_weight", 0.0))
            inter_layer = int(getattr(args, "intermediate_ctc_layer", 0)) or max(1, args.encoder_layers // 2)
            self._inter_ctc_layer = inter_layer if inter_weight > 0 else -1
            if self._inter_ctc_layer >= 0:
                self.inter_norm = nn.LayerNorm(args.encoder_dim)
                self.inter_proj = nn.Sequential(
                    nn.Linear(args.encoder_dim, args.projector_dim), nn.GELU(), nn.Dropout(args.dropout)
                )
                self.inter_ctc_head = nn.Linear(args.projector_dim, vocab_size)

        def forward(self, inputs, input_lengths, forced_expert=None, forced_experts=None, return_aux=False):
            h, out_len = self.subsampling(inputs, input_lengths.to(inputs.device))
            h, pos_enc = self.position(h)
            mask = lengths_to_mask(out_len.to(h.device), h.size(1))
            routing_vals, block_aux = [], []
            inter_log_probs = None
            for bi, block in enumerate(self.blocks):
                fe = forced_expert if forced_experts is None else forced_experts.get(bi, forced_expert)
                h, routing, aux = block(h, mask, pos_enc, forced_expert=fe, return_all_experts=return_aux)
                if routing is not None:
                    routing_vals.append(routing)
                if return_aux:
                    block_aux.append({"block_index": bi, "routing": routing, "aux": aux})
                if self._inter_ctc_layer >= 0 and bi == self._inter_ctc_layer and self.training:
                    inter_log_probs = F.log_softmax(self.inter_ctc_head(self.inter_proj(self.inter_norm(h))), dim=-1)
            h = self.output_norm(h)
            log_probs = F.log_softmax(self.ctc_head(self.projector(h)), dim=-1)
            merged_routing = torch.stack(routing_vals).mean(0) if routing_vals else None
            aux_out = {"block_aux": block_aux, "mask": mask, "output_lengths": out_len} if return_aux else None
            return log_probs, out_len, merged_routing, aux_out, inter_log_probs

        def get_moe_modules(self):
            return [block.ffn for block in self.blocks
                    if isinstance(block.ffn, (SharedAdapterMoEFFN, ExpertParallelSharedAdapterMoEFFN))]

else:
    SinusoidalPositionalEncoding = None
    Conv2dSubsampling = None
    RelativePositionalEncoding = None
    RelativeMultiHeadAttention = None
    DenseFFN = None
    SharedAdapterMoEFFN = None
    ExpertParallelSharedAdapterMoEFFN = None
    ConformerConvModule = None
    TransformerMoEBlock = None
    ConformerMoEBlock = None
    EncoderMoECTCModel = None


# ===========================================================================
# Loss / routing utilities  (unchanged from original)
# ===========================================================================

def routing_regularizer(avg_gates, num_experts):
    if avg_gates is None:
        return torch.tensor(0.0)
    expected = torch.full((num_experts,), 1.0 / num_experts, device=avg_gates.device, dtype=avg_gates.dtype)
    return F.mse_loss(avg_gates.mean(0), expected)


def routing_entropy(gates, eps=1e-8):
    if gates is None:
        return torch.tensor(0.0)
    sg = gates.clamp_min(eps)
    return -(sg * sg.log()).sum(-1).mean()


def split_ctc_targets(targets, target_lengths):
    segs, off = [], 0
    for ln in target_lengths.tolist():
        segs.append(targets[off:off + int(ln)]); off += int(ln)
    return segs


def convert_loss_to_score(losses, args):
    if args.competition_score == "inverse_loss":
        return 1.0 / losses.clamp_min(args.competition_epsilon)
    return torch.exp(-losses)


def compute_per_sample_ctc_losses(log_probs, targets, output_lengths, target_lengths, ctc_loss):
    raw = F.ctc_loss(log_probs.transpose(0, 1), targets, output_lengths.cpu(), target_lengths.cpu(),
                     blank=int(getattr(ctc_loss, "blank", 0)),
                     reduction="none", zero_infinity=bool(getattr(ctc_loss, "zero_infinity", True)))
    return raw / target_lengths.to(raw.device, dtype=raw.dtype).clamp_min(1.0)


@torch.no_grad()
def compute_expert_scores(model, batch, ctc_loss, args, device, *, use_amp=False, block_idx=None):
    """Works for both SharedAdapterMoEFFN and ExpertParallelSharedAdapterMoEFFN.

    In EP mode, each forced_expert call does an all_reduce internally so every
    GPU gets the correct output regardless of which GPU owns the expert.
    """
    if args.ffn_type != "shared_adapter_moe":
        return None
    scores = []
    ac_dev = "cuda" if device.startswith("cuda") else "cpu"
    ac_dtype = torch.float16 if ac_dev == "cuda" else torch.bfloat16
    for ei in range(args.num_experts):
        fwd_kw: dict = {"return_aux": False}
        if block_idx is None:
            fwd_kw["forced_expert"] = ei
        else:
            fwd_kw["forced_experts"] = {block_idx: ei}
        with torch.autocast(device_type=ac_dev, dtype=ac_dtype, enabled=use_amp):
            lp, ol, _, _, _ = model(batch["inputs"], batch["input_lengths"], **fwd_kw)
        loss = compute_per_sample_ctc_losses(lp, batch["targets"], ol, batch["target_lengths"], ctc_loss).detach()
        scores.append(convert_loss_to_score(loss, args))
    return torch.stack(scores, dim=1)


def competition_targets(expert_scores, args):
    z = expert_scores.sum(-1, keepdim=True).clamp_min(args.competition_epsilon)
    q = expert_scores / z
    return q, q.sum(0), z.squeeze(-1)


def routing_alignment_loss(gates, targets, eps=1e-8):
    if gates is None or targets is None:
        dev = (gates or targets).device if (gates is not None or targets is not None) else None
        return torch.tensor(0.0, device=dev)
    return F.kl_div(gates.clamp_min(eps).log(), targets.clamp_min(eps), reduction="batchmean")


def select_expert_parents(scores, eps=1e-6):
    E = scores.size(1)
    if E == 1: return 0, 0, {"fitness": [1.0], "attraction": [0.0]}
    z = scores.sum(-1, keepdim=True).clamp_min(eps)
    q = scores / z
    fitness = q.sum(0)
    pa = int(torch.argmax(fitness).item())
    attr = torch.full((E,), float("-inf"), device=scores.device, dtype=scores.dtype)
    denom = z.squeeze(-1) + eps
    for ci in range(E):
        if ci == pa: continue
        attr[ci] = (torch.relu(scores[:, ci] - scores[:, pa]) / denom).sum()
    pb = int(torch.argmax(attr).item()) if torch.isfinite(attr).any() else int(torch.argsort(fitness, descending=True)[1])
    return pa, pb, {
        "fitness": [round(float(v), 6) for v in fitness.detach().cpu().tolist()],
        "attraction": [round(float(v), 6) if math.isfinite(float(v)) else None for v in attr.detach().cpu().tolist()],
    }


def collect_moe_modules(model, block_limit):
    """Return list of (block_idx, moe_module) for evolution."""
    mods = []
    for bi, block in enumerate(model.blocks):
        ffn = getattr(block, "ffn", None)
        if isinstance(ffn, (SharedAdapterMoEFFN, ExpertParallelSharedAdapterMoEFFN)):
            mods.append((bi, ffn))
    if block_limit > 0:
        mods = mods[:block_limit]
    return mods


def select_replacement_expert(fitness, usage, *, parent_a, parent_b, strategy):
    E = fitness.numel()
    eligible = [i for i in range(E) if i not in {parent_a, parent_b}] or list(range(E))
    if strategy == "random":
        return random.choice(eligible), {"strategy": strategy}
    if strategy == "redundant":
        fn = fitness / fitness.sum().clamp_min(1e-8); un = usage / usage.sum().clamp_min(1e-8)
        sc = torch.tensor([float((fn + un)[i]) for i in eligible])
        return eligible[int(torch.argmin(sc))], {"strategy": strategy}
    sc = torch.tensor([float(fitness[i]) for i in eligible])
    return eligible[int(torch.argmin(sc))], {"strategy": "worst"}


# ---------------------------------------------------------------------------
# Scheduling / weight helpers
# ---------------------------------------------------------------------------

def get_effective_competition_weight(args, epoch):
    if args.ffn_type != "shared_adapter_moe" or args.competition_weight <= 0: return 0.0
    if epoch is None: return float(args.competition_weight)
    if epoch <= args.competition_warmup_epochs: return 0.0
    ramp = max(0, int(args.competition_ramp_epochs))
    if ramp == 0: return float(args.competition_weight)
    return float(args.competition_weight) * min(1.0, max(0.0, (epoch - args.competition_warmup_epochs) / ramp))


def should_compute_train_competition(args, epoch, step):
    if get_effective_competition_weight(args, epoch) <= 0: return False
    return step % max(1, int(args.competition_interval_steps)) == 0


def should_compute_competition_metrics(args, stage, batch_idx, epoch=None):
    if get_effective_competition_weight(args, epoch) <= 0: return False
    if stage.startswith("valid") and not args.competition_on_valid: return False
    return args.competition_batches <= 0 or batch_idx <= args.competition_batches


def get_annealed_temperature(args, epoch):
    n = int(getattr(args, "temperature_anneal_epochs", 0))
    if n <= 0: return float(args.router_temperature)
    t0 = float(getattr(args, "temperature_anneal_start", None) or args.router_temperature)
    t1 = float(getattr(args, "temperature_anneal_end", None) or args.router_temperature)
    return t0 + (t1 - t0) * min(1., max(0., epoch / n))


def should_run_expert_evolution(args, epoch):
    if args.ffn_type != "shared_adapter_moe" or args.expert_evolve_every_epochs <= 0: return False
    if epoch < max(1, int(args.expert_evolve_start_epoch)): return False
    if get_effective_competition_weight(args, epoch) <= 0: return False
    return epoch % args.expert_evolve_every_epochs == 0


def build_lr_scheduler(optimizer, args, steps_per_epoch):
    if args.scheduler == "none": return None
    total = max(1, args.epochs * max(1, steps_per_epoch))
    warmup = int(args.warmup_steps) or int(round(total * float(args.warmup_ratio)))
    warmup = min(max(0, warmup), max(0, total - 1))
    mls = float(args.min_lr_scale)

    def lr_lambda(step):
        s = step + 1
        if warmup > 0 and s <= warmup: return max(1e-8, s / warmup)
        prog = min(1., max(0., (s - warmup) / max(1, total - warmup)))
        return mls + (1 - mls) * 0.5 * (1 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def flatten_scalar_metrics(prefix, metrics):
    payload = {}
    for k, v in metrics.items():
        safe = "".join(c if c.isalnum() or c in "_.-" else "_" for c in str(k)).strip("_") or "unknown"
        payload[f"{prefix}/{safe}"] = float(v)
    return payload


def append_vector_metrics(payload, prefix, values):
    for i, v in enumerate(values):
        payload[f"{prefix}_{i}"] = float(v)


# ===========================================================================
# Evolution (EP-aware)
# ===========================================================================

@torch.no_grad()
def collect_evolution_statistics(model, loader, ctc_loss, args, device, use_amp=False):
    if args.ffn_type != "shared_adapter_moe": return []
    selected = collect_moe_modules(model, args.expert_merge_blocks)
    if not selected: return []
    block_stats = {
        bi: {"module": ffn, "scores": [], "usage_sum": torch.zeros(args.num_experts), "usage_count": 0}
        for bi, ffn in selected
    }
    max_b = args.competition_batches if args.competition_batches > 0 else len(loader)
    ac_dev = "cuda" if device.startswith("cuda") else "cpu"
    ac_dtype = torch.float16 if ac_dev == "cuda" else torch.bfloat16
    model.eval()
    it = build_progress(loader, total=min(len(loader), max_b), desc="evolve stats", leave=False)
    for step, batch in enumerate(it, 1):
        if step > max_b: break
        batch = move_batch_to_device(batch, device, non_blocking=device.startswith("cuda"))
        with torch.autocast(device_type=ac_dev, dtype=ac_dtype, enabled=use_amp):
            _, _, _, aux, _ = model(batch["inputs"], batch["input_lengths"], return_aux=True)
        if aux is not None:
            for be in aux["block_aux"]:
                bi = int(be["block_index"])
                rt = be["routing"]
                if rt is None or bi not in block_stats: continue
                block_stats[bi]["usage_sum"] += rt.detach().sum(0).cpu()
                block_stats[bi]["usage_count"] += rt.size(0)
        for bi in block_stats:
            sc = compute_expert_scores(model, batch, ctc_loss, args, device, use_amp=False, block_idx=bi)
            if sc is not None:
                block_stats[bi]["scores"].append(sc.detach().cpu())
    collected = []
    for bi, stats in block_stats.items():
        if not stats["scores"]: continue
        sc = torch.cat(stats["scores"], 0)
        _, fitness, _ = competition_targets(sc, args)
        cnt = max(1, int(stats["usage_count"]))
        collected.append({"block_idx": bi, "module": stats["module"], "scores": sc,
                           "fitness": fitness, "avg_usage": stats["usage_sum"] / cnt})
    return collected


@torch.no_grad()
def evolve_experts(model, loader, ctc_loss, args, device, use_amp=False) -> list:
    if args.ffn_type != "shared_adapter_moe" or args.expert_evolve_every_epochs <= 0: return []
    ev_stats = collect_evolution_statistics(model, loader, ctc_loss, args, device, use_amp)
    if not ev_stats: return []

    logs = []
    for stats in ev_stats:
        bi = int(stats["block_idx"])
        moe = stats["module"]
        is_ep = isinstance(moe, ExpertParallelSharedAdapterMoEFFN)

        pa, pb, pdiag = select_expert_parents(stats["scores"], eps=args.competition_epsilon)
        ri, rdiag = select_replacement_expert(
            stats["fitness"], stats["avg_usage"],
            parent_a=pa, parent_b=pb, strategy=args.expert_merge_replace
        )

        if is_ep:
            # Gather parent states from their owning GPUs to all ranks
            sa = moe.get_expert_state_distributed(pa, device)
            sb = moe.get_expert_state_distributed(pb, device)
            # Only the GPU owning the child expert applies the merge
            if moe.owns_expert(ri):
                moe.merge_experts(sa, sb, ri, alpha=args.expert_merge_alpha,
                                  split_ratio=args.expert_merge_split_ratio)
        else:
            moe.merge_experts(pa, pb, child_idx=ri,
                              alpha=args.expert_merge_alpha,
                              split_ratio=args.expert_merge_split_ratio)

        logs.append({
            "block": bi,
            "parent_a": pa, "parent_b": pb, "replace_idx": ri,
            "fitness": [round(float(v), 6) for v in stats["fitness"].tolist()],
            "avg_usage": [round(float(v), 6) for v in stats["avg_usage"].tolist()],
            "parent_selection": pdiag, "replacement": rdiag,
        })
    return logs


# ===========================================================================
# Training loop
# ===========================================================================

def train_one_epoch(
    model, loader, tokenizer, optimizer, ctc_loss, args, device,
    epoch, wandb_run=None, scaler=None, use_amp=False, scheduler=None,
    world_size=1, rank=0,
) -> dict:
    del tokenizer
    model.train()
    accum = max(1, int(getattr(args, "grad_accum_steps", 1)))
    rl = rb = rlb = rcomp = rinter = rent = rent_b = rgn = rtw = 0.0
    rgate: torch.Tensor | None = None; rgate_cnt = 0
    ac_dev = "cuda" if device.startswith("cuda") else "cpu"
    ac_dtype = torch.float16 if ac_dev == "cuda" else torch.bfloat16
    it = build_progress(loader, total=len(loader), desc=f"train e{epoch}", leave=False)
    ec_weight = get_effective_competition_weight(args, epoch)
    prof = bool(getattr(args, "profile_performance", False))
    timing = {"data": 0., "transfer": 0., "forward": 0., "competition": 0., "backward": 0., "optimizer": 0.}
    t_end = time.perf_counter()
    it_obj = iter(it)

    for step in range(1, len(loader) + 1):
        try:
            batch = next(it_obj)
        except StopIteration:
            break
        timing["data"] += time.perf_counter() - t_end

        t0 = time.perf_counter()
        batch = move_batch_to_device(batch, device, non_blocking=device.startswith("cuda"))
        synchronize_for_timing(device, prof)
        timing["transfer"] += time.perf_counter() - t0

        if (step - 1) % accum == 0:
            optimizer.zero_grad(set_to_none=True)

        if getattr(args, "spec_augment", False):
            batch["inputs"] = spec_augment(
                batch["inputs"],
                freq_mask_param=int(getattr(args, "freq_mask_param", 27)),
                time_mask_param=int(getattr(args, "time_mask_param", 100)),
                num_freq_masks=int(getattr(args, "num_freq_masks", 2)),
                num_time_masks=int(getattr(args, "num_time_masks", 2)),
            )

        t0 = time.perf_counter()
        with torch.autocast(device_type=ac_dev, dtype=ac_dtype, enabled=use_amp):
            lp, ol, routing, _, inter_lp = model(batch["inputs"], batch["input_lengths"], return_aux=False)
            raw_ctc = ctc_loss(lp.transpose(0, 1), batch["targets"], ol.cpu(), batch["target_lengths"])
            ls = float(getattr(args, "label_smoothing", 0.0))
            base_loss = (1 - ls) * raw_ctc + ls * (-lp.mean()) if ls > 0 else raw_ctc
            inter_ctc = torch.tensor(0., device=lp.device)
            iw = float(getattr(args, "intermediate_ctc_weight", 0.0))
            if iw > 0 and inter_lp is not None:
                ri_raw = ctc_loss(inter_lp.transpose(0, 1), batch["targets"], ol.cpu(), batch["target_lengths"])
                inter_ctc = (1 - ls) * ri_raw + ls * (-inter_lp.mean()) if ls > 0 else ri_raw
            lb_loss = routing_regularizer(routing, args.num_experts) if routing is not None else torch.tensor(0., device=lp.device)
        synchronize_for_timing(device, prof)
        timing["forward"] += time.perf_counter() - t0

        # Competition loss
        comp_loss = torch.tensor(0., device=lp.device)
        t0 = time.perf_counter()
        if routing is not None and should_compute_train_competition(args, epoch, step):
            was_train = model.training; model.eval()
            es = compute_expert_scores(model, batch, ctc_loss, args, device, use_amp=False)
            if was_train: model.train()
            if es is not None:
                ct, _, _ = competition_targets(es, args)
                comp_loss = routing_alignment_loss(routing, ct.detach(), eps=args.competition_epsilon)
        synchronize_for_timing(device, prof)
        timing["competition"] += time.perf_counter() - t0

        eb = float(getattr(args, "entropy_bonus_weight", 0.0))
        rent_val = routing_entropy(routing, eps=args.competition_epsilon) if routing is not None else torch.tensor(0., device=lp.device)
        loss = base_loss + args.load_balance_weight * lb_loss + ec_weight * comp_loss - eb * rent_val + iw * inter_ctc
        scaled = loss / accum

        t0 = time.perf_counter()
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        # ── Expert Parallel: manually reduce non-expert gradients ──
        if world_size > 1 and getattr(args, "expert_parallel", True):
            _reduce_non_expert_gradients(model, world_size)

        synchronize_for_timing(device, prof)
        timing["backward"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        gn = 0.0
        if step % accum == 0 or step == len(loader):
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                gn = float(nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
                scaler.step(optimizer); scaler.update()
            else:
                gn = float(nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
                optimizer.step()
            if world_size > 1 and getattr(args, "expert_parallel", True):
                _sync_batchnorm_buffers(model, world_size)
            if scheduler is not None: scheduler.step()
        synchronize_for_timing(device, prof)
        timing["optimizer"] += time.perf_counter() - t0

        bs = int(batch["input_lengths"].size(0))
        rtw += bs; rl += float(loss.item()) * bs; rb += float(base_loss.item()) * bs
        rlb += float(lb_loss.item()) * bs; rcomp += float(comp_loss.item()) * bs
        rinter += float(inter_ctc.item()) * bs; rgn += gn * bs
        if routing is not None:
            rent += float(routing_entropy(routing, eps=args.competition_epsilon).item()) * routing.size(0)
            if eb > 0: rent_b += float(rent_val.detach().item()) * bs
            gs = routing.detach().sum(0).cpu()
            rgate = gs if rgate is None else rgate + gs
            rgate_cnt += routing.size(0)

        if hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{rl / max(1, rtw):.4f}")

        if step % args.log_interval == 0 or step == len(loader):
            lr_ = float(optimizer.param_groups[0]["lr"])
            if rank == 0:
                print(f"epoch={epoch} step={step}/{len(loader)} "
                      f"train_loss={rl/max(1,rtw):.4f} ctc={rb/max(1,rtw):.4f} "
                      f"lb={rlb/max(1,rtw):.4f} comp={rcomp/max(1,rtw):.4f} "
                      f"grad={rgn/max(1,rtw):.4f} lr={lr_:.6g}", flush=True)
                log_wandb_metrics(wandb_run, {
                    "global_step": (epoch - 1) * len(loader) + step, "epoch": epoch,
                    "train/loss_step": rl / max(1, rtw), "train/lr": lr_,
                })
        t_end = time.perf_counter()

    # ── Reduce metrics across ranks ──
    if world_size > 1:
        rtw    = _all_reduce_scalar(rtw, device)
        rl     = _all_reduce_scalar(rl, device)
        rb     = _all_reduce_scalar(rb, device)
        rlb    = _all_reduce_scalar(rlb, device)
        rcomp  = _all_reduce_scalar(rcomp, device)
        rinter = _all_reduce_scalar(rinter, device)
        rent   = _all_reduce_scalar(rent, device)
        rgn    = _all_reduce_scalar(rgn, device)
        rgate_cnt = int(_all_reduce_scalar(rgate_cnt, device))
        if rgate is not None:
            rgate = _all_reduce_tensor(rgate, device)

    avg_gates = [round(float(v), 6) for v in (rgate / rgate_cnt).tolist()] if rgate is not None and rgate_cnt > 0 else []
    return {
        "loss": rl / max(1, rtw), "base_loss": rb / max(1, rtw),
        "ctc_loss": rb / max(1, rtw), "load_balance_loss": rlb / max(1, rtw),
        "competition_loss": rcomp / max(1, rtw), "inter_ctc_loss": rinter / max(1, rtw),
        "routing_entropy": rent / max(1, rgate_cnt), "grad_norm": rgn / max(1, rtw),
        "lr": float(optimizer.param_groups[0]["lr"]), "effective_competition_weight": ec_weight,
        "avg_gates": avg_gates, "expert_usage": avg_gates,
    }


# ===========================================================================
# Evaluation loop
# ===========================================================================

@torch.no_grad()
def evaluate(model, loader, tokenizer, ctc_loss, args, device, stage="eval",
             use_amp=False, epoch=None, world_size=1, rank=0) -> dict:
    was_train = model.training; model.eval()
    tl = tb = tlb = tc = tmer_cer = tmer_wer = tent = ttw = samp = tgc = 0
    tce = tcl = twe = twl = 0  # char/word edit/length for corpus metrics
    tgate: torch.Tensor | None = None
    r_by_dom: dict = defaultdict(list); dls: dict = defaultdict(float); dlc: dict = defaultdict(int)
    fit_store = []
    ec_weight = get_effective_competition_weight(args, epoch)
    ac_dev = "cuda" if device.startswith("cuda") else "cpu"
    ac_dtype = torch.float16 if ac_dev == "cuda" else torch.bfloat16
    it = build_progress(loader, total=len(loader), desc=stage, leave=False)

    for step, batch in enumerate(it, 1):
        batch = move_batch_to_device(batch, device, non_blocking=device.startswith("cuda"))
        with torch.autocast(device_type=ac_dev, dtype=ac_dtype, enabled=use_amp):
            lp, ol, routing, _, _ = model(batch["inputs"], batch["input_lengths"], return_aux=False)
            psl = compute_per_sample_ctc_losses(lp, batch["targets"], ol, batch["target_lengths"], ctc_loss)
            base_loss = psl.mean()
            lb_loss = routing_regularizer(routing, args.num_experts) if routing is not None else torch.tensor(0., device=lp.device)

        comp_loss = torch.tensor(0., device=lp.device)
        if routing is not None and should_compute_competition_metrics(args, stage, step, epoch=epoch):
            es = compute_expert_scores(model, batch, ctc_loss, args, device, use_amp=False)
            if es is not None:
                ct, fitness, _ = competition_targets(es, args)
                comp_loss = routing_alignment_loss(routing, ct, eps=args.competition_epsilon)
                fit_store.append(fitness.detach().cpu())

        hyps = select_hypotheses(lp, ol, tokenizer, args)
        bsz = len(hyps)
        penalty = args.load_balance_weight * lb_loss.detach() + ec_weight * comp_loss.detach()
        stl = psl.detach() + penalty
        ttw += bsz; tl += float(stl.sum().item()); tb += float(psl.detach().sum().item())
        tlb += float(lb_loss.detach().item()) * bsz; tc += float(comp_loss.detach().item()) * bsz

        for idx, (ref, hyp) in enumerate(zip(batch["texts"], hyps)):
            nr = normalize_eval_text(ref, args); nh = normalize_eval_text(hyp, args)
            tmer_cer += compute_cer(nr, nh); tmer_wer += compute_wer(nr, nh)
            et = compute_text_error_totals(nr, nh)
            tce += et["char_edits"]; tcl += et["char_length"]
            twe += et["word_edits"]; twl += et["word_length"]
            dom = batch["domains"][idx]
            dls[dom] += float(stl[idx].item()); dlc[dom] += 1
            if routing is not None: r_by_dom[dom].append(routing[idx].detach().cpu())
            samp += 1

        if routing is not None:
            tent += float(routing_entropy(routing, eps=args.competition_epsilon).item()) * bsz
            gs = routing.detach().sum(0).cpu()
            tgate = gs if tgate is None else tgate + gs
            tgc += routing.size(0)

        if hasattr(it, "set_postfix"):
            it.set_postfix(loss=f"{tl/max(1,ttw):.4f}", cer=f"{tmer_cer/max(1,samp):.4f}")

    # ── Reduce across ranks ──
    if world_size > 1:
        ttw  = int(_all_reduce_scalar(ttw, device))
        tl   = _all_reduce_scalar(tl, device)
        tb   = _all_reduce_scalar(tb, device)
        tlb  = _all_reduce_scalar(tlb, device)
        tc   = _all_reduce_scalar(tc, device)
        tmer_cer = _all_reduce_scalar(tmer_cer, device)
        tmer_wer = _all_reduce_scalar(tmer_wer, device)
        tce  = int(_all_reduce_scalar(tce, device))
        tcl  = int(_all_reduce_scalar(tcl, device))
        twe  = int(_all_reduce_scalar(twe, device))
        twl  = int(_all_reduce_scalar(twl, device))
        tent = _all_reduce_scalar(tent, device)
        samp = int(_all_reduce_scalar(samp, device))
        tgc  = int(_all_reduce_scalar(tgc, device))
        if tgate is not None:
            tgate = _all_reduce_tensor(tgate, device)

    avg_gates = [round(float(v), 6) for v in (tgate / tgc).tolist()] if tgate is not None and tgc > 0 else []
    ccr = tce / max(1, tcl); cwr = twe / max(1, twl)
    metrics = {
        "loss": tl / max(1, ttw), "total_loss": tl / max(1, ttw),
        "base_loss": tb / max(1, ttw), "ctc_loss": tb / max(1, ttw),
        "load_balance_loss": tlb / max(1, ttw), "competition_loss": tc / max(1, ttw),
        "cer": ccr, "wer": cwr, "mean_cer": tmer_cer / max(1, samp),
        "mean_wer": tmer_wer / max(1, samp), "corpus_cer": ccr, "corpus_wer": cwr,
        "routing": summarize_routing(r_by_dom), "avg_gates": avg_gates, "expert_usage": avg_gates,
        "routing_entropy": tent / max(1, samp),
        "domain_loss": {d: round(dls[d] / max(1, dlc[d]), 6) for d in sorted(dls)},
        "effective_competition_weight": ec_weight,
        "expert_fitness": [round(float(v), 6) for v in torch.stack(fit_store).mean(0).tolist()] if fit_store else [],
    }
    if was_train: model.train()
    return metrics


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args = parse_args()
    ensure_torch()

    # ── Distributed / EP setup ─────────────────────────────────────────────
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    is_main     = rank == 0
    use_ep      = bool(getattr(args, "expert_parallel", True)) and world_size > 1

    if world_size > 1:
        # Select backend: nccl is fastest on GPU clusters; gloo works on Kaggle/sandboxed envs
        _backend_arg = getattr(args, "dist_backend", "auto")
        if _backend_arg == "auto":
            _backend = "nccl" if torch.cuda.is_available() else "gloo"
        else:
            _backend = _backend_arg

        # Apply Kaggle/sandbox-friendly NCCL env vars if not already set
        if _backend == "nccl":
            os.environ.setdefault("NCCL_P2P_DISABLE", "1")
            os.environ.setdefault("NCCL_IB_DISABLE", "1")
            os.environ.setdefault("NCCL_SOCKET_NTHREADS", "2")
            os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "2")

        import datetime
        dist.init_process_group(
            backend=_backend,
            timeout=datetime.timedelta(seconds=1800),
        )
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        if is_main:
            print(f"Distributed backend: {_backend}", flush=True)

        if use_ep:
            if args.ffn_type == "shared_adapter_moe" and args.num_experts % world_size != 0:
                raise ValueError(
                    f"--num-experts {args.num_experts} must be divisible by "
                    f"world_size {world_size} for Expert Parallelism. "
                    f"Use --no-expert-parallel or adjust --num-experts."
                )
            if is_main:
                print(f"Expert Parallel: {world_size} GPUs × "
                      f"{args.num_experts // world_size} local experts "
                      f"= {args.num_experts} total experts.", flush=True)
        else:
            if is_main:
                print(f"Data Parallel (DDP): {world_size} GPUs, all experts replicated.", flush=True)
    else:
        device = choose_device(args.device)

    # Different data-augmentation seed per rank; same model-init seed on all.
    set_seed(args.seed)          # model weights identical across ranks at init
    # (DistributedSampler will shuffle differently per epoch via set_epoch)

    # ── Output dir + config ─────────────────────────────────────────────────
    if is_main:
        output_dir = prepare_output_dir(
            Path(args.output_dir).resolve(),
            allow_existing=bool(getattr(args, "allow_existing_output_dir", False)),
        )
        save_json(output_dir / "config.json", vars(args))
    else:
        output_dir = Path(args.output_dir).resolve()

    if world_size > 1:
        dist.barrier()  # non-main ranks wait for output_dir to exist

    wandb_run = init_wandb_run(args, output_dir, vars(args)) if is_main else None

    try:
        train_records = load_jsonl(args.train_manifest)
        valid_records = load_jsonl(args.valid_manifest)
        test_records  = load_jsonl(args.test_manifest) if args.test_manifest else None

        tokenizer = resolve_training_tokenizer(
            train_records, args=args,
            train_manifest=args.train_manifest, output_dir=output_dir,
        )
        if is_main:
            tokenizer.save(output_dir / "vocab.json")

        use_amp, use_tf32 = configure_runtime(device, args)
        if is_main:
            print(f"device={device} amp={use_amp} tf32={use_tf32}", flush=True)
        scaler = create_grad_scaler(use_amp=use_amp, is_cuda=device.startswith("cuda"))

        # ── Datasets ──────────────────────────────────────────────────────
        def _make_ds(records, manifest_path):
            return build_dataset_for_mode(
                records, tokenizer=tokenizer, sample_rate=args.sample_rate,
                args=args, manifest_path=manifest_path, device=device,
            )

        train_ds = _make_ds(train_records, args.train_manifest)
        valid_ds = _make_ds(valid_records, args.valid_manifest)
        test_ds  = _make_ds(test_records,  args.test_manifest) if test_records else None

        collate_fn = build_collate_fn(args, tokenizer)
        is_cuda    = device.startswith("cuda")
        data_on_device  = dataset_storage_device(train_ds).startswith("cuda")
        mem_resident    = is_memory_resident_dataset(train_ds)
        loader_kw = resolve_loader_kwargs(args, is_cuda=is_cuda,
                                          data_on_device=data_on_device, memory_resident=mem_resident)

        # ── DataLoaders with DistributedSampler ───────────────────────────
        from torch.utils.data.distributed import DistributedSampler

        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) \
            if world_size > 1 else None
        valid_sampler = DistributedSampler(valid_ds, num_replicas=world_size, rank=rank, shuffle=False) \
            if world_size > 1 else None
        test_sampler  = DistributedSampler(test_ds,  num_replicas=world_size, rank=rank, shuffle=False) \
            if world_size > 1 and test_ds else None

        def _make_loader(ds, sampler, shuffle_flag):
            return torch.utils.data.DataLoader(
                ds, batch_size=args.batch_size,
                shuffle=(sampler is None and shuffle_flag),
                sampler=sampler, collate_fn=collate_fn, **loader_kw,
            )

        train_loader = _make_loader(train_ds, train_sampler, True)
        valid_loader = _make_loader(valid_ds, valid_sampler, False)
        test_loader  = _make_loader(test_ds,  test_sampler,  False) if test_ds else None

        # ── Model ──────────────────────────────────────────────────────────
        # Pass rank/world_size so the model builds EP or standard FFN.
        model = EncoderMoECTCModel(
            args, vocab_size=len(tokenizer.id_to_token),
            rank=rank, world_size=world_size if use_ep else 1,
        ).to(device)
        raw_model = model

        if world_size > 1 and not use_ep:
            # Standard DDP – all experts replicated, gradient sync automatic.
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            from torch.nn.parallel import DistributedDataParallel as DDP
            model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
            raw_model = model.module
            if is_main:
                print("Model wrapped in DistributedDataParallel (Data Parallel mode).", flush=True)
        elif use_ep:
            # EP mode: NO DDP wrapper.  Gradient reduction is done manually in
            # train_one_epoch via _reduce_non_expert_gradients().
            if is_main:
                print("Model in Expert Parallel mode (manual gradient reduction).", flush=True)
                if args.encoder_type == "conformer":
                    print(
                        "Conformer EP keeps BatchNorm1d local during forward; "
                        "running stats are averaged after optimizer steps.",
                        flush=True,
                    )

        if is_main:
            total_params = sum(p.numel() for p in raw_model.parameters())
            expert_params = sum(p.numel() for n, p in raw_model.named_parameters()
                                if any(s in n for s in _EP_EXPERT_SUBMODULES))
            print(f"Total params: {total_params:,}  "
                  f"Expert params per GPU: {expert_params:,}  "
                  f"Shared params: {total_params - expert_params:,}", flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_lr_scheduler(optimizer, args, len(train_loader))
        ctc_loss_fn = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True, reduction="mean")

        # ── Training loop ──────────────────────────────────────────────────
        best_cer = float("inf"); no_imp = 0
        eval_every = max(1, int(args.eval_every_epochs))
        patience   = max(0, int(args.early_stop_patience))
        history: list = []

        for epoch in range(1, args.epochs + 1):
            # Set epoch so DistributedSampler re-shuffles correctly each epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            if valid_sampler is not None:
                valid_sampler.set_epoch(epoch)

            if is_main:
                print(f"Starting epoch {epoch}/{args.epochs}", flush=True)

            train_m = train_one_epoch(
                model=model, loader=train_loader, tokenizer=tokenizer,
                optimizer=optimizer, ctc_loss=ctc_loss_fn, args=args, device=device,
                epoch=epoch, wandb_run=wandb_run, scaler=scaler, use_amp=use_amp,
                scheduler=scheduler, world_size=world_size, rank=rank,
            )

            # ── Temperature annealing ─────────────────────────────────────
            if int(getattr(args, "temperature_anneal_epochs", 0)) > 0:
                new_t = get_annealed_temperature(args, epoch)
                for m in raw_model.get_moe_modules():
                    m.temperature = new_t
                if is_main:
                    print(f"epoch={epoch} router_temperature={new_t:.4f}", flush=True)

            # ── Expert evolution ──────────────────────────────────────────
            evolve_logs: list = []
            if should_run_expert_evolution(args, epoch):
                # EP-aware evolve_experts handles cross-GPU state gathering internally.
                evolve_logs = evolve_experts(raw_model, valid_loader, ctc_loss_fn, args, device, use_amp)
                if evolve_logs and is_main:
                    save_json(output_dir / f"expert_evolution_epoch_{epoch}.json", {"events": evolve_logs})
                # Barrier: all ranks finish evolution before continuing
                if world_size > 1:
                    dist.barrier()

            should_eval = (epoch % eval_every == 0) or (epoch == args.epochs)
            if not should_eval:
                if is_main:
                    history.append({"epoch": epoch, "train_loss": round(train_m["loss"], 6),
                                    "train_cer": None, "expert_evolution": evolve_logs})
                    log_wandb_metrics(wandb_run, {
                        "global_step": epoch * len(train_loader), "epoch": epoch,
                        "train/loss_epoch": train_m["loss"],
                        "train/load_balance_loss_epoch": train_m["load_balance_loss"],
                        "train/competition_loss_epoch": train_m["competition_loss"],
                        "train/routing_entropy": train_m["routing_entropy"],
                        "train/grad_norm_epoch": train_m["grad_norm"],
                        "train/lr": train_m["lr"],
                    })
                continue

            valid_m = evaluate(
                model=model, loader=valid_loader, tokenizer=tokenizer,
                ctc_loss=ctc_loss_fn, args=args, device=device,
                stage=f"valid e{epoch}", use_amp=use_amp, epoch=epoch,
                world_size=world_size, rank=rank,
            )

            if is_main:
                print(
                    f"epoch={epoch} valid_loss={valid_m['loss']:.4f} "
                    f"valid_cer={valid_m['cer']:.4f} valid_wer={valid_m['wer']:.4f} "
                    f"mean_cer={valid_m['mean_cer']:.4f} mean_wer={valid_m['mean_wer']:.4f}",
                    flush=True,
                )

            is_best = valid_m["cer"] < best_cer
            if is_main:
                gstep = epoch * len(train_loader)
                log_payload = {
                    "global_step": gstep, "epoch": epoch,
                    "train/loss_epoch": train_m["loss"],
                    "train/base_loss_epoch": train_m["base_loss"],
                    "train/ctc_loss_epoch": train_m["ctc_loss"],
                    "train/load_balance_loss_epoch": train_m["load_balance_loss"],
                    "train/competition_loss_epoch": train_m["competition_loss"],
                    "train/routing_entropy": train_m["routing_entropy"],
                    "train/grad_norm_epoch": train_m["grad_norm"],
                    "train/lr": train_m["lr"],
                    "valid/loss": valid_m["loss"], "valid/cer": valid_m["cer"],
                    "valid/wer": valid_m["wer"], "valid/mean_cer": valid_m["mean_cer"],
                    "valid/mean_wer": valid_m["mean_wer"],
                    "valid/corpus_cer": valid_m["corpus_cer"],
                    "valid/corpus_wer": valid_m["corpus_wer"],
                    "valid/routing_entropy": valid_m["routing_entropy"],
                    "valid/is_best": int(is_best),
                    **flatten_routing_metrics("valid", valid_m["routing"]),
                    **flatten_scalar_metrics("valid/domain_loss", valid_m.get("domain_loss", {})),
                }
                append_vector_metrics(log_payload, "train/avg_gates", train_m.get("avg_gates", []))
                append_vector_metrics(log_payload, "valid/avg_gates", valid_m.get("avg_gates", []))
                append_vector_metrics(log_payload, "valid/expert_fitness", valid_m.get("expert_fitness", []))
                log_wandb_metrics(wandb_run, log_payload)

                epoch_rec = {
                    "epoch": epoch,
                    "train_loss": round(train_m["loss"], 6),
                    "valid_loss": round(valid_m["loss"], 6),
                    "valid_cer": round(valid_m["cer"], 6),
                    "valid_wer": round(valid_m["wer"], 6),
                    "expert_evolution": evolve_logs,
                }
                history.append(epoch_rec)

            if is_best:
                best_cer = valid_m["cer"]; no_imp = 0
                if is_main:
                    torch.save({
                        "model_state": raw_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler else None,
                        "config": vars(args),
                        "vocab": tokenizer.id_to_token,
                        "best_valid_cer": best_cer,
                        "expert_parallel": use_ep,
                        "world_size": world_size,
                    }, output_dir / "best.pt")
                    save_json(output_dir / "best_valid_metrics.json", valid_m)
                    log_wandb_metrics(wandb_run, {"global_step": epoch * len(train_loader),
                                                   "epoch": epoch, "valid/best_cer": best_cer})
            else:
                no_imp += 1
                if patience > 0 and no_imp >= patience:
                    if is_main:
                        print(f"Early stopping at epoch {epoch}.", flush=True)
                    break

        if is_main:
            save_json(output_dir / "train_history.json", {"epochs": history})

        # ── Test evaluation ────────────────────────────────────────────────
        if test_loader is not None and (output_dir / "best.pt").exists():
            ckpt = torch.load(output_dir / "best.pt", map_location=device)
            raw_model.load_state_dict(ckpt["model_state"])
            test_m = evaluate(
                model=model, loader=test_loader, tokenizer=tokenizer,
                ctc_loss=ctc_loss_fn, args=args, device=device, stage="test",
                use_amp=use_amp, epoch=history[-1]["epoch"] if history else args.epochs,
                world_size=world_size, rank=rank,
            )
            if is_main:
                save_json(output_dir / "test_metrics.json", test_m)
                print(f"test_cer={test_m['cer']:.4f} test_wer={test_m['wer']:.4f}", flush=True)

    finally:
        finish_wandb_run(wandb_run)
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
