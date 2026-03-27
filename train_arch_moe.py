from __future__ import annotations

import argparse
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
    build_collate_fn,
    build_dataset_for_mode,
    build_progress,
    choose_device,
    compute_cer,
    compute_wer,
    configure_runtime,
    create_grad_scaler,
    decode_batch,
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
    resolve_loader_kwargs,
    resolve_training_tokenizer,
    save_json,
    set_seed,
    summarize_routing,
    synchronize_for_timing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Transformer/Conformer ASR model with Competitive-Attractive SharedAdapterMoE."
    )
    parser.add_argument("--train-manifest", required=True, help="Training JSONL manifest.")
    parser.add_argument("--valid-manifest", required=True, help="Validation JSONL manifest.")
    parser.add_argument("--test-manifest", default=None, help="Optional test JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument(
        "--encoder-type",
        choices=("transformer", "conformer"),
        default="transformer",
        help="Sequence encoder backbone.",
    )
    parser.add_argument(
        "--ffn-type",
        choices=("dense", "shared_adapter_moe"),
        default="shared_adapter_moe",
        help="FFN block type inside the encoder.",
    )
    parser.add_argument("--num-experts", type=int, default=4, help="Number of experts in the MoE FFN.")
    parser.add_argument(
        "--router-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for the utterance-level gate.",
    )
    parser.add_argument(
        "--load-balance-weight",
        type=float,
        default=0.01,
        help="Weight for the routing load-balance regularizer.",
    )
    parser.add_argument(
        "--competition-weight",
        type=float,
        default=0.05,
        help="Weight for competition-aware KL routing supervision.",
    )
    parser.add_argument(
        "--competition-score",
        choices=("exp_neg_loss", "inverse_loss"),
        default="exp_neg_loss",
        help="How to convert per-sample expert loss into positive competition scores.",
    )
    parser.add_argument(
        "--competition-epsilon",
        type=float,
        default=1e-6,
        help="Numerical stability epsilon used in competition scoring and KL loss.",
    )
    parser.add_argument(
        "--competition-batches",
        type=int,
        default=0,
        help="If > 0, limit expensive competition scoring in eval/evolution to this many batches.",
    )
    parser.add_argument(
        "--competition-on-valid",
        action="store_true",
        help="Compute competition metrics during validation as well. Disabled by default for speed.",
    )
    parser.add_argument(
        "--competition-interval-steps",
        type=int,
        default=1,
        help="Refresh expensive competition targets every N training steps. 1 keeps previous behavior.",
    )
    parser.add_argument(
        "--competition-warmup-epochs",
        type=int,
        default=0,
        help="Skip competition loss for the first N epochs.",
    )
    parser.add_argument(
        "--competition-ramp-epochs",
        type=int,
        default=3,
        help="Linearly ramp competition weight for N epochs after warmup.",
    )
    parser.add_argument(
        "--expert-evolve-every-epochs",
        type=int,
        default=0,
        help="If > 0, periodically merge experts every N epochs.",
    )
    parser.add_argument(
        "--expert-evolve-start-epoch",
        type=int,
        default=3,
        help="Do not evolve experts before this epoch.",
    )
    parser.add_argument(
        "--expert-merge-alpha",
        type=float,
        default=0.5,
        help="Interpolation coefficient used in split-point expert merging.",
    )
    parser.add_argument(
        "--expert-merge-split-ratio",
        type=float,
        default=0.5,
        help="Split ratio in [0, 1] for split-point merging across flattened expert parameters.",
    )
    parser.add_argument(
        "--expert-merge-replace",
        choices=("worst", "random", "redundant"),
        default="worst",
        help="Which expert to replace with the newly merged child.",
    )
    parser.add_argument(
        "--expert-merge-blocks",
        type=int,
        default=0,
        help="Number of early MoE blocks to evolve. <= 0 means evolve all MoE blocks.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader prefetch factor.")
    add_data_pipeline_args(parser)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--scheduler",
        choices=("none", "warmup_cosine"),
        default="warmup_cosine",
        help="Learning-rate scheduler.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Explicit warmup steps for the scheduler. Overrides --warmup-ratio when > 0.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.05,
        help="Warmup ratio used when --warmup-steps is 0.",
    )
    parser.add_argument(
        "--min-lr-scale",
        type=float,
        default=0.1,
        help="Minimum LR scale reached by the cosine scheduler.",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping value.")
    parser.add_argument("--encoder-dim", type=int, default=256, help="Model dimension.")
    parser.add_argument("--encoder-layers", type=int, default=6, help="Number of encoder layers.")
    parser.add_argument("--num-heads", type=int, default=4, help="Attention heads.")
    parser.add_argument(
        "--ffn-hidden-dim",
        type=int,
        default=1024,
        help="Hidden size of the FFN expert trunk.",
    )
    parser.add_argument(
        "--adapter-hidden-dim",
        type=int,
        default=256,
        help="Hidden size of the expert adapter branch.",
    )
    parser.add_argument(
        "--projector-dim",
        type=int,
        default=256,
        help="Hidden size before the CTC classification head.",
    )
    parser.add_argument(
        "--conv-kernel-size",
        type=int,
        default=15,
        help="Depthwise convolution kernel size for Conformer blocks.",
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Expected sample rate.")
    parser.add_argument("--n-fft", type=int, default=512, help="FFT size for log-mel extraction.")
    parser.add_argument("--hop-length", type=int, default=160, help="Hop length for STFT.")
    parser.add_argument("--win-length", type=int, default=400, help="Window length for STFT.")
    parser.add_argument("--n-mels", type=int, default=80, help="Number of mel bins.")
    parser.add_argument("--max-audio-seconds", type=float, default=12.0, help="Crop longer audio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on. Use 'auto', 'cpu', or a CUDA device like 'cuda:0'.",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="Training log interval.")
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
        help="Weights & Biases project name. Leave unset to disable wandb logging.",
    )
    parser.add_argument("--wandb-entity", default=None, help="Optional Weights & Biases entity/team.")
    parser.add_argument("--wandb-run-name", default="arch-casamoe", help="Optional Weights & Biases run name.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases mode when logging is enabled.",
    )
    add_profiling_args(parser)
    return parser.parse_args()


if TORCH_IMPORT_ERROR is None:
    class SinusoidalPositionalEncoding(nn.Module):
        def __init__(self, model_dim: int, max_len: int = 10000):
            super().__init__()
            positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, model_dim, 2, dtype=torch.float32)
                * (-math.log(10000.0) / model_dim)
            )
            encoding = torch.zeros(max_len, model_dim, dtype=torch.float32)
            encoding[:, 0::2] = torch.sin(positions * div_term)
            encoding[:, 1::2] = torch.cos(positions * div_term)
            self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return hidden_states + self.encoding[:, : hidden_states.size(1)]


    class Conv2dSubsampling(nn.Module):
        def __init__(self, input_dim: int, output_dim: int, dropout: float):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, output_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            reduced_freq = ((input_dim + 1) // 2 + 1) // 2
            self.out = nn.Sequential(
                nn.Linear(output_dim * reduced_freq, output_dim),
                nn.Dropout(dropout),
            )

        def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
            return ((input_lengths + 1) // 2 + 1) // 2

        def forward(self, features: torch.Tensor, input_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            hidden = self.conv(features.unsqueeze(1))
            batch_size, channels, time_steps, freq_bins = hidden.shape
            hidden = hidden.transpose(1, 2).contiguous().view(batch_size, time_steps, channels * freq_bins)
            hidden = self.out(hidden)
            return hidden, self.output_lengths(input_lengths)


    class DenseFFN(nn.Module):
        def __init__(self, model_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(model_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, model_dim),
                nn.Dropout(dropout),
            )

        def forward(
            self,
            hidden_states: torch.Tensor,
            mask: torch.Tensor,
            forced_expert: int | None = None,
            return_all_experts: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
            del mask, forced_expert, return_all_experts
            return self.net(hidden_states), None, None


    class SharedAdapterMoEFFN(nn.Module):
        def __init__(
            self,
            model_dim: int,
            hidden_dim: int,
            adapter_hidden_dim: int,
            num_experts: int,
            temperature: float,
            dropout: float,
        ):
            super().__init__()
            self.temperature = float(temperature)
            self.num_experts = int(num_experts)
            self.router = nn.Linear(model_dim, num_experts)
            self.trunks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(model_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    )
                    for _ in range(num_experts)
                ]
            )
            self.share_down = nn.ModuleList([nn.Linear(hidden_dim, model_dim) for _ in range(num_experts)])
            self.adapter_up = nn.ModuleList([nn.Linear(hidden_dim, adapter_hidden_dim) for _ in range(num_experts)])
            self.adapter_down = nn.ModuleList([nn.Linear(adapter_hidden_dim, model_dim) for _ in range(num_experts)])
            self.dropout = nn.Dropout(dropout)

        def _pooled_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (hidden_states * mask.unsqueeze(-1)).sum(dim=1) / denom

        def _expert_forward(self, hidden_states: torch.Tensor, expert_idx: int) -> torch.Tensor:
            expert_hidden = self.trunks[expert_idx](hidden_states)
            share_out = self.share_down[expert_idx](expert_hidden)
            adapter_hidden = F.gelu(self.adapter_up[expert_idx](expert_hidden))
            adapter_out = self.adapter_down[expert_idx](self.dropout(adapter_hidden))
            return self.dropout(share_out + adapter_out)

        def forward(
            self,
            hidden_states: torch.Tensor,
            mask: torch.Tensor,
            forced_expert: int | None = None,
            return_all_experts: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor] | None]:
            pooled = self._pooled_hidden(hidden_states, mask)
            gates = torch.softmax(self.router(pooled) / self.temperature, dim=-1)

            if forced_expert is not None:
                output = self._expert_forward(hidden_states, forced_expert)
                forced_gates = torch.zeros_like(gates)
                forced_gates[:, forced_expert] = 1.0
                aux = {"pooled": pooled, "router_gates": gates}
                if return_all_experts:
                    aux["all_expert_outputs"] = output.unsqueeze(2)
                return output, forced_gates, aux

            expert_outputs = [self._expert_forward(hidden_states, idx) for idx in range(self.num_experts)]
            stacked = torch.stack(expert_outputs, dim=2)
            merged = torch.sum(stacked * gates.unsqueeze(1).unsqueeze(-1), dim=2)
            aux = {"pooled": pooled, "router_gates": gates}
            if return_all_experts:
                aux["all_expert_outputs"] = stacked
            return merged, gates, aux

        def get_expert_state(self, expert_idx: int) -> dict[str, dict[str, torch.Tensor]]:
            return {
                "trunk": {k: v.detach().clone() for k, v in self.trunks[expert_idx].state_dict().items()},
                "share_down": {k: v.detach().clone() for k, v in self.share_down[expert_idx].state_dict().items()},
                "adapter_up": {k: v.detach().clone() for k, v in self.adapter_up[expert_idx].state_dict().items()},
                "adapter_down": {k: v.detach().clone() for k, v in self.adapter_down[expert_idx].state_dict().items()},
            }

        def set_expert_state(self, expert_idx: int, state_dict_like: dict[str, dict[str, torch.Tensor]]) -> None:
            self.trunks[expert_idx].load_state_dict(state_dict_like["trunk"])
            self.share_down[expert_idx].load_state_dict(state_dict_like["share_down"])
            self.adapter_up[expert_idx].load_state_dict(state_dict_like["adapter_up"])
            self.adapter_down[expert_idx].load_state_dict(state_dict_like["adapter_down"])

        @staticmethod
        def _flatten_state_dict_tensors(
            state_dict_like: dict[str, dict[str, torch.Tensor]]
        ) -> list[tuple[str, str, torch.Tensor]]:
            flattened: list[tuple[str, str, torch.Tensor]] = []
            for group_name, group in state_dict_like.items():
                for tensor_name, tensor in group.items():
                    flattened.append((group_name, tensor_name, tensor))
            return flattened

        @staticmethod
        def _merge_tensor(
            tensor_a: torch.Tensor,
            tensor_b: torch.Tensor,
            *,
            alpha: float,
            split_index: int,
            tensor_offset: int,
        ) -> torch.Tensor:
            flat_a = tensor_a.reshape(-1)
            flat_b = tensor_b.to(tensor_a.device, dtype=tensor_a.dtype).reshape(-1)
            flat_out = torch.empty_like(flat_a)
            local_split = max(0, min(flat_a.numel(), split_index - tensor_offset))
            if local_split > 0:
                flat_out[:local_split] = flat_a[:local_split] * (1.0 - alpha) + flat_b[:local_split] * alpha
            if local_split < flat_a.numel():
                beta = 1.0 - alpha
                flat_out[local_split:] = flat_a[local_split:] * (1.0 - beta) + flat_b[local_split:] * beta
            return flat_out.view_as(tensor_a)

        def merge_experts(
            self,
            parent_a: int,
            parent_b: int,
            child_idx: int | None = None,
            alpha: float = 0.5,
            split_ratio: float = 0.5,
            mode: str = "split_linear",
        ) -> dict[str, dict[str, torch.Tensor]]:
            if mode != "split_linear":
                raise ValueError(f"Unsupported merge mode: {mode}")

            state_a = self.get_expert_state(parent_a)
            state_b = self.get_expert_state(parent_b)
            flat_a = self._flatten_state_dict_tensors(state_a)
            flat_b = self._flatten_state_dict_tensors(state_b)
            total_numel = sum(tensor.numel() for _, _, tensor in flat_a)
            split_index = int(round(max(0.0, min(1.0, split_ratio)) * total_numel))

            merged_state: dict[str, dict[str, torch.Tensor]] = {
                "trunk": {},
                "share_down": {},
                "adapter_up": {},
                "adapter_down": {},
            }
            offset = 0
            for (group_name, tensor_name, tensor_a), (_, _, tensor_b) in zip(flat_a, flat_b):
                merged_state[group_name][tensor_name] = self._merge_tensor(
                    tensor_a,
                    tensor_b,
                    alpha=float(alpha),
                    split_index=split_index,
                    tensor_offset=offset,
                )
                offset += tensor_a.numel()

            if child_idx is not None:
                self.set_expert_state(child_idx, merged_state)
            return merged_state


    class ConformerConvModule(nn.Module):
        def __init__(self, model_dim: int, kernel_size: int, dropout: float):
            super().__init__()
            self.layer_norm = nn.LayerNorm(model_dim)
            self.pointwise_in = nn.Conv1d(model_dim, 2 * model_dim, kernel_size=1)
            self.depthwise = nn.Conv1d(
                model_dim,
                model_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=model_dim,
            )
            self.batch_norm = nn.BatchNorm1d(model_dim)
            self.pointwise_out = nn.Conv1d(model_dim, model_dim, kernel_size=1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden = self.layer_norm(hidden_states).transpose(1, 2)
            hidden = F.glu(self.pointwise_in(hidden), dim=1)
            hidden = self.depthwise(hidden)
            hidden = self.batch_norm(hidden)
            hidden = F.silu(hidden)
            hidden = self.pointwise_out(hidden)
            return self.dropout(hidden.transpose(1, 2))


    class TransformerMoEBlock(nn.Module):
        def __init__(self, args: argparse.Namespace):
            super().__init__()
            self.self_attn_norm = nn.LayerNorm(args.encoder_dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=args.encoder_dim,
                num_heads=args.num_heads,
                dropout=args.dropout,
                batch_first=True,
            )
            self.dropout = nn.Dropout(args.dropout)
            self.ffn_norm = nn.LayerNorm(args.encoder_dim)
            if args.ffn_type == "dense":
                self.ffn = DenseFFN(args.encoder_dim, args.ffn_hidden_dim, args.dropout)
            else:
                self.ffn = SharedAdapterMoEFFN(
                    model_dim=args.encoder_dim,
                    hidden_dim=args.ffn_hidden_dim,
                    adapter_hidden_dim=args.adapter_hidden_dim,
                    num_experts=args.num_experts,
                    temperature=args.router_temperature,
                    dropout=args.dropout,
                )

        def forward(
            self,
            hidden_states: torch.Tensor,
            mask: torch.Tensor,
            forced_expert: int | None = None,
            return_all_experts: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any] | None]:
            key_padding_mask = ~mask.bool()
            attn_input = self.self_attn_norm(hidden_states)
            attn_output, _ = self.self_attn(
                attn_input,
                attn_input,
                attn_input,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hidden_states = hidden_states + self.dropout(attn_output)
            ffn_output, routing, aux = self.ffn(
                self.ffn_norm(hidden_states),
                mask.float(),
                forced_expert=forced_expert,
                return_all_experts=return_all_experts,
            )
            hidden_states = hidden_states + self.dropout(ffn_output)
            return hidden_states, routing, aux


    class ConformerMoEBlock(nn.Module):
        def __init__(self, args: argparse.Namespace):
            super().__init__()
            self.macaron_norm = nn.LayerNorm(args.encoder_dim)
            self.macaron_ffn = DenseFFN(args.encoder_dim, args.ffn_hidden_dim, args.dropout)
            self.self_attn_norm = nn.LayerNorm(args.encoder_dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=args.encoder_dim,
                num_heads=args.num_heads,
                dropout=args.dropout,
                batch_first=True,
            )
            self.conv_module = ConformerConvModule(
                model_dim=args.encoder_dim,
                kernel_size=args.conv_kernel_size,
                dropout=args.dropout,
            )
            self.ffn_norm = nn.LayerNorm(args.encoder_dim)
            if args.ffn_type == "dense":
                self.ffn = DenseFFN(args.encoder_dim, args.ffn_hidden_dim, args.dropout)
            else:
                self.ffn = SharedAdapterMoEFFN(
                    model_dim=args.encoder_dim,
                    hidden_dim=args.ffn_hidden_dim,
                    adapter_hidden_dim=args.adapter_hidden_dim,
                    num_experts=args.num_experts,
                    temperature=args.router_temperature,
                    dropout=args.dropout,
                )
            self.final_norm = nn.LayerNorm(args.encoder_dim)
            self.dropout = nn.Dropout(args.dropout)

        def forward(
            self,
            hidden_states: torch.Tensor,
            mask: torch.Tensor,
            forced_expert: int | None = None,
            return_all_experts: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any] | None]:
            macaron_out, _, _ = self.macaron_ffn(self.macaron_norm(hidden_states), mask.float())
            hidden_states = hidden_states + 0.5 * self.dropout(macaron_out)

            key_padding_mask = ~mask.bool()
            attn_input = self.self_attn_norm(hidden_states)
            attn_output, _ = self.self_attn(
                attn_input,
                attn_input,
                attn_input,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            hidden_states = hidden_states + self.dropout(attn_output)
            hidden_states = hidden_states + self.conv_module(hidden_states)

            ffn_output, routing, aux = self.ffn(
                self.ffn_norm(hidden_states),
                mask.float(),
                forced_expert=forced_expert,
                return_all_experts=return_all_experts,
            )
            hidden_states = hidden_states + 0.5 * self.dropout(ffn_output)
            return self.final_norm(hidden_states), routing, aux


    class EncoderMoECTCModel(nn.Module):
        def __init__(self, args: argparse.Namespace, vocab_size: int):
            super().__init__()
            self.num_experts = int(args.num_experts)
            self.ffn_type = args.ffn_type
            self.subsampling = Conv2dSubsampling(args.n_mels, args.encoder_dim, args.dropout)
            self.position = SinusoidalPositionalEncoding(args.encoder_dim)
            block_cls = TransformerMoEBlock if args.encoder_type == "transformer" else ConformerMoEBlock
            self.blocks = nn.ModuleList([block_cls(args) for _ in range(args.encoder_layers)])
            self.output_norm = nn.LayerNorm(args.encoder_dim)
            self.projector = nn.Sequential(
                nn.Linear(args.encoder_dim, args.projector_dim),
                nn.GELU(),
                nn.Dropout(args.dropout),
            )
            self.ctc_head = nn.Linear(args.projector_dim, vocab_size)

        def forward(
            self,
            inputs: torch.Tensor,
            input_lengths: torch.Tensor,
            forced_expert: int | None = None,
            forced_experts: dict[int, int] | None = None,
            return_aux: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any] | None]:
            hidden_states, output_lengths = self.subsampling(inputs, input_lengths.to(inputs.device))
            hidden_states = self.position(hidden_states)
            mask = lengths_to_mask(output_lengths.to(hidden_states.device), hidden_states.size(1))

            routing_values: list[torch.Tensor] = []
            block_aux: list[dict[str, Any]] = []
            for block_idx, block in enumerate(self.blocks):
                block_forced_expert = forced_expert
                if forced_experts is not None:
                    block_forced_expert = forced_experts.get(block_idx)
                hidden_states, routing, aux = block(
                    hidden_states,
                    mask,
                    forced_expert=block_forced_expert,
                    return_all_experts=return_aux,
                )
                if routing is not None:
                    routing_values.append(routing)
                if return_aux:
                    block_aux.append(
                        {
                            "block_index": block_idx,
                            "routing": routing,
                            "aux": aux,
                        }
                    )

            hidden_states = self.output_norm(hidden_states)
            hidden_states = self.projector(hidden_states)
            logits = self.ctc_head(hidden_states)
            log_probs = F.log_softmax(logits, dim=-1)
            merged_routing = torch.stack(routing_values, dim=0).mean(dim=0) if routing_values else None
            aux_out = None
            if return_aux:
                aux_out = {"block_aux": block_aux, "mask": mask, "output_lengths": output_lengths}
            return log_probs, output_lengths, merged_routing, aux_out

        def get_moe_modules(self) -> list[SharedAdapterMoEFFN]:
            modules: list[SharedAdapterMoEFFN] = []
            for block in self.blocks:
                ffn = getattr(block, "ffn", None)
                if isinstance(ffn, SharedAdapterMoEFFN):
                    modules.append(ffn)
            return modules


else:
    SinusoidalPositionalEncoding = None
    Conv2dSubsampling = None
    DenseFFN = None
    SharedAdapterMoEFFN = None
    ConformerConvModule = None
    TransformerMoEBlock = None
    ConformerMoEBlock = None
    EncoderMoECTCModel = None


def routing_regularizer(avg_gates: torch.Tensor | None, num_experts: int) -> torch.Tensor:
    if avg_gates is None:
        return torch.tensor(0.0, dtype=torch.float32)
    expected = torch.full(
        (num_experts,),
        1.0 / num_experts,
        device=avg_gates.device,
        dtype=avg_gates.dtype,
    )
    return F.mse_loss(avg_gates.mean(dim=0), expected)


def routing_entropy(gates: torch.Tensor | None, eps: float = 1e-8) -> torch.Tensor:
    if gates is None:
        return torch.tensor(0.0, dtype=torch.float32)
    safe_gates = gates.clamp_min(eps)
    return -(safe_gates * safe_gates.log()).sum(dim=-1).mean()


def split_ctc_targets(targets: torch.Tensor, target_lengths: torch.Tensor) -> list[torch.Tensor]:
    segments: list[torch.Tensor] = []
    offset = 0
    for length in target_lengths.tolist():
        next_offset = offset + int(length)
        segments.append(targets[offset:next_offset])
        offset = next_offset
    return segments


def convert_loss_to_score(losses: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.competition_score == "inverse_loss":
        return 1.0 / losses.clamp_min(args.competition_epsilon)
    return torch.exp(-losses)


def compute_per_sample_ctc_losses(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    output_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    ctc_loss,
) -> torch.Tensor:
    blank_id = int(getattr(ctc_loss, "blank", 0))
    zero_infinity = bool(getattr(ctc_loss, "zero_infinity", True))
    raw_losses = F.ctc_loss(
        log_probs.transpose(0, 1),
        targets,
        output_lengths.cpu(),
        target_lengths.cpu(),
        blank=blank_id,
        reduction="none",
        zero_infinity=zero_infinity,
    )
    return raw_losses / target_lengths.to(raw_losses.device, dtype=raw_losses.dtype).clamp_min(1.0)


@torch.no_grad()
def compute_expert_scores(
    model: EncoderMoECTCModel,
    batch: dict[str, Any],
    ctc_loss,
    args: argparse.Namespace,
    device: str,
    *,
    use_amp: bool = False,
    block_idx: int | None = None,
) -> torch.Tensor | None:
    if args.ffn_type != "shared_adapter_moe":
        return None

    scores = []
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16

    for expert_idx in range(args.num_experts):
        forward_kwargs: dict[str, Any] = {"return_aux": False}
        if block_idx is None:
            forward_kwargs["forced_expert"] = expert_idx
        else:
            forward_kwargs["forced_experts"] = {block_idx: expert_idx}

        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            expert_log_probs, expert_output_lengths, _, _ = model(
                batch["inputs"],
                batch["input_lengths"],
                **forward_kwargs,
            )

        loss_tensor = compute_per_sample_ctc_losses(
            expert_log_probs,
            batch["targets"],
            expert_output_lengths,
            batch["target_lengths"],
            ctc_loss,
        ).detach()
        scores.append(convert_loss_to_score(loss_tensor, args))

    return torch.stack(scores, dim=1)


def competition_targets(
    expert_scores: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = expert_scores.sum(dim=-1, keepdim=True).clamp_min(args.competition_epsilon)
    q = expert_scores / z
    fitness = q.sum(dim=0)
    return q, fitness, z.squeeze(-1)


def routing_alignment_loss(
    gates: torch.Tensor | None,
    targets: torch.Tensor | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if gates is None or targets is None:
        device = gates.device if gates is not None else (targets.device if targets is not None else None)
        return torch.tensor(0.0, dtype=torch.float32, device=device)
    safe_gates = gates.clamp_min(eps)
    safe_targets = targets.clamp_min(eps)
    return F.kl_div(safe_gates.log(), safe_targets, reduction="batchmean")


def select_expert_parents(
    scores: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[int, int, dict[str, Any]]:
    if scores.ndim != 2:
        raise ValueError(f"Expected scores with shape [B, E], got {tuple(scores.shape)}")
    num_experts = scores.size(1)
    if num_experts == 0:
        raise ValueError("Cannot select parents from zero experts.")
    if num_experts == 1:
        return 0, 0, {"fitness": [1.0], "attraction": [0.0]}

    z = scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = scores / z
    fitness = q.sum(dim=0)
    parent_a = int(torch.argmax(fitness).item())

    attraction = torch.zeros(num_experts, device=scores.device, dtype=scores.dtype)
    denom = z.squeeze(-1) + eps
    for candidate_idx in range(num_experts):
        if candidate_idx == parent_a:
            attraction[candidate_idx] = float("-inf")
            continue
        gains = torch.relu(scores[:, candidate_idx] - scores[:, parent_a]) / denom
        attraction[candidate_idx] = gains.sum()

    if torch.isfinite(attraction).any():
        parent_b = int(torch.argmax(attraction).item())
    else:
        sorted_fitness = torch.argsort(fitness, descending=True)
        parent_b = int(sorted_fitness[1].item())

    diagnostics = {
        "fitness": [round(float(v), 6) for v in fitness.detach().cpu().tolist()],
        "attraction": [
            round(float(v), 6) if math.isfinite(float(v)) else None
            for v in attraction.detach().cpu().tolist()
        ],
        "mean_partition": round(float(z.mean().item()), 6),
    }
    return parent_a, parent_b, diagnostics


def collect_moe_modules(
    model: EncoderMoECTCModel,
    block_limit: int,
) -> list[tuple[int, SharedAdapterMoEFFN]]:
    modules: list[tuple[int, SharedAdapterMoEFFN]] = []
    for block_idx, block in enumerate(model.blocks):
        ffn = getattr(block, "ffn", None)
        if isinstance(ffn, SharedAdapterMoEFFN):
            modules.append((block_idx, ffn))
    if block_limit > 0:
        modules = modules[:block_limit]
    return modules


def select_replacement_expert(
    fitness: torch.Tensor,
    usage: torch.Tensor,
    *,
    parent_a: int,
    parent_b: int,
    strategy: str,
) -> tuple[int, dict[str, Any]]:
    num_experts = fitness.numel()
    eligible = [idx for idx in range(num_experts) if idx not in {parent_a, parent_b}]
    if not eligible:
        eligible = list(range(num_experts))

    if strategy == "random":
        replace_idx = random.choice(eligible)
        return replace_idx, {"strategy": strategy}

    if strategy == "redundant":
        fitness_norm = fitness / fitness.sum().clamp_min(1e-8)
        usage_norm = usage / usage.sum().clamp_min(1e-8)
        redundancy_score = fitness_norm + usage_norm
        eligible_scores = torch.tensor([float(redundancy_score[idx].item()) for idx in eligible])
        best_local = int(torch.argmin(eligible_scores).item())
        replace_idx = eligible[best_local]
        return replace_idx, {
            "strategy": strategy,
            "redundancy_score": [round(float(v), 6) for v in redundancy_score.detach().cpu().tolist()],
        }

    eligible_fitness = torch.tensor([float(fitness[idx].item()) for idx in eligible])
    best_local = int(torch.argmin(eligible_fitness).item())
    replace_idx = eligible[best_local]
    return replace_idx, {"strategy": "worst"}


def should_compute_competition_metrics(
    args: argparse.Namespace,
    stage: str,
    batch_idx: int,
    epoch: int | None = None,
) -> bool:
    if get_effective_competition_weight(args, epoch) <= 0.0:
        return False
    if stage.startswith("valid") and not args.competition_on_valid:
        return False
    if args.competition_batches > 0 and batch_idx > args.competition_batches:
        return False
    return True


def get_effective_competition_weight(args: argparse.Namespace, epoch: int | None) -> float:
    if args.ffn_type != "shared_adapter_moe" or args.competition_weight <= 0.0:
        return 0.0
    if epoch is None:
        return float(args.competition_weight)
    if epoch <= args.competition_warmup_epochs:
        return 0.0
    ramp_epochs = max(0, int(args.competition_ramp_epochs))
    if ramp_epochs == 0:
        return float(args.competition_weight)
    ramp_progress = min(1.0, max(0.0, (epoch - args.competition_warmup_epochs) / ramp_epochs))
    return float(args.competition_weight) * ramp_progress


def should_compute_train_competition(args: argparse.Namespace, epoch: int, step: int) -> bool:
    if get_effective_competition_weight(args, epoch) <= 0.0:
        return False
    interval = max(1, int(args.competition_interval_steps))
    return step % interval == 0


def should_run_expert_evolution(args: argparse.Namespace, epoch: int) -> bool:
    if args.ffn_type != "shared_adapter_moe" or args.expert_evolve_every_epochs <= 0:
        return False
    if epoch < max(1, int(args.expert_evolve_start_epoch)):
        return False
    if get_effective_competition_weight(args, epoch) <= 0.0:
        return False
    return epoch % args.expert_evolve_every_epochs == 0


def flatten_scalar_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    payload: dict[str, float] = {}
    for raw_key, value in metrics.items():
        safe_key = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(raw_key)).strip("_") or "unknown"
        payload[f"{prefix}/{safe_key}"] = float(value)
    return payload


def build_lr_scheduler(optimizer, args: argparse.Namespace, steps_per_epoch: int):
    if args.scheduler == "none":
        return None
    total_steps = max(1, int(args.epochs) * max(1, steps_per_epoch))
    warmup_steps = int(args.warmup_steps)
    if warmup_steps <= 0:
        warmup_steps = int(round(total_steps * float(args.warmup_ratio)))
    warmup_steps = min(max(0, warmup_steps), max(0, total_steps - 1))
    min_lr_scale = float(args.min_lr_scale)

    def lr_lambda(current_step: int) -> float:
        step = current_step + 1
        if warmup_steps > 0 and step <= warmup_steps:
            return max(1e-8, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(
    model: EncoderMoECTCModel,
    loader,
    tokenizer: CharTokenizer,
    ctc_loss,
    args: argparse.Namespace,
    device: str,
    stage: str = "eval",
    use_amp: bool = False,
    epoch: int | None = None,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_base_loss = 0.0
    total_lb_loss = 0.0
    total_comp_loss = 0.0
    total_cer = 0.0
    total_wer = 0.0
    total_entropy = 0.0
    total_gate_mass: torch.Tensor | None = None
    total_gate_count = 0
    samples = 0
    total_weight = 0
    routing_by_domain: dict[str, list[torch.Tensor]] = defaultdict(list)
    domain_loss_sum: dict[str, float] = defaultdict(float)
    domain_loss_count: dict[str, int] = defaultdict(int)
    expert_fitness_storage: list[torch.Tensor] = []
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16
    effective_comp_weight = get_effective_competition_weight(args, epoch)

    iterator = build_progress(loader, total=len(loader), desc=stage, leave=False)
    for step, batch in enumerate(iterator, start=1):
        batch = move_batch_to_device(batch, device, non_blocking=device.startswith("cuda"))
        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            log_probs, output_lengths, routing, _ = model(batch["inputs"], batch["input_lengths"], return_aux=False)
            per_sample_base_loss = compute_per_sample_ctc_losses(
                log_probs,
                batch["targets"],
                output_lengths,
                batch["target_lengths"],
                ctc_loss,
            )
            base_loss = per_sample_base_loss.mean()
            lb_loss = (
                routing_regularizer(routing, args.num_experts)
                if routing is not None
                else torch.tensor(0.0, device=log_probs.device)
            )

        comp_loss = torch.tensor(0.0, device=log_probs.device)
        if routing is not None and should_compute_competition_metrics(args, stage, step, epoch=epoch):
            expert_scores = compute_expert_scores(model, batch, ctc_loss, args, device, use_amp=False)
            if expert_scores is not None:
                comp_targets, fitness, _ = competition_targets(expert_scores, args)
                comp_loss = routing_alignment_loss(routing, comp_targets, eps=args.competition_epsilon)
                expert_fitness_storage.append(fitness.detach().cpu())

        hypotheses = decode_batch(log_probs, output_lengths, tokenizer)
        batch_size = len(hypotheses)
        batch_shared_penalty = args.load_balance_weight * lb_loss.detach() + effective_comp_weight * comp_loss.detach()
        sample_total_loss = per_sample_base_loss.detach() + batch_shared_penalty
        total_weight += batch_size
        total_loss += float(sample_total_loss.sum().item())
        total_base_loss += float(per_sample_base_loss.detach().sum().item())
        total_lb_loss += float(lb_loss.detach().item()) * batch_size
        total_comp_loss += float(comp_loss.detach().item()) * batch_size
        for idx, (ref, hyp) in enumerate(zip(batch["texts"], hypotheses)):
            total_cer += compute_cer(ref, hyp)
            total_wer += compute_wer(ref, hyp)
            domain = batch["domains"][idx]
            domain_loss_sum[domain] += float(sample_total_loss[idx].item())
            domain_loss_count[domain] += 1
            if routing is not None:
                routing_by_domain[domain].append(routing[idx].detach().cpu())
            samples += 1

        if routing is not None:
            batch_entropy = routing_entropy(routing, eps=args.competition_epsilon)
            total_entropy += float(batch_entropy.item()) * batch_size
            gate_sum = routing.detach().sum(dim=0).cpu()
            total_gate_mass = gate_sum if total_gate_mass is None else total_gate_mass + gate_sum
            total_gate_count += routing.size(0)

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{total_loss / max(1, total_weight):.4f}", cer=f"{total_cer / max(1, samples):.4f}")

    avg_gates = []
    if total_gate_mass is not None and total_gate_count > 0:
        avg_gates = [round(float(v), 6) for v in (total_gate_mass / total_gate_count).tolist()]

    metrics = {
        "loss": total_loss / max(1, total_weight),
        "total_loss": total_loss / max(1, total_weight),
        "base_loss": total_base_loss / max(1, total_weight),
        "ctc_loss": total_base_loss / max(1, total_weight),
        "load_balance_loss": total_lb_loss / max(1, total_weight),
        "competition_loss": total_comp_loss / max(1, total_weight),
        "cer": total_cer / max(1, samples),
        "wer": total_wer / max(1, samples),
        "routing": summarize_routing(routing_by_domain),
        "avg_gates": avg_gates,
        "expert_usage": avg_gates,
        "routing_entropy": total_entropy / max(1, samples),
        "domain_loss": {
            domain: round(domain_loss_sum[domain] / max(1, domain_loss_count[domain]), 6)
            for domain in sorted(domain_loss_sum)
        },
        "effective_competition_weight": effective_comp_weight,
    }
    if expert_fitness_storage:
        metrics["expert_fitness"] = [
            round(float(v), 6) for v in torch.stack(expert_fitness_storage).mean(dim=0).tolist()
        ]
    else:
        metrics["expert_fitness"] = []
    if was_training:
        model.train()
    return metrics


def train_one_epoch(
    model: EncoderMoECTCModel,
    loader,
    tokenizer: CharTokenizer,
    optimizer,
    ctc_loss,
    args: argparse.Namespace,
    device: str,
    epoch: int,
    wandb_run=None,
    scaler=None,
    use_amp: bool = False,
    scheduler=None,
) -> dict[str, Any]:
    del tokenizer
    model.train()
    running_loss = 0.0
    running_base_loss = 0.0
    running_lb_loss = 0.0
    running_comp_loss = 0.0
    running_entropy = 0.0
    running_grad_norm = 0.0
    running_gate_mass: torch.Tensor | None = None
    running_gate_count = 0
    running_total_weight = 0
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16
    iterator = build_progress(loader, total=len(loader), desc=f"train e{epoch}", leave=False)
    profile_enabled = bool(getattr(args, "profile_performance", False))
    effective_comp_weight = get_effective_competition_weight(args, epoch)
    timing_sums = {
        "data": 0.0,
        "transfer": 0.0,
        "forward": 0.0,
        "competition": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }
    loop_end_time = time.perf_counter()
    iterator_obj = iter(iterator)
    if profile_enabled:
        print(f"epoch={epoch} awaiting first batch...", flush=True)

    for step in range(1, len(loader) + 1):
        try:
            batch = next(iterator_obj)
        except StopIteration:
            break
        data_wait = time.perf_counter() - loop_end_time
        timing_sums["data"] += data_wait
        if profile_enabled and step == 1:
            print(f"epoch={epoch} first batch fetched after {data_wait:.4f}s", flush=True)

        transfer_start = time.perf_counter()
        batch = move_batch_to_device(batch, device, non_blocking=device.startswith("cuda"))
        synchronize_for_timing(device, profile_enabled)
        transfer_time = time.perf_counter() - transfer_start
        timing_sums["transfer"] += transfer_time
        if profile_enabled and step == 1:
            print(f"epoch={epoch} first batch moved to device in {transfer_time:.4f}s", flush=True)
        optimizer.zero_grad(set_to_none=True)

        forward_start = time.perf_counter()
        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            log_probs, output_lengths, routing, _ = model(batch["inputs"], batch["input_lengths"], return_aux=False)
            base_loss = ctc_loss(
                log_probs.transpose(0, 1),
                batch["targets"],
                output_lengths.cpu(),
                batch["target_lengths"],
            )
            lb_loss = (
                routing_regularizer(routing, args.num_experts)
                if routing is not None
                else torch.tensor(0.0, device=log_probs.device)
            )
        synchronize_for_timing(device, profile_enabled)
        forward_time = time.perf_counter() - forward_start
        timing_sums["forward"] += forward_time
        if profile_enabled and step == 1:
            print(f"epoch={epoch} first batch forward done in {forward_time:.4f}s", flush=True)

        comp_loss = torch.tensor(0.0, device=log_probs.device)
        competition_start = time.perf_counter()
        if routing is not None and should_compute_train_competition(args, epoch, step):
            was_training = model.training
            model.eval()
            expert_scores = compute_expert_scores(model, batch, ctc_loss, args, device, use_amp=False)
            if was_training:
                model.train()
            if expert_scores is not None:
                comp_targets, _, _ = competition_targets(expert_scores, args)
                comp_loss = routing_alignment_loss(routing, comp_targets.detach(), eps=args.competition_epsilon)
        synchronize_for_timing(device, profile_enabled)
        competition_time = time.perf_counter() - competition_start
        timing_sums["competition"] += competition_time
        if profile_enabled and step == 1:
            print(f"epoch={epoch} first batch competition done in {competition_time:.4f}s", flush=True)

        loss = base_loss + args.load_balance_weight * lb_loss + effective_comp_weight * comp_loss

        backward_start = time.perf_counter()
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        synchronize_for_timing(device, profile_enabled)
        backward_time = time.perf_counter() - backward_start
        timing_sums["backward"] += backward_time
        if profile_enabled and step == 1:
            print(f"epoch={epoch} first batch backward done in {backward_time:.4f}s", flush=True)

        optimizer_start = time.perf_counter()
        if scaler is not None and scaler.is_enabled():
            scaler.unscale_(optimizer)
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
        else:
            grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        synchronize_for_timing(device, profile_enabled)
        optimizer_time = time.perf_counter() - optimizer_start
        timing_sums["optimizer"] += optimizer_time
        if profile_enabled and step == 1:
            print(
                f"epoch={epoch} first batch optimizer done in {optimizer_time:.4f}s "
                f"loss={float(loss.item()):.4f} grad={grad_norm:.4f}",
                flush=True,
            )

        batch_size = int(batch["input_lengths"].size(0))
        running_total_weight += batch_size
        running_loss += float(loss.item()) * batch_size
        running_base_loss += float(base_loss.item()) * batch_size
        running_lb_loss += float(lb_loss.item()) * batch_size
        running_comp_loss += float(comp_loss.item()) * batch_size
        running_grad_norm += grad_norm * batch_size
        if routing is not None:
            running_entropy += float(routing_entropy(routing, eps=args.competition_epsilon).item()) * routing.size(0)
            gate_sum = routing.detach().sum(dim=0).cpu()
            running_gate_mass = gate_sum if running_gate_mass is None else running_gate_mass + gate_sum
            running_gate_count += routing.size(0)

        avg_loss = running_loss / max(1, running_total_weight)
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(loss=f"{avg_loss:.4f}")

        if step % args.log_interval == 0 or step == len(loader):
            current_lr = float(optimizer.param_groups[0]["lr"])
            print(
                f"epoch={epoch} step={step}/{len(loader)} train_loss={avg_loss:.4f} "
                f"ctc={running_base_loss / max(1, running_total_weight):.4f} "
                f"lb={running_lb_loss / max(1, running_total_weight):.4f} "
                f"comp={running_comp_loss / max(1, running_total_weight):.4f} "
                f"grad={running_grad_norm / max(1, running_total_weight):.4f} "
                f"lr={current_lr:.6g}",
                flush=True,
            )
            log_wandb_metrics(
                wandb_run,
                {
                    "global_step": (epoch - 1) * len(loader) + step,
                    "epoch": epoch,
                    "train/loss_step": avg_loss,
                    "train/base_loss_step": running_base_loss / max(1, running_total_weight),
                    "train/ctc_loss_step": running_base_loss / max(1, running_total_weight),
                    "train/load_balance_loss_step": running_lb_loss / max(1, running_total_weight),
                    "train/competition_loss_step": running_comp_loss / max(1, running_total_weight),
                    "train/grad_norm_step": running_grad_norm / max(1, running_total_weight),
                    "train/lr": current_lr,
                },
            )

        if profile_enabled and (step % max(1, args.log_timing_every) == 0 or step == len(loader)):
            avg_timing = {key: value / step for key, value in timing_sums.items()}
            print(
                f"timing epoch={epoch} step={step}/{len(loader)} "
                f"data={avg_timing['data']:.4f}s transfer={avg_timing['transfer']:.4f}s "
                f"forward={avg_timing['forward']:.4f}s competition={avg_timing['competition']:.4f}s "
                f"backward={avg_timing['backward']:.4f}s optimizer={avg_timing['optimizer']:.4f}s",
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
                    "timing/competition": avg_timing["competition"],
                    "timing/backward": avg_timing["backward"],
                    "timing/optimizer": avg_timing["optimizer"],
                },
            )
        loop_end_time = time.perf_counter()

    avg_gates = []
    if running_gate_mass is not None and running_gate_count > 0:
        avg_gates = [round(float(v), 6) for v in (running_gate_mass / running_gate_count).tolist()]

    return {
        "loss": running_loss / max(1, running_total_weight),
        "base_loss": running_base_loss / max(1, running_total_weight),
        "ctc_loss": running_base_loss / max(1, running_total_weight),
        "load_balance_loss": running_lb_loss / max(1, running_total_weight),
        "competition_loss": running_comp_loss / max(1, running_total_weight),
        "routing_entropy": running_entropy / max(1, running_gate_count),
        "grad_norm": running_grad_norm / max(1, running_total_weight),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "effective_competition_weight": effective_comp_weight,
        "avg_gates": avg_gates,
        "expert_usage": avg_gates,
    }


@torch.no_grad()
def collect_evolution_statistics(
    model: EncoderMoECTCModel,
    loader,
    ctc_loss,
    args: argparse.Namespace,
    device: str,
    use_amp: bool = False,
) -> list[dict[str, Any]]:
    if args.ffn_type != "shared_adapter_moe":
        return []

    selected_modules = collect_moe_modules(model, args.expert_merge_blocks)
    if not selected_modules:
        return []

    block_stats: dict[int, dict[str, Any]] = {
        block_idx: {
            "module": moe_module,
            "scores": [],
            "usage_sum": torch.zeros(args.num_experts),
            "usage_count": 0,
        }
        for block_idx, moe_module in selected_modules
    }

    max_batches = args.competition_batches if args.competition_batches > 0 else len(loader)
    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    autocast_dtype = torch.float16 if autocast_device == "cuda" else torch.bfloat16

    model.eval()
    iterator = build_progress(loader, total=min(len(loader), max_batches), desc="evolve stats", leave=False)
    for step, batch in enumerate(iterator, start=1):
        if step > max_batches:
            break
        batch = move_batch_to_device(batch, device, non_blocking=device.startswith("cuda"))

        with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=use_amp):
            _, _, _, aux = model(batch["inputs"], batch["input_lengths"], return_aux=True)

        if aux is not None:
            for block_entry in aux["block_aux"]:
                block_idx = int(block_entry["block_index"])
                routing = block_entry["routing"]
                if routing is None or block_idx not in block_stats:
                    continue
                block_stats[block_idx]["usage_sum"] += routing.detach().sum(dim=0).cpu()
                block_stats[block_idx]["usage_count"] += routing.size(0)

        for block_idx in block_stats:
            scores = compute_expert_scores(
                model,
                batch,
                ctc_loss,
                args,
                device,
                use_amp=False,
                block_idx=block_idx,
            )
            if scores is not None:
                block_stats[block_idx]["scores"].append(scores.detach().cpu())

    collected: list[dict[str, Any]] = []
    for block_idx, stats in block_stats.items():
        if not stats["scores"]:
            continue
        score_tensor = torch.cat(stats["scores"], dim=0)
        _, fitness, _ = competition_targets(score_tensor, args)
        usage_count = max(1, int(stats["usage_count"]))
        avg_usage = stats["usage_sum"] / usage_count
        collected.append(
            {
                "block_idx": block_idx,
                "module": stats["module"],
                "scores": score_tensor,
                "fitness": fitness,
                "avg_usage": avg_usage,
            }
        )
    return collected


@torch.no_grad()
def evolve_experts(
    model: EncoderMoECTCModel,
    loader,
    ctc_loss,
    args: argparse.Namespace,
    device: str,
    use_amp: bool = False,
) -> list[dict[str, Any]]:
    if args.ffn_type != "shared_adapter_moe" or args.expert_evolve_every_epochs <= 0:
        return []

    evolution_stats = collect_evolution_statistics(model, loader, ctc_loss, args, device, use_amp=use_amp)
    if not evolution_stats:
        return []

    logs: list[dict[str, Any]] = []
    for stats in evolution_stats:
        block_idx = int(stats["block_idx"])
        moe_module: SharedAdapterMoEFFN = stats["module"]
        scores: torch.Tensor = stats["scores"]
        fitness: torch.Tensor = stats["fitness"]
        avg_usage: torch.Tensor = stats["avg_usage"]

        parent_a, parent_b, parent_diag = select_expert_parents(scores, eps=args.competition_epsilon)
        replace_idx, replace_diag = select_replacement_expert(
            fitness,
            avg_usage,
            parent_a=parent_a,
            parent_b=parent_b,
            strategy=args.expert_merge_replace,
        )
        old_state = moe_module.get_expert_state(replace_idx)
        moe_module.merge_experts(
            parent_a,
            parent_b,
            child_idx=replace_idx,
            alpha=args.expert_merge_alpha,
            split_ratio=args.expert_merge_split_ratio,
            mode="split_linear",
        )
        moe_module._last_replaced_state = old_state  # type: ignore[attr-defined]

        logs.append(
            {
                "block": block_idx,
                "parent_a": parent_a,
                "parent_b": parent_b,
                "replace_idx": replace_idx,
                "fitness": [round(float(v), 6) for v in fitness.tolist()],
                "avg_usage": [round(float(v), 6) for v in avg_usage.tolist()],
                "parent_selection": parent_diag,
                "replacement": replace_diag,
            }
        )
    return logs


def append_vector_metrics(payload: dict[str, float], prefix: str, values: list[float]) -> None:
    for idx, value in enumerate(values):
        payload[f"{prefix}_{idx}"] = float(value)


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
        n_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0
        if n_gpus > 1:
            print(f"Using {n_gpus} GPUs with DataParallel.", flush=True)
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

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            **loader_kwargs,
        )
        test_loader = (
            torch.utils.data.DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                **loader_kwargs,
            )
            if test_dataset
            else None
        )

        model = EncoderMoECTCModel(args, vocab_size=len(tokenizer.id_to_token)).to(device)
        raw_model = model
        if n_gpus > 1:
            model = nn.DataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = build_lr_scheduler(optimizer, args, len(train_loader))
        ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True, reduction="mean")
        if scheduler is not None:
            print(
                f"Scheduler: {args.scheduler} warmup_steps="
                f"{int(args.warmup_steps) if args.warmup_steps > 0 else int(round(args.warmup_ratio * args.epochs * len(train_loader)))} "
                f"min_lr_scale={args.min_lr_scale}",
                flush=True,
            )

        best_valid_loss = float("inf")
        no_improve_rounds = 0
        eval_every = max(1, int(args.eval_every_epochs))
        patience = max(0, int(args.early_stop_patience))
        history: list[dict[str, Any]] = []

        for epoch in range(1, args.epochs + 1):
            print(f"Starting epoch {epoch}/{args.epochs}", flush=True)
            train_metrics = train_one_epoch(
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

            evolve_logs: list[dict[str, Any]] = []
            if should_run_expert_evolution(args, epoch):
                evolve_logs = evolve_experts(raw_model, valid_loader, ctc_loss, args, device, use_amp=use_amp)
                if evolve_logs:
                    save_json(output_dir / f"expert_evolution_epoch_{epoch}.json", {"events": evolve_logs})

            should_eval = (epoch % eval_every == 0) or (epoch == args.epochs)
            if not should_eval:
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": round(train_metrics["loss"], 6),
                        "train_base_loss": round(train_metrics["base_loss"], 6),
                        "train_ctc_loss": round(train_metrics["ctc_loss"], 6),
                        "train_load_balance_loss": round(train_metrics["load_balance_loss"], 6),
                        "train_competition_loss": round(train_metrics["competition_loss"], 6),
                        "train_routing_entropy": round(train_metrics["routing_entropy"], 6),
                        "train_grad_norm": round(train_metrics["grad_norm"], 6),
                        "train_lr": round(train_metrics["lr"], 8),
                        "train_effective_competition_weight": round(
                            train_metrics["effective_competition_weight"], 6
                        ),
                        "train_avg_gates": train_metrics.get("avg_gates", []),
                        "train_expert_usage": train_metrics.get("expert_usage", []),
                        "expert_evolution": evolve_logs,
                    }
                )
                train_log_payload = {
                    "global_step": epoch * len(train_loader),
                    "epoch": epoch,
                    "train/loss_epoch": train_metrics["loss"],
                    "train/base_loss_epoch": train_metrics["base_loss"],
                    "train/ctc_loss_epoch": train_metrics["ctc_loss"],
                    "train/load_balance_loss_epoch": train_metrics["load_balance_loss"],
                    "train/competition_loss_epoch": train_metrics["competition_loss"],
                    "train/routing_entropy": train_metrics["routing_entropy"],
                    "train/grad_norm_epoch": train_metrics["grad_norm"],
                    "train/lr": train_metrics["lr"],
                    "train/effective_competition_weight": train_metrics["effective_competition_weight"],
                }
                append_vector_metrics(train_log_payload, "train/avg_gates", train_metrics.get("avg_gates", []))
                append_vector_metrics(
                    train_log_payload,
                    "train/expert_usage",
                    train_metrics.get("expert_usage", train_metrics.get("avg_gates", [])),
                )
                log_wandb_metrics(wandb_run, train_log_payload)
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
                epoch=epoch,
            )
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_base_loss": round(train_metrics["base_loss"], 6),
                "train_ctc_loss": round(train_metrics["ctc_loss"], 6),
                "train_load_balance_loss": round(train_metrics["load_balance_loss"], 6),
                "train_competition_loss": round(train_metrics["competition_loss"], 6),
                "train_routing_entropy": round(train_metrics["routing_entropy"], 6),
                "train_grad_norm": round(train_metrics["grad_norm"], 6),
                "train_lr": round(train_metrics["lr"], 8),
                "train_effective_competition_weight": round(train_metrics["effective_competition_weight"], 6),
                "train_avg_gates": train_metrics.get("avg_gates", []),
                "train_expert_usage": train_metrics.get("expert_usage", []),
                "valid_loss": round(valid_metrics["loss"], 6),
                "valid_base_loss": round(valid_metrics["base_loss"], 6),
                "valid_ctc_loss": round(valid_metrics["ctc_loss"], 6),
                "valid_load_balance_loss": round(valid_metrics["load_balance_loss"], 6),
                "valid_competition_loss": round(valid_metrics["competition_loss"], 6),
                "valid_cer": round(valid_metrics["cer"], 6),
                "valid_wer": round(valid_metrics["wer"], 6),
                "valid_routing": valid_metrics["routing"],
                "valid_avg_gates": valid_metrics.get("avg_gates", []),
                "valid_expert_usage": valid_metrics.get("expert_usage", []),
                "valid_expert_fitness": valid_metrics.get("expert_fitness", []),
                "valid_routing_entropy": round(valid_metrics["routing_entropy"], 6),
                "valid_domain_loss": valid_metrics.get("domain_loss", {}),
                "valid_effective_competition_weight": round(
                    valid_metrics.get("effective_competition_weight", 0.0), 6
                ),
                "expert_evolution": evolve_logs,
            }
            history.append(epoch_metrics)
            print(
                f"epoch={epoch} valid_loss={valid_metrics['loss']:.4f} "
                f"valid_cer={valid_metrics['cer']:.4f} valid_wer={valid_metrics['wer']:.4f}",
                flush=True,
            )

            global_step = epoch * len(train_loader)
            is_best = valid_metrics["loss"] < best_valid_loss
            log_payload = {
                "global_step": global_step,
                "epoch": epoch,
                "train/loss_epoch": train_metrics["loss"],
                "train/base_loss_epoch": train_metrics["base_loss"],
                "train/ctc_loss_epoch": train_metrics["ctc_loss"],
                "train/load_balance_loss_epoch": train_metrics["load_balance_loss"],
                "train/competition_loss_epoch": train_metrics["competition_loss"],
                "train/routing_entropy": train_metrics["routing_entropy"],
                "train/grad_norm_epoch": train_metrics["grad_norm"],
                "train/lr": train_metrics["lr"],
                "train/effective_competition_weight": train_metrics["effective_competition_weight"],
                "valid/loss": valid_metrics["loss"],
                "valid/base_loss": valid_metrics["base_loss"],
                "valid/ctc_loss": valid_metrics["ctc_loss"],
                "valid/load_balance_loss": valid_metrics["load_balance_loss"],
                "valid/competition_loss": valid_metrics["competition_loss"],
                "valid/cer": valid_metrics["cer"],
                "valid/wer": valid_metrics["wer"],
                "valid/routing_entropy": valid_metrics["routing_entropy"],
                "valid/effective_competition_weight": valid_metrics.get("effective_competition_weight", 0.0),
                "valid/is_best": int(is_best),
                **flatten_routing_metrics("valid", valid_metrics["routing"]),
                **flatten_scalar_metrics("valid/domain_loss", valid_metrics.get("domain_loss", {})),
            }
            append_vector_metrics(log_payload, "train/avg_gates", train_metrics.get("avg_gates", []))
            append_vector_metrics(
                log_payload,
                "train/expert_usage",
                train_metrics.get("expert_usage", train_metrics.get("avg_gates", [])),
            )
            append_vector_metrics(log_payload, "valid/avg_gates", valid_metrics.get("avg_gates", []))
            append_vector_metrics(
                log_payload,
                "valid/expert_usage",
                valid_metrics.get("expert_usage", valid_metrics.get("avg_gates", [])),
            )
            append_vector_metrics(log_payload, "valid/expert_fitness", valid_metrics.get("expert_fitness", []))
            log_wandb_metrics(wandb_run, log_payload)

            if is_best:
                best_valid_loss = valid_metrics["loss"]
                no_improve_rounds = 0
                torch.save(
                    {
                        "model_state": raw_model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                        "config": vars(args),
                        "vocab": tokenizer.id_to_token,
                        "best_valid_loss": best_valid_loss,
                    },
                    output_dir / "best.pt",
                )
                save_json(output_dir / "best_valid_metrics.json", valid_metrics)
                log_wandb_metrics(
                    wandb_run,
                    {
                        "global_step": global_step,
                        "epoch": epoch,
                        "valid/best_loss": best_valid_loss,
                    },
                )
            else:
                no_improve_rounds += 1
                if patience > 0 and no_improve_rounds >= patience:
                    print(
                        f"Early stopping at epoch {epoch}: no validation loss improvement in "
                        f"{no_improve_rounds} eval rounds.",
                        flush=True,
                    )
                    break

        save_json(output_dir / "train_history.json", {"epochs": history})

        if test_loader is not None and (output_dir / "best.pt").exists():
            checkpoint = torch.load(output_dir / "best.pt", map_location=device)
            raw_model.load_state_dict(checkpoint["model_state"])
            test_metrics = evaluate(
                model=model,
                loader=test_loader,
                tokenizer=tokenizer,
                ctc_loss=ctc_loss,
                args=args,
                device=device,
                stage="test",
                use_amp=use_amp,
                epoch=history[-1]["epoch"] if history else args.epochs,
            )
            save_json(output_dir / "test_metrics.json", test_metrics)
            print(
                f"test_loss={test_metrics['loss']:.4f} test_cer={test_metrics['cer']:.4f} "
                f"test_wer={test_metrics['wer']:.4f}",
                flush=True,
            )
            test_log_payload = {
                "global_step": args.epochs * len(train_loader),
                "epoch": args.epochs,
                "test/loss": test_metrics["loss"],
                "test/base_loss": test_metrics["base_loss"],
                "test/ctc_loss": test_metrics["ctc_loss"],
                "test/load_balance_loss": test_metrics["load_balance_loss"],
                "test/competition_loss": test_metrics["competition_loss"],
                "test/cer": test_metrics["cer"],
                "test/wer": test_metrics["wer"],
                "test/routing_entropy": test_metrics["routing_entropy"],
                **flatten_routing_metrics("test", test_metrics["routing"]),
                **flatten_scalar_metrics("test/domain_loss", test_metrics.get("domain_loss", {})),
            }
            append_vector_metrics(test_log_payload, "test/avg_gates", test_metrics.get("avg_gates", []))
            append_vector_metrics(
                test_log_payload,
                "test/expert_usage",
                test_metrics.get("expert_usage", test_metrics.get("avg_gates", [])),
            )
            append_vector_metrics(test_log_payload, "test/expert_fitness", test_metrics.get("expert_fitness", []))
            log_wandb_metrics(wandb_run, test_log_payload)
    finally:
        finish_wandb_run(wandb_run)


if __name__ == "__main__":
    main()
