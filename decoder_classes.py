    class MultiHeadAttention(nn.Module):
        """Standard multi-head attention (used for both self-attn and cross-attn in decoder)."""

        def __init__(self, model_dim: int, num_heads: int, dropout: float, causal: bool = False):
            super().__init__()
            assert model_dim % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = model_dim // num_heads
            self.model_dim = model_dim
            self.scale = self.head_dim ** -0.5
            self.causal = causal

            self.w_q = nn.Linear(model_dim, model_dim)
            self.w_k = nn.Linear(model_dim, model_dim)
            self.w_v = nn.Linear(model_dim, model_dim)
            self.w_out = nn.Linear(model_dim, model_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor | None = None,
            value: torch.Tensor | None = None,
            key_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """
            Args:
                query: (B, T_q, D)
                key, value: (B, T_k, D). If None, uses query (self-attention).
                key_padding_mask: (B, T_k) True = position to ignore
            """
            if key is None:
                key = query
            if value is None:
                value = query

            B, T_q, _ = query.shape
            T_k = key.size(1)

            q = self.w_q(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.w_k(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.w_v(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            if self.causal and T_q == T_k:
                causal_mask = torch.triu(torch.ones(T_q, T_k, device=query.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            output = torch.matmul(attn, v)
            output = output.transpose(1, 2).contiguous().view(B, T_q, self.model_dim)
            return self.w_out(output)


    class TransformerDecoderLayer(nn.Module):
        def __init__(self, args: argparse.Namespace):
            super().__init__()
            self.self_attn_norm = nn.LayerNorm(args.decoder_dim)
            self.self_attn = MultiHeadAttention(
                model_dim=args.decoder_dim,
                num_heads=args.decoder_heads,
                dropout=args.dropout,
                causal=True,
            )
            self.self_attn_dropout = nn.Dropout(args.dropout)

            self.cross_attn_norm = nn.LayerNorm(args.decoder_dim)
            self.cross_attn = MultiHeadAttention(
                model_dim=args.decoder_dim,
                num_heads=args.decoder_heads,
                dropout=args.dropout,
                causal=False,
            )
            self.cross_attn_dropout = nn.Dropout(args.dropout)

            self.ffn_norm = nn.LayerNorm(args.decoder_dim)
            if getattr(args, "decoder_moe", False):
                self.ffn = SharedAdapterMoEFFN(
                    model_dim=args.decoder_dim,
                    hidden_dim=args.decoder_ffn_hidden_dim,
                    adapter_hidden_dim=args.decoder_adapter_hidden_dim,
                    num_experts=args.num_experts,
                    temperature=args.router_temperature,
                    dropout=args.dropout,
                )
            else:
                self.ffn = DenseFFN(args.decoder_dim, args.decoder_ffn_hidden_dim, args.dropout)

        def forward(
            self,
            dec_states: torch.Tensor,
            enc_out: torch.Tensor,
            dec_padding_mask: torch.Tensor | None = None,
            enc_padding_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            # Masked self-attention
            residual = dec_states
            dec_states = self.self_attn_norm(dec_states)
            dec_states = self.self_attn(dec_states, key_padding_mask=dec_padding_mask)
            dec_states = residual + self.self_attn_dropout(dec_states)

            # Cross-attention to encoder
            residual = dec_states
            dec_states = self.cross_attn_norm(dec_states)
            dec_states = self.cross_attn(dec_states, key=enc_out, value=enc_out, key_padding_mask=enc_padding_mask)
            dec_states = residual + self.cross_attn_dropout(dec_states)

            # FFN
            residual = dec_states
            dec_states = self.ffn_norm(dec_states)
            dec_states, _, _ = self.ffn(dec_states, mask=torch.ones(dec_states.size(0), dec_states.size(1), device=dec_states.device))
            dec_states = residual + self.self_attn_dropout(dec_states)

            return dec_states


    class HybridCTCModel(nn.Module):
        def __init__(self, args: argparse.Namespace, vocab_size: int):
            super().__init__()
            self.num_experts = int(args.num_experts)
            self.gradient_checkpoint = bool(getattr(args, "gradient_checkpoint", False))
            self.ffn_type = args.ffn_type
            self.use_decoder = getattr(args, "decoder_layers", 0) > 0

            # Encoder (giu nguyen)
            self.subsampling = Conv2dSubsampling(args.n_mels, args.encoder_dim, args.dropout)
            self.position = RelativePositionalEncoding(args.encoder_dim, dropout=args.dropout)
            block_cls = TransformerMoEBlock if args.encoder_type == "transformer" else ConformerMoEBlock
            self.blocks = nn.ModuleList([block_cls(args) for _ in range(args.encoder_layers)])
            self.output_norm = nn.LayerNorm(args.encoder_dim)

            # CTC projection & head
            self.projector = nn.Sequential(
                nn.Linear(args.encoder_dim, args.projector_dim),
                nn.GELU(),
                nn.Dropout(args.dropout),
            )
            self.ctc_head = nn.Linear(args.projector_dim, vocab_size)

            # Intermediate CTC (giu nguyen)
            inter_weight = float(getattr(args, "intermediate_ctc_weight", 0.0))
            inter_layer = int(getattr(args, "intermediate_ctc_layer", 0))
            if inter_layer <= 0:
                inter_layer = max(1, args.encoder_layers // 2)
            self._inter_ctc_layer = inter_layer if inter_weight > 0 else -1
            if self._inter_ctc_layer >= 0:
                self.inter_norm = nn.LayerNorm(args.encoder_dim)
                self.inter_proj = nn.Sequential(
                    nn.Linear(args.encoder_dim, args.projector_dim),
                    nn.GELU(),
                    nn.Dropout(args.dropout),
                )
                self.inter_ctc_head = nn.Linear(args.projector_dim, vocab_size)

            # Attention Decoder
            if self.use_decoder:
                self.text_embed = nn.Embedding(vocab_size, args.decoder_dim)
                self.dec_pos = SinusoidalPositionalEncoding(args.decoder_dim, max_len=500)
                self.decoder_layers = nn.ModuleList([
                    TransformerDecoderLayer(args) for _ in range(args.decoder_layers)
                ])
                self.dec_norm = nn.LayerNorm(args.decoder_dim)
                self.dec_head = nn.Linear(args.decoder_dim, vocab_size)

        def forward(
            self,
            inputs: torch.Tensor,
            input_lengths: torch.Tensor,
            targets: torch.Tensor | None = None,
            target_lengths: torch.Tensor | None = None,
            forced_expert: int | None = None,
            forced_experts: dict[int, int] | None = None,
            return_aux: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any] | None, torch.Tensor | None, torch.Tensor | None]:
            # Encoder forward
            hidden_states, output_lengths = self.subsampling(inputs, input_lengths.to(inputs.device))
            hidden_states, pos_enc = self.position(hidden_states)
            enc_mask = lengths_to_mask(output_lengths.to(hidden_states.device), hidden_states.size(1))

            routing_values: list[torch.Tensor] = []
            block_aux: list[dict[str, Any]] = []
            intermediate_log_probs: torch.Tensor | None = None

            for block_idx, block in enumerate(self.blocks):
                block_forced_expert = forced_expert
                if forced_experts is not None:
                    block_forced_expert = forced_experts.get(block_idx)
                if self.gradient_checkpoint and self.training and block_forced_expert is None and not return_aux:
                    hidden_states, routing, aux = torch.utils.checkpoint.checkpoint(
                        block, hidden_states, enc_mask, pos_enc, block_forced_expert, return_aux,
                        use_reentrant=False,
                    )
                else:
                    hidden_states, routing, aux = block(
                        hidden_states, enc_mask, pos_enc,
                        forced_expert=block_forced_expert,
                        return_all_experts=return_aux,
                    )
                if routing is not None:
                    routing_values.append(routing)
                if return_aux:
                    block_aux.append({"block_index": block_idx, "routing": routing, "aux": aux})
                if self._inter_ctc_layer >= 0 and block_idx == self._inter_ctc_layer and self.training:
                    inter_hidden = self.inter_norm(hidden_states)
                    inter_proj = self.inter_proj(inter_hidden)
                    intermediate_log_probs = F.log_softmax(self.inter_ctc_head(inter_proj), dim=-1)

            enc_out = self.output_norm(hidden_states)

            # CTC branch
            ctc_hidden = self.projector(enc_out)
            ctc_logits = self.ctc_head(ctc_hidden)
            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)

            merged_routing = torch.stack(routing_values, dim=0).mean(dim=0) if routing_values else None
            aux_out = None
            if return_aux:
                aux_out = {"block_aux": block_aux, "mask": enc_mask, "output_lengths": output_lengths}

            # Attention decoder branch
            dec_log_probs = None
            if self.use_decoder and targets is not None and target_lengths is not None:
                dec_out = self._decode(targets, target_lengths, enc_out, enc_mask)
                dec_logits = self.dec_head(dec_out)
                dec_log_probs = F.log_softmax(dec_logits, dim=-1)

            return ctc_log_probs, output_lengths, merged_routing, aux_out, intermediate_log_probs, dec_log_probs

        def _decode(self, targets: torch.Tensor, target_lengths: torch.Tensor, enc_out: torch.Tensor, enc_mask: torch.Tensor) -> torch.Tensor:
            """Autoregressive decoding for training (teacher forcing)."""
            # Shift targets right: <sos> + targets[:-1]
            sos_token = 0  # Assuming 0 is <blank> or <sos>
            shifted = torch.cat([
                torch.full((targets.size(0), 1), sos_token, device=targets.device, dtype=targets.dtype),
                targets[:, :-1]
            ], dim=1)

            # Embed and add positional encoding
            dec_states = self.text_embed(shifted)
            dec_states = self.dec_pos(dec_states)

            # Create decoder padding mask
            dec_mask = lengths_to_mask(target_lengths, shifted.size(1)).to(shifted.device)
            dec_padding_mask = ~dec_mask.bool()
            enc_padding_mask = ~enc_mask.bool()

            # Pass through decoder layers
            for layer in self.decoder_layers:
                dec_states = layer(dec_states, enc_out, dec_padding_mask=dec_padding_mask, enc_padding_mask=enc_padding_mask)

            dec_states = self.dec_norm(dec_states)
            return dec_states

        def get_moe_modules(self) -> list[SharedAdapterMoEFFN]:
            modules: list[SharedAdapterMoEFFN] = []
            for block in self.blocks:
                ffn = getattr(block, "ffn", None)
                if isinstance(ffn, SharedAdapterMoEFFN):
                    modules.append(ffn)
            if self.use_decoder:
                for layer in self.decoder_layers:
                    ffn = getattr(layer, "ffn", None)
                    if isinstance(ffn, SharedAdapterMoEFFN):
                        modules.append(ffn)
            return modules
