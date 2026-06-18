from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return self.dropout(x + self.embedding(positions))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.wi = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.wo = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.wi(x).chunk(2, dim=-1)
        return self.wo(self.dropout(value * F.silu(gate)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.self_attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = SwiGLUFeedForward(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attn_in = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(
            attn_in,
            attn_in,
            attn_in,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.residual_scale * self.dropout(attn_out)
        ffn_out = self.ffn(self.ffn_norm(x))
        return x + self.residual_scale * self.dropout(ffn_out)


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.self_attn_norm = RMSNorm(d_model)
        self.cross_attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = SwiGLUFeedForward(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = residual_scale

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attn_in = self.self_attn_norm(x)
        self_attn_out, _ = self.self_attn(
            self_attn_in,
            self_attn_in,
            self_attn_in,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + self.residual_scale * self.dropout(self_attn_out)

        cross_query = self.cross_attn_norm(x)
        cross_out, _ = self.cross_attn(
            cross_query,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.residual_scale * self.dropout(cross_out)
        ffn_out = self.ffn(self.ffn_norm(x))
        return x + self.residual_scale * self.dropout(ffn_out)


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        share_embeddings: bool = True,
        share_decoder_generator: bool = True,
        learned_positional: bool = True,
        activation: str = "swiglu",
    ) -> None:
        super().__init__()
        if activation != "swiglu":
            raise ValueError("This architecture uses SwiGLU feed-forward layers; set activation='swiglu'.")

        self.pad_id = pad_id
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = self.src_embed if share_embeddings else nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        if not learned_positional:
            raise ValueError("This architecture expects learned positional embeddings.")
        self.positional = LearnedPositionalEmbedding(d_model, max_len=max_len, dropout=dropout)

        total_layers = num_encoder_layers + num_decoder_layers
        residual_scale = (2 * total_layers) ** -0.25
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(d_model, nhead, dim_feedforward, dropout, residual_scale)
                for _ in range(num_encoder_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(d_model, nhead, dim_feedforward, dropout, residual_scale)
                for _ in range(num_decoder_layers)
            ]
        )
        self.encoder_norm = RMSNorm(d_model)
        self.decoder_norm = RMSNorm(d_model)
        self.generator = nn.Linear(d_model, vocab_size, bias=False)
        if share_decoder_generator:
            self.generator.weight = self.tgt_embed.weight
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.d_model**-0.5)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        src_key_padding_mask = src.eq(self.pad_id)
        tgt_key_padding_mask = tgt_in.eq(self.pad_id)
        tgt_mask = causal_mask(tgt_in.size(1), tgt_in.device)
        memory = self._encode(src, src_key_padding_mask)
        decoded = self._decode(tgt_in, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
        return self.generator(decoded)

    @torch.no_grad()
    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask = src.eq(self.pad_id)
        return self._encode(src, src_key_padding_mask), src_key_padding_mask

    @torch.no_grad()
    def decode_step(
        self,
        ys: torch.Tensor,
        memory: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_mask = causal_mask(ys.size(1), ys.device)
        tgt_key_padding_mask = ys.eq(self.pad_id)
        out = self._decode(ys, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
        return self.generator(out[:, -1])

    def _encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self.positional(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask)
        return self.encoder_norm(x)

    def _decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.positional(self.tgt_embed(tgt_in) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)
        return self.decoder_norm(x)


def causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
