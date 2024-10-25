import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter

class LinformerAttention(nn.Module):
    """Multi-headed linformer attention.

    Projects the key and values down to the compressed dimension, before computing self-attention.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        compressed=1,
        max_seq_len=256,
        shared_kv_compressed=0,
        shared_compress_layer=None,
        freeze_compress=0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.kdim == self.embed_dim and self.vdim == self.embed_dim, (
            "Self-attention requires query, key and value to be of the same size"
        )

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Used for compress sequence to subsequence
        if shared_compress_layer is None:
            self.compress_seq_len = max_seq_len // compressed
            self.compress_k = nn.Linear(max_seq_len, self.compress_seq_len, bias=False)
            self.compress_v = nn.Linear(max_seq_len, self.compress_seq_len, bias=False) if shared_kv_compressed == 0 else None
            self.layerwise_sharing = False
        else:
            self.compress_k = shared_compress_layer
            self.compress_v = shared_compress_layer if shared_kv_compressed == 0 else None
            self.layerwise_sharing = True

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        if freeze_compress == 1:
            self.compress_k.weight.requires_grad = False
            if shared_kv_compressed == 0:
                self.compress_v.weight.requires_grad = False

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        need_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel"""
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim

        if self.self_attention:
            q = self.q_proj(query)
            k_input = query.permute(1, 2, 0).contiguous()
            k_input = F.linear(k_input, self.compress_k.weight[:, :tgt_len]).permute(2, 0, 1).contiguous()
            k = self.k_proj(k_input)
            v_input = query.permute(1, 2, 0).contiguous()
            v_input = F.linear(v_input, self.compress_v.weight[:, :tgt_len]).permute(2, 0, 1).contiguous() if self.compress_v else k_input
            v = self.v_proj(v_input)
        elif self.encoder_decoder_attention:
            q = self.q_proj(query)
            k = self.k_proj(key) if key is not None else None
            v = self.v_proj(key) if key is not None else None
        else:
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q *= self.scaling

        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)], dim=1)
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)], dim=1)

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2)) if k is not None else None
        if attn_weights is not None:
            attn_weights = F.softmax(attn_weights, dim=-1)

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training) if attn_weights is not None else None
        attn = torch.bmm(attn_probs, v) if attn_probs is not None else None

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim) if attn is not None else None
        attn = self.out_proj(attn) if attn is not None else None

        final_attn_weights: Optional[Tensor] = None
        if need_weights and attn_weights is not None:
            final_attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, -1).mean(dim=1)

        return attn, final_attn_weights

# Positionwise Feed Forward Class
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = LinformerAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)  # Use Linformer here
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        attn_output, _ = self.self_attn(x, x, x, attn_mask=src_mask)  # Include mask if needed
        x = self.norm1(x + attn_output)

        ff_output = self.linear2(self.dropout1(F.relu(self.linear1(x))))
        x = self.norm2(x + ff_output)

        return x



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = LinformerAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)  # Self-attention
        self.cross_attn = LinformerAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)  # Cross-attention
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, memory: Tensor, tgt_mask: Tensor, memory_mask: Tensor) -> Tensor:
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=tgt_mask)
        x = self.norm1(x + attn_output)

        # Cross-attention
        cross_output, _ = self.cross_attn(memory, memory, x, attn_mask=memory_mask)
        x = self.norm2(x + cross_output)

        # Feed-forward network
        ff_output = self.linear2(self.dropout1(F.relu(self.linear1(x))))
        x = self.norm3(x + ff_output)

        return x
