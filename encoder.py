# coding: utf-8

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from helpers import freeze_params
from transformer_layers import TransformerEncoderLayer, PositionalEncoding

# Base encoder class
class Encoder(nn.Module):
    """
    Base encoder class
    """
    @property
    def output_size(self):
        """
        Return the output size
        """
        return self._output_size

class TransformerEncoder(Encoder):
    """
    Transformer Encoder using Linformer Attention
    """

    def __init__(self,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Transformer with Linformer attention layers.
        :param hidden_size: Hidden size and size of embeddings
        :param ff_size: Position-wise feed-forward layer size (usually 2*hidden_size)
        :param num_layers: Number of layers in the transformer
        :param num_heads: Number of heads in multi-headed attention
        :param dropout: Dropout probability for transformer layers
        :param emb_dropout: Dropout for input embeddings
        :param freeze: Freeze encoder parameters during training
        """
        super(TransformerEncoder, self).__init__()

        # Define the transformer layers using Linformer-based TransformerEncoderLayer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model=hidden_size, d_ff=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Layer normalization and positional encoding
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size, dropout=emb_dropout)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self._output_size = hidden_size

        # Freeze parameters if required
        if freeze:
            freeze_params(self)

    def forward(self,
                embed_src: Tensor,
                src_length: Tensor,
                mask: Optional[Tensor] = None) -> (Tensor, Tensor):
        """
        Pass the input through each layer in sequence with padded sequences support.
        :param embed_src: Embedded source inputs (batch_size, src_len, embed_size)
        :param src_length: Source input lengths (batch_size)
        :param mask: Padding mask (batch_size, src_len, embed_size)
        :return:
            - output: Hidden states (batch_size, max_length, hidden)
        """
        
        # Pack the input sequence
        packed_input = pack_padded_sequence(embed_src, src_length.cpu(), batch_first=True, enforce_sorted=False)

        # Add positional encoding and dropout
        x = self.pe(packed_input.data)
        x = self.emb_dropout(x)

        # Apply each transformer layer
        for layer in self.layers:
            x = layer(x, mask)
        
        # Pad the sequence back to its original shape
        padded_output, _ = pad_packed_sequence(x, batch_first=True)

        # Final layer normalization
        return self.layer_norm(padded_output), None

    def __repr__(self):
        return f"{self.__class__.__name__}(num_layers={len(self.layers)}, num_heads={self.layers[0].self_attn.num_heads})"
