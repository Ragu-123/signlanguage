# coding: utf-8

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from helpers import freeze_params
from transformer_layers import TransformerEncoderLayer, PositionalEncoding
from embeddings import MaskedNorm, Embeddings

# pylint: disable=abstract-method
class Encoder(nn.Module):
    """
    Base encoder class
    """
    @property
    def output_size(self):
        """
        Return the output size

        :return:
        """
        return self._output_size


class TransformerEncoder(Encoder):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim: int = 64,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 **kwargs):
        """
        Initializes the Transformer.
        
        :param embedding_dim: dimension of the embedding space
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size (Typically this is 2*hidden_size)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings)
        :param vocab_size: size of the vocabulary for embeddings
        :param padding_idx: padding index for embeddings
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()

        # Initialize the embeddings layer
        self.embeddings = Embeddings(embedding_dim=embedding_dim,
                                     vocab_size=vocab_size,
                                     padding_idx=padding_idx)

        # build all (num_layers) layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(size=hidden_size, ff_size=ff_size,
                                    num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)])

        # Use MaskedNorm instead of LayerNorm
        self.layer_norm = MaskedNorm(hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(hidden_size)
        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self._output_size = hidden_size

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self,
                input_ids: Tensor,
                src_length: Tensor,
                mask: Tensor) -> (Tensor, Tensor):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param input_ids: token indices for the input,
            shape (batch_size, src_len)
        :param src_length: length of src inputs (counting tokens before padding),
            shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len)
        :return:
            - output: hidden states with shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with shape (batch_size, directions*hidden)
        """

        # Create embeddings from input_ids
        embed_src = self.embeddings(input_ids)

        x = embed_src

        # Add position encoding to word embeddings
        x = self.pe(x)
        # Add Dropout
        x = self.emb_dropout(x)

        # Apply each layer to the input
        for layer in self.layers:
            x = layer(x, mask)

        return self.layer_norm(x, mask), None  # Pass mask to MaskedNorm

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].src_src_att.num_heads)
