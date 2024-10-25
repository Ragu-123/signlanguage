import math
import torch
from torch import nn, Tensor
from helpers import freeze_params

class MaskedNorm(nn.Module):
    """
    Custom normalization layer that handles different normalization types.
    It applies normalization while considering the input mask.
    Original Code from:
    https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type: str, num_groups: int, num_features: int):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])

class Embeddings(nn.Module):
    """
    Simple embeddings class for input representation.
    """

    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim: Size of each embedding vector.
        :param scale: If True, scale embeddings by sqrt(embedding_dim).
        :param vocab_size: Number of unique tokens in the vocabulary.
        :param padding_idx: Index used for padding.
        :param freeze: If True, freeze the embeddings during training.
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim, padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: Tensor of indices in the vocabulary.
        :return: Embedded representation for `x`.
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim}, vocab_size={self.vocab_size})"
