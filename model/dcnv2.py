"""Summary
"""
import torch
import torch.nn as nn
import lightning as L

class FeatureEmbeddingLayer(nn.Module):

    """
    Implementation of Sparse Feature Embedding Layer of DCN-V2

    Attributes:
        sparse_embeds (nn.ModuleList): List of embbeding layers for each sparse feature
    """

    def __init__(self, sparse_vocab_sizes:list[int], embed_dims:list[int]):
        """
        Initializes Embedding Layer of DCN-V2

        Args:
            sparse_vocab_sizes (list[int]): list of vocabulary sizes for sparse features
            embed_dims (list[int]): list of embedding vector sizes for each sparse feature
        """
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(sparse_vocab_size, embed_dim) for sparse_vocab_size, embed_dim in zip(sparse_vocab_sizes, embed_dims)
        ])

    def forward(self, dense_x:torch.Tensor, sparse_x:torch.LongTensor):
        """
        Args:
            dense_x (torch.Tensor): (B, num_dense_feat)
            sparse_x (torch.LongTensor): (B, num_sparse_feat)

        Returns:
            torch.Tensor: concatenation of embedding vector of sparse vectors and dense feature vector
        """
        vectors =[]

        for i, embedding_layer in enumerate(self.sparse_embeds):
            vectors.append(embedding_layer(sparse_x[:, i]))

        vectors.append(dense_x)
        return torch.concat(vectors, axis=1)

class CrossLayer(nn.Module):

    """
    Implementation of Cross Layer of DCN-V2 to learn polynomial approximation of feature crosses.

    Attributes:
        bias (nn.Parameter): Learnable Bias Vector fo Cross Layer
        linear (nn.Linear): Weight Matrix for Cross Layer
    """

    def __init__(self, input_dim, initial_bias=None:torch.Tensor):
        """
        Args:
            input_dim (int): size of the input vector after embedding
            initial_bias (torch.Tensor, optional): Optional bias initialization
        """
        self.linear = nn.Linear(input_dim, input_dim, bias=False)

        if initial_bias is None:
            self.bias = nn.Parameter(torch.zeros(input_dim)) # as in section 7.1:optimization of the original paper
        else:
            self.bias = nn.Parameter(initial_bias)

    def forward(self, x_l, x_0):
        """
        Args:
            x_l (torch.Tensor): output of previous cross layer
            x_0 (torch.Tensor): initial input vector

        Returns:
            torch.Tensor: output vector
        """
        return (x_0 * self.linear(x_l) + self.bias) + x_l

class DCNV2(L.LightningModule):

    """
    TODO
    """
    pass

