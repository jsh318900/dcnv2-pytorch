
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

    def __init__(self, **kwargs):
        """
        Builds DCN V2 model based on given metadata and model hyperparameters.
        Args:
            stacked (bool): Indicator to use stacked structure. If `False`, it uses parallel structure.
            num_cross_layer (int): number of cross layers to use. If `stacked` is `False`, it must equal to num_hidden_layer
            num_deep_layer (int): number of deep layers to use. If `stacked` is `False`, it is ignored
            num_dense_feat (int): number of dense features of input
            sparse_vocab_sizes (list[int]): list of vocabulary sizes of sparse features.
            sparse_embed_dims (list[int]): list of embed vector sizes of sparse features. If None, dimensions are set as an average of :math: `6 \cdot \|V\|^{\frac{1}{4}}` over all sparse features.
            deep_out_dim (int): size of the output tensor of the deep layer before calculating logit
        """
        
        self.stacked = kwargs['stacked']
        self.num_cross_layer = kwargs['num_cross_layer']
        self.num_deep_layer = self.num_cross_layer if not self.stacked else kwargs['num_deep_layer']
        self.num_dense_feat = kwargs['num_dense_feat']
        self.sparse_vocab_sizes = kwargs['sprase_vocab_sizes']
        self.deep_out_dim = kwargs['deep_out_dim']

        if 'sparse_embed_dims' not in kwargs or kwargs['sparse_embed_dims'] is None:
            default_embed_dim = int([x ** 0.25 for x in self.sparse_vocab_sizes].sum() * 6 / len(self.sparse_vocab_sizes))
            default_embed_dim = max(*self.sparse_vocab_sizes, default_embed_dim)
            self.sparse_embed_dims = [default_embed_dim for _ in range(len(self.sparse_vocab_sizes))]
        else:
            self.sparse_embed_dims = kwargs['spares_embed_dims']
            assert len(self.sparse_vocab_sizes) == len(self.sparse_embed_dims)

    def _build_sparse_model(self):
        pass

    def _build_parallel_model(self):
        pass

    def forward(self, batch_x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

