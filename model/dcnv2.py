"""
Implementation of DCN-V2 Recommender System Model appeared in WWW'2021
"""

import torch
import torch.nn as nn
import lightning as L
from functools import reduce

class FeatureEmbeddingLayer(nn.Module):

    """
    Implementation of Sparse Feature Embedding Layer of DCN-V2

    One should map all unknown categorical value or null value to the length of the known vocabulary size to handle new or null values.
    
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
        super().__init__()
        self.sparse_embeds = nn.ModuleList([
            nn.Embedding(sparse_vocab_size + 1, embed_dim, padding_idx=sparse_vocab_size) for sparse_vocab_size, embed_dim in zip(sparse_vocab_sizes, embed_dims)
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

    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): size of the input vector after embedding
            initial_bias (torch.Tensor, optional): Optional bias initialization
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim)

        with torch.no_grad():
            self.linear.bias.fill_(0.0)

    def forward(self, x_l, x_0):
        """
        Args:
            x_l (torch.Tensor): output of previous cross layer
            x_0 (torch.Tensor): initial input vector

        Returns:
            torch.Tensor: output vector
        """
        return x_0 * self.linear(x_l) + x_l

class DCNV2(L.LightningModule):

    """
    LightningModule class for DCN-V2 model [Wang et al. 2020]
    """

    def __init__(self, args):
        """
        Builds DCN V2 model based on given metadata and model hyperparameters.
        Args:
            stacked (bool): Indicator to use stacked structure. If `False`, it uses parallel structure.
            num_cross_layer (int): number of cross layers to use. If `stacked` is `False`, it must equal to num_hidden_layer
            num_deep_layer (int): number of deep layers to use.
            num_dense_feat (int): number of dense features of input
            sparse_vocab_sizes (list[int]): list of vocabulary sizes of sparse features.
            sparse_embed_dims (list[int]): list of embed vector sizes of sparse features. If None, dimensions are set as an average of :math: `6 \cdot \|V\|^{\frac{1}{4}}` over all sparse features.
            deep_out_dim (int): size of the output tensor of the deep layer before calculating logit
            lr (float): initial learning rate
            
        """
        super().__init__()

        # Model Parameters
        self.stacked = args.stacked
        self.num_cross_layer = args.num_cross_layer
        self.num_deep_layer = args.num_deep_layer
        self.num_dense_feat = args.num_dense_feat
        self.sparse_vocab_sizes = args.sparse_vocab_sizes
        self.deep_out_dim = args.deep_out_dim

        # Optimization Parameters
        self.lr = args.lr
        self.gradient_clip = args.gradient_clip
        self.beta1 = args.beta1
        self.weight_decay = args.weight_decay
        self.lr_reduce_factor = args.lr_reduce_factor
        self.patience = args.patience


        # Build Embedding Layer
        if 'sparse_embed_dims' not in args or args.sparse_embed_dims is None: # Default option described in paper
            default_embed_dim = int(sum([x ** 0.25 for x in self.sparse_vocab_sizes]) * 6 / len(self.sparse_vocab_sizes))
            self.sparse_embed_dims = [default_embed_dim for _ in range(len(self.sparse_vocab_sizes))]
        else:
            self.sparse_embed_dims = args.sparse_embed_dims

        self.embedding_layer = FeatureEmbeddingLayer(self.sparse_vocab_sizes, self.sparse_embed_dims)

        # Build Cross Layer
        input_dim = self.num_dense_feat + sum(self.sparse_embed_dims)
        self.cross_layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(self.num_cross_layer)
        ])

        # Build Deep Layer
        self.deep_layers = nn.ModuleList()

        dim = input_dim
        cnt = 1
        if input_dim <= self.deep_out_dim: # increase dim size with order of 2
            while dim * 2 < self.deep_out_dim and cnt < self.num_deep_layer:
                layer = nn.Linear(dim, dim * 2)
                with torch.no_grad():
                    layer.bias.fill_(0.0)
                self.deep_layers.append(layer)
                self.deep_layers.append(nn.ReLU())
                dim *= 2
                cnt += 1
        else: # decrease dim size with order of 2
            while dim // 2 > self.deep_out_dim and cnt < self.num_deep_layer:
                layer = nn.Linear(dim, dim // 2)
                with torch.no_grad():
                    layer.bias.fill_(0.0)
                self.deep_layers.append(nn.Linear(dim, dim // 2))
                self.deep_layers.append(nn.ReLU())
                dim = dim // 2
                cnt += 1
        
        while cnt < self.num_deep_layer: # build remaining deep layers
            self.deep_layers.append(nn.Linear(dim, dim))
            self.deep_layers.append(nn.ReLU())
            cnt += 1

        self.deep_layers.append(nn.Linear(dim, self.deep_out_dim)) # Final output FC layer

        # Initialize logit vector
        if self.stacked:
            vector = torch.empty(self.deep_out_dim)
        else:
            vector = torch.empty(input_dim + self.deep_out_dim)

        nn.init.normal_(vector)
        self.logit_vector = nn.Parameter(vector)

    def forward(self, dense_x, sparse_x):
        x_0 = self.embedding_layer(dense_x, sparse_x)

        if self.stacked:
            x_c = reduce(lambda result, layer: layer(result, x_0), self.cross_layers, x_0)
            x_d = torch.nn.functional.relu(x_c)
            x_d = reduce(lambda result, layer : layer(result), self.deep_layers, x_d)

            return torch.nn.functional.sigmoid(torch.matmul(x_d, self.logit_vector))
        else:
            x_c = reduce(lambda result, layer: layer(result, x_0), self.cross_layers, x_0)
            x_d = reduce(lambda result, layer: layer(result), self.deep_layers, x_0)
            x_final = torch.concat([x_c, x_d], axis=1)

            return torch.nn.functional.sigmoid(torch.matmul(x_final, self.logit_vector))

    def training_step(self, batch, batch_idx):
        dense_x, sparse_x, y = batch
        y_hat = self.forward(dense_x, sparse_x).unsqueeze(-1)
        return torch.nn.functional.nll_loss(y_hat, y)

    def validation_step(self, batch, batch_idx):
        dense_x, sparse_x, y = batch
        y_hat = self.forward(dense_x, sparse_x).unsqueeze(-1)
        self.log("val_loss", torch.nn.functional.nll_loss(y_hat, y))

    def test_step(self, batch, batch_idx):
        dense_x, sparse_x, y = batch
        y_hat = self.forward(dense_x, sparse_x).unsqueeze(-1)
        self.log("test_loss", torch.nn.functional.nll_loss(y_hat, y))

    def predict_step(self, batch):
        dense_x, sparse_x = batch
        return self.forward(dense_x, sparse_x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, beta=(self.beta1, 0.999), weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.lr_reduce_factor, patience=self.patience, cooldown=2)
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

def module_test(model_args, batch_size=24):
    """
    Check model structure given model arguments and input batch size

    Args:
        model_args (Namespace): Hyperparameters for model structures and optimization process
        batch_size (int, optional): batch size of the input
    """
    from torchinfo import summary

    model = DCNV2(model_args)

    dense_x = torch.randn(batch_size, model_args.num_dense_feat)
    sparse_x = torch.randint(0, min(model_args.sparse_vocab_sizes), size=(batch_size, model_args.num_sparse_feat), dtype=torch.long)

    summary(model, input_data=(dense_x, sparse_x), verbose=2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='DCN Model Tester', description='Simple sanity check for model structure')

    # Model Args
    parser.add_argument('--stacked', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--num_cross_layer', type=int, default=4)
    parser.add_argument('--num_deep_layer', type=int, default=4)
    parser.add_argument('--num_dense_feat', type=int, default=10)
    parser.add_argument('--num_sparse_feat', type=int, default=20)
    parser.add_argument('--sparse_vocab_sizes', nargs='*', required=True)
    parser.add_argument('--sparse_embed_dims', nargs='*')
    parser.add_argument('--deep_out_dim', type=int, default=562)

    # Optimizer Args
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gradient_clip', type=float, default=10.0)
    parser.add_argument('--beta1', type=float, default=0.9999)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)

    args = parser.parse_args()

    # Argument Check
    if not (args.stacked ^ args.parallel):
        raise argparse.ArgumentTypeError('Exactly one of "--stacked" or "--parallel" must be given')
    elif args.stacked:
        args.stacked = True
    else:
        args.stacked = False

    args.sparse_vocab_sizes = [int(x) for x in args.sparse_vocab_sizes]
    assert args.num_sparse_feat == len(args.sparse_vocab_sizes)

    module_test(args)
