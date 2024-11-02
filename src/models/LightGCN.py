import numpy as np
import torch
from scipy.sparse import csr_matrix, vstack, hstack, diags
from ._helpers import MLP_Base

class LightGCN(torch.nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.user_cnt = data['field_dims'][0]
        self.item_cnt = data['field_dims'][1]
        self.adj_matrix = data['adj_matrix']
        
        self.user_embedding = torch.nn.Embedding(
            self.user_cnt, args.embed_dim
        )
        self.item_embedding = torch.nn.Embedding(
            self.item_cnt, args.embed_dim
        )
        self.aggregator = self.create_aggregator()
        self.num_layers = args.num_layers

        if args.interaction == 'neural':
            self.mlp = MLP_Base(args.embed_dim * 2, args.mlp_dims, output_layer=True)
            self.interaction = lambda x, y: self.mlp(torch.cat([x, y], dim=1)).squeeze(1)
        elif args.interaction == 'dot':
            self.interaction = lambda x, y: torch.sum(x * y, dim=1)
    
    def get_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        full_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embeddings.append(full_embedding)
        for _ in range(self.num_layers):
            full_embedding = torch.sparse.mm(self.aggregator, full_embedding)
            embeddings.append(full_embedding)
        final_embedding = torch.stack(embeddings, dim=0).mean(dim=0)
        final_user_embedding, final_item_embedding = torch.split(
            final_embedding, [self.user_cnt, self.item_cnt]
        )
        return final_user_embedding, final_item_embedding     

    def forward(self, x):
        user_ids, item_ids = x[:, 0], x[:, 1]
        final_user_embedding, final_item_embedding = self.get_embeddings()
        output = self.interaction(final_user_embedding[user_ids], final_item_embedding[item_ids])
        return output
    
    def create_aggregator(self):
        def get_extended_adj_matrix(adj_matrix: csr_matrix) -> csr_matrix:
            upper_left_zeros = csr_matrix((self.user_cnt, self.user_cnt))
            upper_part = hstack([upper_left_zeros, adj_matrix])
            lower_right_zeros = csr_matrix((self.item_cnt, self.item_cnt))
            lower_part = hstack([adj_matrix.transpose(), lower_right_zeros])
            full_adj_matrix = vstack([upper_part, lower_part])
            return full_adj_matrix

        def get_normalized_matrix(full_adj_matrix: csr_matrix) -> csr_matrix:
            row_sum = np.array(full_adj_matrix.sum(axis=1)).squeeze()
            row_sum[row_sum == 0] = 1.0
            normalizer = diags(row_sum ** -0.5)
            normalized_matrix = normalizer @ full_adj_matrix @ normalizer
            return normalized_matrix
    
        full_adj_matrix = get_extended_adj_matrix(self.adj_matrix)
        normalized_matrix = get_normalized_matrix(full_adj_matrix)

        coo = normalized_matrix.tocoo()
        indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size=coo.shape)
        return sparse_tensor

    def to(self, device):
        super().to(device)
        self.aggregator = self.aggregator.to(device)
        return self