import torch
import torch.nn as nn
from ._helpers import FeaturesEmbedding, FeaturesLinear, MLP_Base


# Wide: memorization을 담당하는 generalized linear model
# Deep: generalization을 담당하는 feed-forward neural network
# wide and deep model은 위의 wide 와 deep 을 결합하는 모델입니다.
# 데이터를 embedding 하여 MLP 으로 학습시킨 Deep 모델과 parameter에 bias를 더한 linear 모델을 합하여 최종결과를 도출합니다.
class WideAndDeep(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.linear = FeaturesLinear(self.field_dims)
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        self.embed_output_dim = len(self.field_dims) * args.embed_dim
        self.mlp = MLP_Base(self.embed_output_dim, args.mlp_dims, args.batchnorm, args.dropout, output_layer=True)


    def forward(self, x: torch.Tensor):
        embed_x = self.embedding(x)
        y_wide = self.linear(x).squeeze(1)
        y_deep = self.mlp(embed_x.view(-1, self.embed_output_dim)).squeeze(1)

        return y_wide + y_deep
