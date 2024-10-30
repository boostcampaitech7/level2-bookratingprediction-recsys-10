import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, MLP_Base



# DNN과 FM을 결합한 DeepFM 모델을 구현합니다.
class DeepFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()
        
        # deep network를 통해 feature를 학습하는 부분
        self.dnn = MLP_Base(
                             input_dim=(args.embed_dim * len(self.field_dims)),
                             embed_dims=args.mlp_dims,
                             batchnorm=args.batchnorm,
                             dropout=args.dropout,
                             output_layer=True
                            )


    def forward(self, x: torch.Tensor):
        # first-order interaction / sparse feature only
        first_order = self.linear(x).squeeze(1)

        # sparse to dense
        embedding = self.embedding(x)  # (batch_size, num_fields, embed_dim)

        # second-order interaction / dense
        second_order = self.fm(embedding)

        # deep network를 통해 feature를 학습하는 부분
        deep_out = self.dnn(embedding.view(-1, embedding.size(1) * embedding.size(2))).squeeze(1)

        return first_order + second_order + deep_out

