import torch
import torch.nn as nn
from numpy import cumsum
from ._helpers import FeaturesLinear

# FFM모델을 구현합니다.
# feature간의 상호작용을 파악하기 위해서 잠재백터를 두는 과정을 보여줍니다.
# FFM은 FM과 다르게 필드별로 여러개의 잠재백터를 가지므로 필드 개수만큼의 embedding parameter를 선언합니다.
class FFMLayer(nn.Module):
    def __init__(self, field_dims:list, factor_dim:int):
        super().__init__()
        self.num_fields = len(field_dims)
        self.feature_dim = sum(field_dims)
        self.factor_dim = factor_dim

        self.offsets = [0, *cumsum(field_dims)[:-1]]
        self.embeddings = nn.ModuleList([
            # FILL HERE : Fill in the places `None` with                                      #
            #             either `self.factor_dim`, `self.num_fields`, or `self.feature_dim`. #

            # nn.Embedding(
            #     None, None
            # ) for _ in range(None)
            nn.Embedding(
                self.feature_dim, self.factor_dim
            ) for _ in range(self.num_fields)
        ])
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        # xv = [None                   # FILL HERE : Fill in the places `None` using `self.embedding` #
        #       for f in range(None)]  # FILL HERE : Fill in the places `None` #
        xv = [self.embeddings[f](x) for f in range(self.num_fields)]
        
        y = list()
        for f in range(self.num_fields - 1):
            for g in range(f + 1, self.num_fields):
                y.append(xv[f][:, g] * xv[g][:, f])
        y = torch.stack(y, dim=1)

        return torch.sum(y, dim=(2,1))


# 최종적인 FFM모델입니다.
# 각 필드별로 곱해져 계산된 embedding 결과를 합하고, 마지막으로 embedding 결과를 합하여 마무리합니다.
class FieldAwareFactorizationMachine(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim
        self.feature_dim = sum(self.field_dims)

        # self.linear = FeaturesLinear(None, None, bias=True) # FILL HERE : Fill in the places `None` #
        # self.ffm = FFMLayer(None, None) # FILL HERE : Fill in the places `None` #
        self.linear = FeaturesLinear(self.field_dims, 1, bias=True)
        self.ffm = FFMLayer(self.field_dims, self.factor_dim)


    def forward(self, x: torch.Tensor):
        # y =   # FILL HERE : Use `self.linear()` and `self.ffm()` #
        y = self.linear(x).squeeze(1) + self.ffm(x)

        return y
