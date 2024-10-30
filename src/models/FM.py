import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FMLayer_Sparse, MLP_Base

    


class FactorizationMachine(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.factor_dim = args.embed_dim

        # self.linear = FeaturesLinear(None, None, bias=True) # FILL HERE : Fill in the places `None` #
        # self.fm = FMLayer_Sparse(None, None) # FILL HERE : Fill in the places `None` #
        self.linear = FeaturesLinear(self.field_dims, 1, bias=True)
        self.fm = FMLayer_Sparse(self.field_dims, self.factor_dim)


    def forward(self, x: torch.Tensor):
        # y =   # FILL HERE : Use `self.linear()` and `self.fm()` #
        y = self.linear(x).squeeze(1) + self.fm(x)

        return y