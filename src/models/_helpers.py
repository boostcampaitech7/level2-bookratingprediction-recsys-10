import torch
import torch.nn as nn
from numpy import cumsum


# factorization을 통해 얻은 feature를 embedding 합니다.
# 사용되는 모델 : FM, CNN-FM, DCN
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims:list, embed_dim:int):
        super().__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = [0, *cumsum(field_dims)[:-1]]
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start

    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return self.embedding(x)  # (batch_size, num_fields, embed_dim)


# FM 계열 모델에서 활용되는 선형 결합 부분을 정의합니다.
# 사용되는 모델 : FM, FFM, WDN, CNN-FM
class FeaturesLinear(nn.Module):
    def __init__(self, field_dims:list, output_dim:int=1, bias:bool=True):
        super().__init__()
        self.feature_dims = sum(field_dims)
        self.output_dim = output_dim
        self.offsets = [0, *cumsum(field_dims)[:-1]]

        self.fc = nn.Embedding(self.feature_dims, self.output_dim)
        if bias:
            self.bias = nn.Parameter(torch.empty((self.output_dim,)), requires_grad=True)
    
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight.data, 0)  # cold-start
            if isinstance(m, nn.Parameter):
                nn.init.constant_(m, 0)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.fc(x), dim=1) + self.bias if hasattr(self, 'bias') \
               else torch.sum(self.fc(x), dim=1)



# dense feature 사이의 상호작용을 효율적으로 계산합니다.
# 사용되는 모델 : DeepFM, Image_FM, Image_DeepFM, Text_FM, Text_DeepFM
class FMLayer_Dense(nn.Module):
    def __init__(self):
        super().__init__()

    def square(self, x:torch.Tensor):
        return torch.pow(x,2)

    def forward(self, x):
        # square_of_sum =   # FILL HERE : Use `torch.sum()` and `self.square()` #
        # sum_of_square =   # FILL HERE : Use `torch.sum()` and `self.square()` #
        square_of_sum = self.square(torch.sum(x, dim=1))
        sum_of_square = torch.sum(self.square(x), dim=1)
        
        return 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1)
    


# sparse feature 사이의 상호작용을 효율적으로 계산합니다.
# 사용되는 모델 : FM
class FMLayer_Sparse(nn.Module):
    def __init__(self, field_dims:list, factor_dim:int):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, factor_dim)
        self.fm = FMLayer_Dense()


    def square(self, x):
        return torch.pow(x,2)
    

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.fm(x)
        
        return x
    


# 기본적인 형태의 MLP를 구현합니다.
# 사용되는 모델 : DeepFM, Image_DeepFM, Text_DeepFM, WDN, DCN, NCF
class MLP_Base(nn.Module):
    def __init__(self, input_dim, embed_dims, 
                 batchnorm=True, dropout=0.2, output_layer=False):
        super().__init__()
        self.mlp = nn.Sequential()
        for idx, embed_dim in enumerate(embed_dims):
            self.mlp.add_module(f'linear{idx}', nn.Linear(input_dim, embed_dim))
            if batchnorm:
                self.mlp.add_module(f'batchnorm{idx}', nn.BatchNorm1d(embed_dim))
            self.mlp.add_module(f'relu{idx}', nn.ReLU())
            if dropout > 0:
                self.mlp.add_module(f'dropout{idx}', nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            self.mlp.add_module('output', nn.Linear(input_dim, 1))
        
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)


    def forward(self, x):
        return self.mlp(x)
    


# 기본적인 형태의 CNN을 정의합니다. 이미지 데이터의 특징을 추출하기 위해 사용됩니다.
# 사용되는 모델 : Image_FM, Image_DeepFM
class CNN_Base(nn.Module):
    def __init__(self, input_size=(3, 64, 64), 
                 channel_list=[8,16,32], kernel_size=3, stride=2, padding=1,
                 dropout=0.2, batchnorm=True):
        super().__init__()

        # CNN 구조 : Conv2d -> BatchNorm2d -> ReLU -> Dropout 
        #           -> Conv2d -> BatchNorm2d -> ReLU -> Dropout -> MaxPool2d -> ...
        self.cnn = nn.Sequential()
        in_channel_list = [input_size[0]] + channel_list[:-1]
        for idx, (in_channel, out_channel) in enumerate(zip(in_channel_list, channel_list)):
            self.cnn.add_module(f'conv{idx}', nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding))
            if batchnorm:
                self.cnn.add_module(f'batchnorm{idx}', nn.BatchNorm2d(out_channel))
            self.cnn.add_module(f'relu{idx}', nn.ReLU())
            if dropout > 0:
                self.cnn.add_module(f'dropout{idx}', nn.Dropout(p=dropout))
            if idx % 2 == 1:
                self.cnn.add_module(f'maxpool{idx}', nn.MaxPool2d(kernel_size=2, stride=2))

        self.output_dim = self.compute_output_shape((1, *input_size))

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
                nn.init.constant_(m.bias.data, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)


    def compute_output_shape(self, input_shape):
        x = torch.rand(input_shape)
        for layer in self.cnn:
            x = layer(x)

        return x.size()
        

    def forward(self, x):
        x = self.cnn(x)  # (batch_size, out_channel, H, W)

        return x