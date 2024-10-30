import numpy as np
import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, CNN_Base, MLP_Base



# 기존 유저/상품 벡터와 이미지 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
# FM 모델과 유사하게 모델을 작성하되, second-order interaction 부분에 CNN을 통해 임베딩된 이미지 벡터를 추가하여 교호작용을 계산합니다.
class Image_FM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        
        # 이미지 feature를 cnn을 통해 임베딩하는 부분
        self.cnn = CNN_Base(
                            input_size=(3, args.img_size, args.img_size),  # default: (3, 224, 224)
                            channel_list=args.channel_list,                # default: [4, 8, 16]
                            kernel_size=args.kernel_size,                  # default: 3
                            stride=args.stride,                            # default: 2
                            padding=args.padding,                          # default: 1
                            batchnorm=args.cnn_batchnorm,                  # default: True
                            dropout=args.cnn_dropout                       # default: 0.2
                           )
        
        # cnn을 통해 임베딩된 이미지 벡터가 fm에 사용될 수 있도록 embed_dim 크기로 변환하는 부분
        self.cnn_embedding = nn.Linear(np.prod(self.cnn.output_dim), args.embed_dim)
        
        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()


    def forward(self, x):
        user_book_vector, img_vector = x[0], x[1]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)

        # image to dense
        img_feature = self.cnn(img_vector)  # (batch_size, out_channels, H, W)
        img_feature = img_feature.view(-1, np.prod(self.cnn.output_dim))  # (batch_size, out_channels * H * W)
        img_feature = self.cnn_embedding(img_feature)  # (batch_size, embed_dim)
        img_feature = img_feature.view(-1, 1, self.embed_dim)  # (batch_size, 1, embed_dim)
        
        # second-order interaction / dense
        dense_feature = torch.cat([user_book_embedding, img_feature], dim=1)  # (batch_size, num_fields + 1, embed_dim)
        second_order = self.fm(dense_feature)  # (batch_size,)

        return first_order.squeeze(1) + second_order




class Image_DeepFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # 이미지 feature를 cnn을 통해 임베딩하는 부분
        self.cnn = CNN_Base(
                            input_size=(3, args.img_size, args.img_size),  # default: (3, 224, 224)
                            channel_list=args.channel_list,                # default: [4, 8, 16]
                            kernel_size=args.kernel_size,                  # default: 3
                            stride=args.stride,                            # default: 2
                            padding=args.padding,                          # default: 1
                            batchnorm=args.cnn_batchnorm,                  # default: True
                            dropout=args.cnn_dropout                       # default: 0.2
                           )
        
        # cnn을 통해 임베딩된 이미지 벡터가 fm에 사용될 수 있도록 embed_dim 크기로 변환하는 부분
        self.cnn_embedding = nn.Linear(np.prod(self.cnn.output_dim), args.embed_dim)
        
        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()

        # deep network를 통해 dense feature를 학습하는 부분
        self.deep = MLP_Base(
                             input_dim=(args.embed_dim * len(self.field_dims)) + np.prod(self.cnn.output_dim),
                             embed_dims=args.mlp_dims,
                             batchnorm=args.batchnorm,
                             dropout=args.dropout,
                             output_layer=True
                            )


    def forward(self, x):
        user_book_vector, img_vector = x[0], x[1]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)

        # image to dense
        img_feature = self.cnn(img_vector)  # (batch_size, out_channels, H, W)
        img_feature_deep = img_feature.view(-1, np.prod(self.cnn.output_dim))  # (batch_size, out_channels * H * W)
        img_feature_fm = self.cnn_embedding(img_feature_deep)  # (batch_size, embed_dim)
        img_feature_fm = img_feature_fm.view(-1, 1, self.embed_dim)  # (batch_size, 1, embed_dim)
        
        # second-order interaction / dense feature
        dense_feature_fm = torch.cat([user_book_embedding, img_feature_fm], dim=1)  # (batch_size, num_fields + 1, embed_dim)
        second_order = self.fm(dense_feature_fm)  # (batch_size,)

        output_fm = first_order.squeeze(1) + second_order

        # deep network를 통해 feature를 학습하는 부분
        # fm과 달리, cnn을 통해 얻은 이미지 벡터를 embed_dim으로 축소하지 않은 원본 벡터로 사용합니다.
        dense_feature_deep = torch.cat([user_book_embedding.view(-1, len(self.field_dims) * self.embed_dim), 
                                        img_feature_deep], dim=1)
        output_dnn = self.deep(dense_feature_deep).squeeze(1)  # (batch_size,)

        return output_fm + output_dnn
    


class ResNet_DeepFM(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)

        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)

        # 이미지 feature를 resnet을 통해 임베딩하는 부분
        self.resnet = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        # in_features를 미리 저장해둠
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        # resnet을 통해 임베딩된 이미지 벡터가 fm에 사용될 수 있도록 embed_dim 크기로 변환하는 부분
        self.resnet_embedding = nn.Linear(in_features, args.embed_dim)
        
        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()

        # deep network를 통해 dense feature를 학습하는 부분
        self.deep = MLP_Base(
                             input_dim=(args.embed_dim * len(self.field_dims)) + in_features,
                             embed_dims=args.mlp_dims,
                             batchnorm=args.batchnorm,
                             dropout=args.dropout,
                             output_layer=True
                            )


    def forward(self, x):
        user_book_vector, img_vector = x[0], x[1]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)

        # image to dense
        img_feature = self.resnet(img_vector)  # (batch_size, resnet.fc.in_features)
        img_feature_deep = img_feature
        img_feature_fm = self.resnet_embedding(img_feature)
        img_feature_fm = img_feature_fm.view(-1, 1, self.embed_dim)

        # second-order interaction / dense feature
        dense_feature_fm = torch.cat([user_book_embedding, img_feature_fm], dim=1)
        second_order = self.fm(dense_feature_fm)

        output_fm = first_order.squeeze(1) + second_order

        # deep network를 통해 feature를 학습하는 부분
        dense_feature_deep = torch.cat([user_book_embedding.view(-1, len(self.field_dims) * self.embed_dim), 
                                        img_feature_deep], dim=1)
        output_dnn = self.deep(dense_feature_deep).squeeze(1)

        return output_fm + output_dnn