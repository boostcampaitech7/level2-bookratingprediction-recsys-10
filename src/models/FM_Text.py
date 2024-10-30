import torch
import torch.nn as nn
from ._helpers import FeaturesLinear, FeaturesEmbedding, FMLayer_Dense, CNN_Base, MLP_Base



# 기존 유저/상품 벡터와 텍스트 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
# FM 모델과 유사하게 모델을 작성하되, second-order interaction 부분에 사전학습모델을 통해 임베딩된 텍스트 벡터를 추가하여 교호작용을 계산합니다.
# 이 때, 텍스트 벡터에는 책의 제목과 요약 정보를 활용해서 만든 일종의 책 정보 벡터와,
# 해당 유저가 읽은 책들 중 5권의 제목과 요약 정보를 활용해서 만든 일종의 유저 정보 벡터를 사용합니다.
class Text_FM(nn.Module):
    def __init__(self, args, data):
        super(Text_FM, self).__init__()
        self.field_dims = data['field_dims']

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)
        
        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        
        # 텍스트 벡터(베이스라인 기준 768차원)가 fm에 사용될 수 있도록 embed_dim 크기로 변환하는 부분
        # 유저와 아이템 각각에 대해 서로 다른 선형변환을 적용해도 되나, 여기서는 간편하게 하나의 선형변환을 적용합니다.
        self.text_embedding = nn.Linear(args.word_dim, args.embed_dim)

        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()


    def forward(self, x):
        user_book_vector, user_text_vector, book_text_vector = x[0], x[1], x[2]

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)

        # text to dense
        user_text_feature = self.text_embedding(user_text_vector)  # (batch_size, embed_dim)
        user_text_feature = user_text_feature.view(-1, 1, user_text_feature.size(1))  # (batch_size, 1, embed_dim)
        item_text_feature = self.text_embedding(book_text_vector)  # (batch_size, embed_dim)
        item_text_feature = item_text_feature.view(-1, 1, item_text_feature.size(1))  # (batch_size, 1, embed_dim)

        # second-order interaction / dense
        dense_feature = torch.cat([user_book_embedding, user_text_feature, item_text_feature], dim=1)  # (batch_size, num_fields + 2, embed_dim)
        second_order = self.fm(dense_feature)  # (batch_size,)

        return first_order.squeeze(1) + second_order



class Text_DeepFM(nn.Module):
    def __init__(self, args, data):
        super(Text_DeepFM, self).__init__()
        self.field_dims = data['field_dims']
        self.embed_dim = args.embed_dim

        # sparse feature를 위한 선형 결합 부분
        self.linear = FeaturesLinear(self.field_dims)
        
        # sparse feature를 dense하게 임베딩하는 부분
        self.embedding = FeaturesEmbedding(self.field_dims, args.embed_dim)
        
        # 텍스트 벡터(베이스라인 기준 768차원)가 fm에 사용될 수 있도록 embed_dim 크기로 변환하는 부분
        # 유저와 아이템 각각에 대해 서로 다른 선형변환을 적용해도 되나, 여기서는 간편하게 하나의 선형변환을 적용합니다.
        self.text_embedding = nn.Linear(args.word_dim, args.embed_dim)

        # dense feature 사이의 상호작용을 효율적으로 계산하는 부분
        self.fm = FMLayer_Dense()

        # MLP를 통해 dense feature를 학습하는 부분
        self.mlp = MLP_Base(
                            input_dim=(len(self.field_dims) * self.embed_dim) + 2 * args.word_dim,
                            embed_dims=args.mlp_dims,
                            batchnorm=args.batchnorm,
                            dropout=args.dropout,
                            output_layer=True
                           )


    def forward(self, x):
        user_book_vector, user_text_vector, book_text_vector = x[0], x[1], x[2]  # (batch_size, num_fields), (batch_size, word_dim), (batch_size, word_dim)

        # first-order interaction / sparse feature only
        first_order = self.linear(user_book_vector)  # (batch_size, 1)

        # sparse to dense
        user_book_embedding = self.embedding(user_book_vector)  # (batch_size, num_fields, embed_dim)
        
        # text to dense
        user_text_feature = self.text_embedding(user_text_vector)  # (batch_size, embed_dim)
        user_text_feature = user_text_feature.view(-1, 1, user_text_feature.size(1))  # (batch_size, 1, embed_dim)
        item_text_feature = self.text_embedding(book_text_vector)  # (batch_size, embed_dim)
        item_text_feature = item_text_feature.view(-1, 1, item_text_feature.size(1))  # (batch_size, 1, embed_dim)

        # second-order interaction / dense
        dense_feature_fm = torch.cat([user_book_embedding, user_text_feature, item_text_feature], dim=1)
        second_order = self.fm(dense_feature_fm)

        output_fm = first_order.squeeze(1) + second_order

        # deep network를 통해 dense feature를 학습하는 부분
        dense_feature_deep = torch.cat([user_book_embedding.view(-1, len(self.field_dims) * self.embed_dim), 
                                        user_text_vector, book_text_vector], dim=1)
        output_dnn = self.mlp(dense_feature_deep).squeeze(1)

        return output_fm + output_dnn