from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from omegaconf import OmegaConf



class CatBoost(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        
        # 설정된 파라미터를 dictionary 형태로 변환
        args_dict = OmegaConf.to_container(args, resolve=True)
        valid_params = CatBoostRegressor().get_params()  # 빈 모델에서 기본 파라미터 가져오기
        args_dict = {k: v for k, v in args_dict.items() if k in valid_params}
        
        # 데이터와 feature 컬럼 정의
        X_train, y_train = data['X_train'], data['y_train']
        X_valid, y_valid = data['X_valid'], data['y_valid']
        
        user_features = ['user_id', 'age_range', 'location_country', 'location_state', 'location_city']
        book_features = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category', 'publication_range']
        all_cols = user_features + book_features
        
        # 필요한 컬럼만 선택
        X_train, X_valid = X_train[all_cols], X_valid[all_cols]
        
        # 모델이 학습 가능한 더미 파라미터 추가
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        
        # CatBoost 모델 정의
        self.model = CatBoostRegressor(**args_dict)
        
        # CatBoost Pool 객체로 데이터 준비
        train_pool = Pool(data=X_train, label=y_train)
        valid_pool = Pool(data=X_valid, label=y_valid)
        
        # 모델 학습
        self.model.fit(
            train_pool,
            eval_set=valid_pool,
            verbose=100,
            use_best_model=True,
            early_stopping_rounds=100
        )

    def forward(self, x):
        # 예측 수행 및 tensor 변환
        output = self.model.predict(x.cpu().numpy())
        return torch.tensor(output, device=x.device, requires_grad=True, dtype=torch.float32)

