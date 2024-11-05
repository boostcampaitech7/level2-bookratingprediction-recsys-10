import torch
import torch.nn as nn
from omegaconf import OmegaConf
import xgboost as xgb

class XGBoost(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        
        # 설정된 파라미터를 dictionary 형태로 변환
        args_dict = OmegaConf.to_container(args, resolve=True)
        
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
        
        # XGBoost 모델 정의
        self.model = xgb.XGBRegressor(**args_dict)
        
        # 모델 학습
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=100
        )

    def forward(self, x):
        # 예측 수행 및 tensor 변환
        output = self.model.predict(x.cpu().numpy())
        return torch.tensor(output, device=x.device, requires_grad=True, dtype=torch.float32)
