import lightgbm as lgb
import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

class LightGBM(torch.nn.Module):
    def __init__(self, args, data):
        super().__init__()
        args_dict = OmegaConf.to_container(args, resolve=True)
        X_train, y_train = data['X_train'], data['y_train']
        X_valid, y_valid = data['X_valid'], data['y_valid']
        user_features = ['user_id', 'location_country','specific_age','author_multiple_works','frequently_reviewed','age_range','location_state','location_city'] 
        book_features = ['isbn', 'publisher', 'book_author','language','category','publication_range','title_count','frequent_reviewer','rare_reviewer','book_title']
        all_cols = user_features + book_features
        
        X_train, X_valid = X_train[all_cols], X_valid[all_cols]
        self.dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        self.model = lgb.train(
            params=args_dict,
            train_set=lgb.Dataset(X_train, y_train),
            valid_sets=lgb.Dataset(X_valid, y_valid),
            callbacks=[
                lgb.log_evaluation(period=100),
            ]
        )

    def forward(self, x):
        output = self.model.predict(x.cpu().numpy())
        return torch.tensor(output, device=x.device, requires_grad=True, dtype=torch.float32)