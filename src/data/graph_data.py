import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .basic_data import basic_data_load, basic_data_split, basic_data_loader

def graph_data_load(args):
    return basic_data_load(args)

def graph_data_split(args, data):
    data = basic_data_split(args, data)
    user_cnt, item_cnt = data['field_dims'][0], data['field_dims'][1]
    user_indices, item_indices = data['X_train']['user_id'], data['X_train']['isbn']
    
    all_adj_matrix = csr_matrix(
        (np.ones(len(user_indices)), (user_indices, item_indices)),
        shape=(user_cnt, item_cnt)
    )
    data['adj_matrix'] = all_adj_matrix

    pos_mask = data['y_train'] >= 8
    pos_user_indices, pos_item_indices = user_indices[pos_mask], item_indices[pos_mask]
    pos_adj_matrix = csr_matrix(
        (np.ones(len(pos_user_indices)), (pos_user_indices, pos_item_indices)),
        shape=(user_cnt, item_cnt)
    )
    data['pos_adj_matrix'] = pos_adj_matrix

    neg_mask = data['y_train'] < 8
    neg_user_indices, neg_item_indices = user_indices[neg_mask], item_indices[neg_mask]
    neg_adj_matrix = csr_matrix(
        (np.ones(len(neg_user_indices)), (neg_user_indices, neg_item_indices)),
        shape=(user_cnt, item_cnt)
    )
    data['neg_adj_matrix'] = neg_adj_matrix
    return data

def graph_data_loader(args, data):
    return basic_data_loader(args, data)
