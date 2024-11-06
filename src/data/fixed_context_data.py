import numpy as np
import pandas as pd
import re
import regex
import torch
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from .basic_data import basic_data_split

def str2list(x: str) -> list:
    '''문자열을 리스트로 변환하는 함수'''
    return x[1:-1].split(', ')


def split_location(x: str) -> list:
    '''
    Parameters
    ----------
    x : str
        location 데이터

    Returns
    -------
    res : list
        location 데이터를 나눈 뒤, 정제한 결과를 반환합니다.
        순서는 country, state, city, ... 입니다.
    '''
    res = x.split(',')
    res = [i.strip().lower() for i in res]
    res = [regex.sub(r'[^a-zA-Z/ ]', '', i) for i in res]  # remove special characters
    res = [i if i not in ['n/a', ''] else np.nan for i in res]  # change 'n/a' into NaN
    res.reverse()  # reverse the list to get country, state, city, ... order

    for i in range(len(res)-1, 0, -1):
        if (res[i] in res[:i]) and (not pd.isna(res[i])):  # remove duplicated values if not NaN
            res.pop(i)

    return res


def process_fixed_context_data(users, books):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    
    Returns
    -------
    label_to_idx : dict
        데이터를 인덱싱한 정보를 담은 딕셔너리
    idx_to_label : dict
        인덱스를 다시 원래 데이터로 변환하는 정보를 담은 딕셔너리
    train_df : pd.DataFrame
        train 데이터
    test_df : pd.DataFrame
        test 데이터
    """

    users_ = users.copy()
    books_ = books.copy()

    # 데이터 전처리
    # book_author 값 정제
    books_['book_author'] = books_['book_author'].replace(['Not Applicable (Na)', 'Not Applicable (Na )', 'Na', 'Not Applicable'], pd.NA)

    # author 이름에 있는 특수문자 제거 후 토큰화
    books_['split_name'] = books_['book_author'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s]', '', str(x)).split() if pd.notnull(x) else [])

    # 동일한 book_title 그룹 내에서 공통 토큰이 최소 min_common_tokens개 이상인 경우 가장 빈도 높은 author로 통일
    def has_sufficient_common_tokens(group, min_common_tokens=2):
        tokens = group['split_name'].tolist()
        indices_with_common_tokens = set()

        for i in range(len(tokens)):
            for j in range(len(tokens)):
                if i != j:
                    if len(set(tokens[i]).intersection(tokens[j])) >= min_common_tokens:
                        indices_with_common_tokens.add(i)
                        indices_with_common_tokens.add(j)
    
        return list(indices_with_common_tokens)
    
    def get_most_common(series):
        non_null_values = series.dropna()
        if not non_null_values.empty:
            return Counter(non_null_values).most_common(1)[0][0]
        return None
    
    for title, group in books_.groupby('book_title'):
        common_token_indices = has_sufficient_common_tokens(group)
        if common_token_indices:
            most_common_author = get_most_common(group.iloc[common_token_indices]['book_author'])
            books_.loc[group.index[common_token_indices], 'book_author'] = most_common_author

    # category 및 summary의 최빈값으로 통일
    def get_most_common_or_frequent(series, overall_freq):
        non_null_values = series.dropna()
        group_common = Counter(non_null_values).most_common()
        if group_common:
            max_count = group_common[0][1]
            max_items = [item[0] for item in group_common if item[1] == max_count]
            if len(max_items) == 1:
                return max_items[0]
            else:
                return max(sorted(max_items, key=lambda x: overall_freq[x], reverse=True))
        return None
    
    overall_category_freq = Counter(books_['category'].dropna())
    overall_summary_freq = Counter(books_['summary'].dropna())

    for (title, author), group in books_.groupby(['book_title', 'book_author']):
        most_common_category = get_most_common_or_frequent(group['category'], overall_category_freq)
        most_common_summary = get_most_common_or_frequent(group['summary'], overall_summary_freq)
        
        books_.loc[group.index, 'category'] = most_common_category
        books_.loc[group.index, 'summary'] = most_common_summary

    books_.drop(columns=['split_name'], inplace=True)




    books_['category'] = books_['category'].apply(lambda x: str2list(x)[0] if not pd.isna(x) else np.nan)
    books_['language'] = books_['language'].fillna(books_['language'].mode()[0])
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)  # 1990년대, 2000년대, 2010년대, ...

    users_['age'] = users_['age'].fillna(users_['age'].mode()[0])
    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)  # 10대, 20대, 30대, ...

    users_['location_list'] = users_['location'].apply(lambda x: split_location(x)) 
    users_['location_country'] = users_['location_list'].apply(lambda x: x[0])
    users_['location_state'] = users_['location_list'].apply(lambda x: x[1] if len(x) > 1 else np.nan)
    users_['location_city'] = users_['location_list'].apply(lambda x: x[2] if len(x) > 2 else np.nan)
    for idx, row in users_.iterrows():
        if (not pd.isna(row['location_state'])) and pd.isna(row['location_country']):
            fill_country = users_[users_['location_state'] == row['location_state']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
            users_.loc[idx, 'location_country'] = fill_country
        elif (not pd.isna(row['location_city'])) and pd.isna(row['location_state']):
            if not pd.isna(row['location_country']):
                fill_state = users_[(users_['location_country'] == row['location_country']) 
                                    & (users_['location_city'] == row['location_city'])]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                users_.loc[idx, 'location_state'] = fill_state
            else:
                fill_state = users_[users_['location_city'] == row['location_city']]['location_state'].mode()
                fill_state = fill_state[0] if len(fill_state) > 0 else np.nan
                fill_country = users_[users_['location_city'] == row['location_city']]['location_country'].mode()
                fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
                users_.loc[idx, 'location_country'] = fill_country
                users_.loc[idx, 'location_state'] = fill_state

               
    
    users_ = users_.drop(['location'], axis=1)

    return users_, books_


def fixed_context_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    
    Returns
    -------
    data : dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.dataset.data_path + 'users.csv')
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    users_, books_ = process_fixed_context_data(users, books)
    
    
    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성
    # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
    # 베이스라인에서는 가능한 모든 컬럼을 사용하도록 구성하였습니다.
    # NCF를 사용할 경우, idx 0, 1은 각각 user_id, isbn이어야 합니다.
    user_features = ['user_id', 'age_range', 'location_country', 'location_state', 'location_city']
    book_features = ['isbn', 'book_title', 'book_author', 'publisher', 'language', 'category', 'publication_range']
    if args.model == 'NCF':
        sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features) - {'user_id', 'isbn'})
    else:
        sparse_cols = user_features + book_features

    # 선택한 컬럼만 추출하여 데이터 조인
    train_df = train.merge(users_, on='user_id', how='left')\
                    .merge(books_, on='isbn', how='left')[sparse_cols + ['rating']]
    test_df = test.merge(users_, on='user_id', how='left')\
                  .merge(books_, on='isbn', how='left')[sparse_cols]
    all_df = pd.concat([train_df, test_df], axis=0)

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        train_df[col] = train_df[col].fillna('unknown')
        test_df[col] = test_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = pd.Categorical(train_df[col], categories=unique_labels).codes
        train_df[col] = pd.Categorical(test_df[col], categories=unique_labels).codes
    
    field_dims = [len(label2idx[col]) for col in train_df.columns if col != 'rating']

    data = {
            'train':train_df,
            'test':test_df,
            'field_names':sparse_cols,
            'field_dims':field_dims,
            'label2idx':label2idx,
            'idx2label':idx2label,
            'sub':sub,
            }

    return data


def fixed_context_data_split(args, data):
    '''data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.'''
    return basic_data_split(args, data)


def fixed_context_data_loader(args, data):
    """
    Parameters
    ----------
    args.dataloader.batch_size : int
        데이터 batch에 사용할 데이터 사이즈
    args.dataloader.shuffle : bool
        data shuffle 여부
    args.dataloader.num_workers: int
        dataloader에서 사용할 멀티프로세서 수
    args.dataset.valid_ratio : float
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용합니다.
    data : dict
        context_data_load 함수에서 반환된 데이터
    
    Returns
    -------
    data : dict
        DataLoader가 추가된 데이터를 반환합니다.
    """

    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values)) if args.dataset.valid_ratio != 0 else None
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data