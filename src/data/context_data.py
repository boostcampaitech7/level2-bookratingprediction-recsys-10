import numpy as np
import pandas as pd
import regex
import torch
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


def process_context_data(users, books,all_):
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
    all_ : pd.DataFrame
        train과 test 데이터를 합친 데이터
    
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
    all_ = all_.copy()
    
    # 'isbn'을 기준으로 all_과 books 병합
    item_review = all_.merge(books_, how='left', on='isbn')
    
    # 카테고리 전처리: 첫 번째 항목만 남기고 소문자로 변환
    item_review['category'] = item_review['category'].apply(lambda x: str2list(x)[0].lower() if not pd.isna(x) else 'others')
    
    # 카테고리별 리뷰 수 집계
    category_review_count = item_review.groupby('category').size().reset_index(name='category_review_count')
    
    # 상위 60개 카테고리 선택
    top_60_categories = category_review_count.nlargest(80, 'category_review_count')['category']
    
    # 상위 60개에 포함되지 않는 카테고리를 'others'로 변경
    books_['category'] = books_['category'].apply(lambda x: x if x in top_60_categories.values else 'others')





    # 출판사 범주화 코드
    #하위 10% 이하의 출판 빈도를 'others'로 통합
    publisher_counts = books_['publisher'].value_counts()
    books_['publisher_count'] = books_['publisher'].map(publisher_counts)
    threshold_10 = np.percentile(books_['publisher_count'], 10)
    books_['publisher_others_10'] = books_['publisher'].where(books_['publisher_count'] >   threshold_10, 'others')
 



    
    books_['language'] = books_['language'].fillna(books_['language'].mode()[0])
    books_['publication_range'] = books_['year_of_publication'].apply(lambda x: x // 10 * 10)  # 1990년대, 2000년대, 2010년대, ...

    users_['age_range'] = users_['age'].apply(lambda x: x // 10 * 10)  # 10대, 20대, 30대, ...
    
    users_['specific_age'] = users_['age'].apply(lambda x: 0 if pd.isna(x) else 1) # 나이 존재시 1, 없으면 0
    # 책 제목별 개수 추가
    title_counts = books_.groupby('book_title').size().reset_index(name='title_count')
    books_ = books_.merge(title_counts, on='book_title', how='left')
    
    author_counts = books_.groupby('book_author').size().reset_index(name='author_work_count')
    books_ = books_.merge(author_counts, on='book_author', how='left')
    books_['author_multiple_works'] = np.where(books_['author_work_count'] > 1, 1, 0)
    
    # 아이템 리뷰 빈도 추가
    item_review_count = all_['isbn'].value_counts().reset_index()
    item_review_count.columns = ['isbn', 'item_review_count']
    books_ = books_.merge(item_review_count, on='isbn', how='left')
    books_['item_review_count'] = books_['item_review_count'].fillna(0)
    books_['frequently_reviewed'] = books_['item_review_count'].apply(lambda x: 1 if x >= 30 else 0)

    # 유저 리뷰 횟수 추가
    user_review_count = all_['user_id'].value_counts().reset_index()
    user_review_count.columns = ['user_id', 'user_review_count']
    users_ = users_.merge(user_review_count, on='user_id', how='left')
    users_['user_review_count'] = users_['user_review_count'].fillna(0)
    users_['frequent_reviewer'] = users_['user_review_count'].apply(lambda x: 1 if x>=100 else 0)
    users_['rare_reviewer'] = users_['user_review_count'].apply(lambda x: 1 if x <= 5 else 0)
    # publiser의 book count에 대한 범주 추가
    # publisher_count = books_['publisher'].value_counts().reset_index()
    # publisher_count.columns = ['publisher', 'publisher_review_count']
    # books_ = books_.merge(publisher_count, on='publisher', how='left')
    # books_['publisher_review_count'] = books_['publisher_review_count'].fillna(0)
    # books_['rare_publisher'] = books_['publisher_review_count'].apply(lambda x: 1 if x < 30 else 0)
    # books_['nomal_publisher'] = books_['publisher_review_count'].apply(lambda x: 1 if x >= 30 and x < 500 else 0)
    # books_['frequently_publisher'] = books_['publisher_review_count'].apply(lambda x: 1 if x >= 500 else 0)

    #publisher별 user_rating count 범주화
    check = books_.copy()
    view = all_.merge(check, on='isbn', how='left')
    publisher_rating_count = view.groupby('publisher')['rating'].count().reset_index()
    publisher_rating_count.columns = ['publisher', 'publisher_rating_count']  # 열 이름 설정
    books_ = books_.merge(publisher_rating_count, on='publisher', how='left')
    books_['publisher_rating_count'] = books_['publisher_rating_count'].fillna(-1)  # 결측치는 0으로 대체
    books_['rare_publisher'] = books_['publisher_rating_count'].apply(lambda x: 1 if x < 30 else 0)
    books_['nomal_publisher'] = books_['publisher_rating_count'].apply(lambda x: 1 if 30 <= x < 500 else 0)
    books_['frequently_publisher'] = books_['publisher_rating_count'].apply(lambda x: 1 if x >= 500 else 0)



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
            
            
            fill_country = users_[users_['location_city'] == row['location_city']]['location_country'].mode()
            fill_country = fill_country[0] if len(fill_country) > 0 else np.nan
            users_.loc[idx, 'location_country'] = fill_country
                

               
    
    users_ = users_.drop(['location'], axis=1)

    return users_, books_


def context_data_load(args):
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
    
    all_ = pd.concat([train,test], axis = 0)

    users_, books_ = process_context_data(users, books, all_)
    
    
    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성
    # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
    # 베이스라인에서는 가능한 모든 컬럼을 사용하도록 구성하였습니다.
    # NCF를 사용할 경우, idx 0, 1은 각각 user_id, isbn이어야 합니다.
    user_features = ['user_id', 'location_country','specific_age','author_multiple_works','frequently_reviewed']
    book_features = ['isbn', 'book_author','language','category','publication_range','title_count','frequent_reviewer','rare_reviewer','publisher_others_10'
                     ,'rare_publisher','nomal_publisher','frequently_publisher']
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
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = pd.Categorical(train_df[col], categories=unique_labels).codes
        test_df[col] = pd.Categorical(test_df[col], categories=unique_labels).codes
    
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


def context_data_split(args, data):
    '''data 내의 학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다.'''
    return basic_data_split(args, data)


def context_data_loader(args, data):
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
