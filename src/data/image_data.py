import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader, Dataset
from .basic_data import basic_data_split


class Image_Dataset(Dataset):
    def __init__(self, user_book_vector, img_vector, rating=None):
        """
        Parameters
        ----------
        user_book_vector : np.ndarray
            모델 학습에 사용할 유저 및 책 정보(범주형 데이터)를 입력합니다.
        img_vector : np.ndarray
            벡터화된 이미지 데이터를 입력합니다.
        rating : np.ndarray
            정답 데이터를 입력합니다.
        """
        self.user_book_vector = user_book_vector
        self.img_vector = img_vector
        self.rating = rating
    def __len__(self):
        return self.user_book_vector.shape[0]
    def __getitem__(self, i):
        return {
                'user_book_vector' : torch.tensor(self.user_book_vector[i], dtype=torch.long),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32),
                'rating' : torch.tensor(self.rating[i], dtype=torch.float32)
                } if self.rating is not None else \
                {
                'user_book_vector' : torch.tensor(self.user_book_vector[i], dtype=torch.long),
                'img_vector' : torch.tensor(self.img_vector[i], dtype=torch.float32)
                }


def image_vector(path, img_size):
    """
    Parameters
    ----------
    path : str
        이미지가 존재하는 경로를 입력합니다.

    Returns
    -------
    img_fe : np.ndarray
        이미지를 벡터화한 결과를 반환합니다.
        베이스라인에서는 grayscale일 경우 RGB로 변경한 뒤, img_size x img_size 로 사이즈를 맞추어 numpy로 반환합니다.
    """
    img = Image.open(path)
    transform = v2.Compose([
        v2.Lambda(lambda x: x.convert('RGB') if x.mode != 'RGB' else x),
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(img).numpy()


def process_img_data(books, args):
    """
    Parameters
    ----------
    books : pd.DataFrame
        책 정보에 대한 데이터 프레임을 입력합니다.
    
    Returns
    -------
    books_ : pd.DataFrame
        이미지 정보를 벡터화하여 추가한 데이터 프레임을 반환합니다.
    """
    books_ = books.copy()
    books_['img_path'] = books_['img_path'].apply(lambda x: f'data/{x}')
    img_vecs = []
    for idx in tqdm(books_.index):
        img_vec = image_vector(books_.loc[idx, 'img_path'], args.model_args[args.model].img_size)
        img_vecs.append(img_vec)

    books_['img_vector'] = img_vecs

    return books_


def image_data_load(args):
    """
    Parameters
    ----------
    args.dataset.data_path : str
        데이터 경로를 설정할 수 있는 parser
    data : dict
        image_data_split로 부터 학습/평가/테스트 데이터가 담긴 사전 형식의 데이터를 입력합니다.
    
    Returns
    -------
    data : Dict
        학습 및 테스트 데이터가 담긴 사전 형식의 데이터를 반환합니다.
    """
    users = pd.read_csv(args.dataset.data_path + 'users.csv')  # 베이스라인 코드에서는 사실상 사용되지 않음
    books = pd.read_csv(args.dataset.data_path + 'books.csv')
    train = pd.read_csv(args.dataset.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.dataset.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.dataset.data_path + 'sample_submission.csv')

    # 이미지를 벡터화하여 데이터 프레임에 추가
    books_ = process_img_data(books, args)

    # 유저 및 책 정보를 합쳐서 데이터 프레임 생성 (단, 베이스라인에서는 user_id, isbn, img_vector만 사용함)
    # 사용할 컬럼을 user_features와 book_features에 정의합니다. (단, 모두 범주형 데이터로 가정)
    user_features = []
    book_features = []
    sparse_cols = ['user_id', 'isbn'] + list(set(user_features + book_features) - {'user_id', 'isbn'})

    train_df = train.merge(books_, on='isbn', how='left')\
                    .merge(users, on='user_id', how='left')[sparse_cols + ['img_vector', 'rating']]
    test_df = test.merge(books_, on='isbn', how='left')\
                  .merge(users, on='user_id', how='left')[sparse_cols + ['img_vector']]
    all_df = pd.concat([train_df, test_df], axis=0)

    # feature_cols의 데이터만 라벨 인코딩하고 인덱스 정보를 저장
    label2idx, idx2label = {}, {}
    for col in sparse_cols:
        all_df[col] = all_df[col].fillna('unknown')
        unique_labels = all_df[col].astype("category").cat.categories
        label2idx[col] = {label:idx for idx, label in enumerate(unique_labels)}
        idx2label[col] = {idx:label for idx, label in enumerate(unique_labels)}
        train_df[col] = train_df[col].astype("category").cat.codes
        test_df[col] = test_df[col].astype("category").cat.codes

    field_dims = [len(label2idx[col]) for col in sparse_cols]

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


def image_data_split(args, data):
    """학습 데이터를 학습/검증 데이터로 나누어 추가한 후 반환합니다."""
    return basic_data_split(args, data)


def image_data_loader(args, data):
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
        Train/Valid split 비율로, 0일 경우에 대한 처리를 위해 사용
    data : Dict
        image_data_split()에서 반환된 데이터
    
    Returns
    -------
    data : Dict
        Image_Dataset 형태의 학습/검증/테스트 데이터를 DataLoader로 변환하여 추가한 후 반환합니다.
    """
    train_dataset = Image_Dataset(
                                data['X_train'][data['field_names']].values,
                                data['X_train']['img_vector'].values,
                                data['y_train'].values
                                )
    valid_dataset = Image_Dataset(
                                data['X_valid'][data['field_names']].values,
                                data['X_valid']['img_vector'].values,
                                data['y_valid'].values
                                ) if args.dataset.valid_ratio != 0 else None
    test_dataset = Image_Dataset(
                                data['test'][data['field_names']].values,
                                data['test']['img_vector'].values
                                )

    train_dataloader = DataLoader(train_dataset, batch_size=args.dataloader.batch_size, shuffle=args.dataloader.shuffle, num_workers=args.dataloader.num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers) if args.dataset.valid_ratio != 0 else None
    test_dataloader = DataLoader(test_dataset, batch_size=args.dataloader.batch_size, shuffle=False, num_workers=args.dataloader.num_workers)
    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader
    return data
