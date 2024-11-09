# 📖 Book Rating Prediction
> 사용자의 책 평점 데이터를 이용하여 새로운 책에 대한 선호도를 예측하는 테스크입니다.

## Team
|곽정무|박준하|박태지|신경호|이효준
|:-:|:-:|:-:|:-:|:-:|
|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/20788198?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/81938013?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/112858891?v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/103016689?s=64&v=4'>|<img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/176903280?v=4'>|
|<a href = 'https://github.com/jkwag'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/joshua5301'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/spsp4755'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/Human3321'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|<a href = 'https://github.com/Jun9096'><img src = 'https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white'> </a>|






## 프로젝트 구조
```
 📦level2-bookratingprediction-recsys-10
 ┣ 📂config
 ┃ ┣ 📜config_baseline.yaml
 ┃ ┗ 📜sweep_example.yaml
 ┣ 📂src
 ┃ ┣ 📂data
 ┃ ┃ ┣ 📜basic_data.py
 ┃ ┃ ┣ 📜context_data.py
 ┃ ┃ ┣ 📜fixed_context_data.py
 ┃ ┃ ┣ 📜graph_data.py
 ┃ ┃ ┣ 📜image_data.py
 ┃ ┃ ┣ 📜text_data.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂ensembles
 ┃ ┃ ┗ 📜ensembles.py
 ┃ ┣ 📂loss
 ┃ ┃ ┗ 📜loss.py
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📜CatBoost.py
 ┃ ┃ ┣ 📜DCN.py
 ┃ ┃ ┣ 📜DeepFM.py
 ┃ ┃ ┣ 📜FFM.py
 ┃ ┃ ┣ 📜FM.py
 ┃ ┃ ┣ 📜FM_Image.py
 ┃ ┃ ┣ 📜FM_Text.py
 ┃ ┃ ┣ 📜LightGBM.py
 ┃ ┃ ┣ 📜LightGCN.py
 ┃ ┃ ┣ 📜NCF.py
 ┃ ┃ ┣ 📜WDN.py
 ┃ ┃ ┣ 📜XGBoost.py
 ┃ ┃ ┣ 📜_helpers.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂train
 ┃ ┃ ┣ 📜trainer.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📜utils.py
 ┃ ┗ 📜__init__.py
 ┣ 📂tem
 ┣ 📜.gitignore
 ┣ 📜ensemble.py
 ┣ 📜ensemble_val.py
 ┣ 📜main.py
 ┣ 📜README.md
 ┣ 📜requirement.txt
 ┣ 📜run_ensemble.sh
 ┣ 📜tem.py
 ┗ 📜test.py
```

## 개발환경
- python 3.10.0

## 기술스택
<img src = 'https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54'> <img src = 'https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white'> <img src= 'https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white'> <img src ='https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white'> 

### 협업툴
<img src ='https://img.shields.io/badge/jira-%230A0FFF.svg?style=for-the-badge&logo=jira&logoColor=white'> <img src = 'https://img.shields.io/badge/confluence-%23172BF4.svg?style=for-the-badge&logo=confluence&logoColor=white'>


## 라이브러리 설치
```shell
$ pip install -r requirement.txt
```

## 기능 및 예시
- train
```shell
$ python main.py  -c config/config_baseline.yaml  -m FFM
```
FFM으로 모델을 학습하는 쉘 스크립트입니다. 인자를 직접 입력하지 않으면 yaml파일에 설정된 인자들을 불러옵니다. 모델이 학습되면 submit 폴더에 csv파일에 저장됩니다.

- ensemble
```shell
$ python emsemble.py --ensemble_files {실행결과1.csv},{실행결과2.csv}
```

다수의 모델을 학습시킨 후 앙상블을 진행합니다. 학습결과의 csv 파일을 입력 단자로 받습니다.

