######## 기본 베이스라인 실행 스크립트 ########
# 예) $ bash run_baseline.sh
# -c : --config / -m : --model / -w : --wandb / -r : --run_name
python main.py  -c config/config_baseline.yaml  -m FM  -w True  -r FM_baseline
python main.py  -c config/config_baseline.yaml  -m FFM  -w True  -r FFM_baseline
python main.py  -c config/config_baseline.yaml  -m DeepFM  -w True  -r DeepFM_baseline
python main.py  -c config/config_baseline.yaml  -m WDN  -w True  -r WDN_baseline
python main.py  -c config/config_baseline.yaml  -m DCN  -w True  -r DCN_baseline
python main.py  -c config/config_baseline.yaml  -m NCF  -w True  -r NCF_baseline
python main.py  -c config/config_baseline.yaml  -m Image_FM  -w True  -r Image_FM_baseline
python main.py  -c config/config_baseline.yaml  -m Image_DeepFM  -w True  -r Image_DeepFM_baseline
python main.py  -c config/config_baseline.yaml  -m Text_FM  -w True  -r Text_FM_baseline
python main.py  -c config/config_baseline.yaml  -m Text_DeepFM  -w True  -r Text_DeepFM_baseline
python main.py  -c config/config_baseline.yaml  -m ResNet_DeepFM  -w True  -r ResNet_DeepFM_baseline


######## 추가 베이스라인 실행 스크립트 ########
# # 학습 없이 저장된 모델을 불러와 추론만 하고자 할 경우
# # 예) 저장된 파일명이 20240827_035641_Image_DeepFM_best.pt이고, 
# #     configuration이 saved/log/20240827_035641_Image_DeepFM/config.yaml에 저장되었다면,
# #     $ python main.py  -c saved/log/20240827_035641_Image_DeepFM/config.yaml  -m Image_DeepFM  --pred True  --ckpt 20240827_035641_Image_DeepFM_best.pt
# #     로 실행하면 됩니다.
# python main.py  -c saved/log/FM/config.yaml  -m FM  --pred True  --ckpt saved/checkpoint/FM_best.pt

# # 앙상블 실행
# # saved/submit/20001231_235959_NCF.csv 인 경우, `20001231_235959_NCF`만 작성해주어야 함
# # 예) Image_FM과 NCF를 균일하게 앙상블하고자 할 경우
# python ensemble.py  --ensemble_files Image_FM_baseline,NCF_baseline
# #    또는 가중치를 부여하고자 할 경우
# python ensemble.py  --ensemble_files Image_FM_baseline,NCF_baseline  --ensemble_strategy weighted  --ensemble_weight 0.5,0.5

# # sweep 실행
# # 예) sweep_example.yaml로 sweep을 실행하고자 할 경우
# wandb sweep config/sweep_example.yaml
# # 를 실행하면 생성되는 SWEEP_ID를 아래와 같이 wandb agent에 넣어주어 실행 (ex. userid/book-rating/sweep_id)
# wandb agent SWEEP_ID