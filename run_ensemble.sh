python tem.py --c config/config_baseline.yaml -d cuda -m FFM
python test.py  -c config/config_baseline.yaml  -m FM -d cuda -v m1 
python test.py  -c config/config_baseline.yaml  -m FFM -d cuda -v m2 
python ensemble_val.py  --ensemble_files m1,m2
