# 0.001, 0.0007 for instruction eval set, # 0.0002, 0.9945 for normal eval set
# python3 finetuning.py --lr 1e-3 --decay 0.0001

# the first one is indiv tokenizer, the second one is joint tokenizer
python3 finetuning.py --lr 1e-7 --decay 0.0001   # 0.0005, 0.9964
python3 finetuning.py --lr 3e-7 --decay 0.0001   # 0.0005, 0.9964
python3 finetuning.py --lr 6e-7 --decay 0.0001   # 0.0005, 0.9963
python3 finetuning.py --lr 1e-6 --decay 0.0001   # 0.0004, 0.9962
python3 finetuning.py --lr 3e-6 --decay 0.0001   # 0.0003, 0.9958
python3 finetuning.py --lr 6e-6 --decay 0.0001   # 0.0001, 0.9948
python3 finetuning.py --lr 1e-5 --decay 0.0001   # 7.2694e-05, 0.9925
python3 finetuning.py --lr 3e-5 --decay 0.0001   # 0, 0.8478
python3 finetuning.py --lr 6e-5 --decay 0.0001   # 0, 0.4520
python3 finetuning.py --lr 1e-4 --decay 0.0001   # 0, 0.3745
python3 finetuning.py --lr 3e-4 --decay 0.0001   # 0, 0.0139
python3 finetuning.py --lr 6e-4 --decay 0.0001   # 0, 0.0180 
python3 finetuning.py --lr 1e-3 --decay 0.0001   # 0, 0.0240



