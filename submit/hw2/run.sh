# the first annotation is the accuracy of individual tokenizer, while the second is that of joint tokenizer

python3 train_model.py --lr 1e-6 --decay 0.0001 # 0.7097, 0.4603
python3 train_model.py --lr 1e-6 --decay 0.001 # 0.7097, 0.4603
python3 train_model.py --lr 1e-6 --decay 0.01 # 0.7097, 0.4603

python3 train_model.py --lr 1e-5 --decay 0.0001 # 0.2189 0.2169
python3 train_model.py --lr 1e-5 --decay 0.001 # 0.2189, 0.2169
python3 train_model.py --lr 1e-5 --decay 0.01 # 0.2189, 0.2169

python3 train_model.py --lr 1e-4 --decay 0.0001 # 7.2611e-06, 0.0219
python3 train_model.py --lr 1e-4 --decay 0.001 # 7.2611e-06, 0.0219
python3 train_model.py --lr 1e-4 --decay 0.01 # 7.2611e-06, 0.0219

python3 train_model.py --lr 1e-3 --decay 0.0001 # 2.4203e-07, 8.9623e-05
python3 train_model.py --lr 1e-3 --decay 0.001 # 9.6815e-07, 4.1564e-05
python3 train_model.py --lr 1e-3 --decay 0.01 # 7.9872e-06, 1.8184e-05?

python3 train_model.py --lr 1e-7 --decay 0.0001 # 0.6116, 0.5519
python3 train_model.py --lr 1e-7 --decay 0.001 # 0.6116, 0.5519
python3 train_model.py --lr 1e-7 --decay 0.01 # 0.6116, 0.5519

python3 train_model.py --lr 6e-7 --decay 0.0001 # 0.7329, 0.4923
python3 train_model.py --lr 6e-7 --decay 0.001 # 0.7329, 0.4923
python3 train_model.py --lr 6e-7 --decay 0.01 # 0.7329, 0.4923

python3 train_model.py --lr 3e-7 --decay 0.0001 # 0.6276, 0.5177
python3 train_model.py --lr 3e-7 --decay 0.001 # 0.6276, 0.5177
python3 train_model.py --lr 3e-7 --decay 0.01 # 0.6276, 0.5177