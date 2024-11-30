# the first annotation is the accuracy of individual tokenizer, while the second is that of joint tokenizer

# python3 train_model.py --lr 1e-6 --decay 0.0001   # 0.7097,       0.4603        / 0.7530,         0.4680 
# python3 train_model.py --lr 1e-6 --decay 0.001    # 0.7097,       0.4603        / 0.7530,         0.4680
# python3 train_model.py --lr 1e-6 --decay 0.01     # 0.7097,       0.4603        / 0.7530,         0.4680

# python3 train_model.py --lr 1e-5 --decay 0.0001   # 0.2189        0.2169        / 0.4196,         0.4197
# python3 train_model.py --lr 1e-5 --decay 0.001    # 0.2189,       0.2169        / 0.4196,         0.4197
# python3 train_model.py --lr 1e-5 --decay 0.01     # 0.2189,       0.2169        / 0.4196,         0.4197

# python3 train_model.py --lr 1e-4 --decay 0.0001   # 7.2611e-06,   0.0219        / 0.8444          0.0003
# python3 train_model.py --lr 1e-4 --decay 0.001    # 7.2611e-06,   0.0219        / 0.8444,         0.0003
# python3 train_model.py --lr 1e-4 --decay 0.01     # 7.2611e-06,   0.0219        / 2.4203e-06,     0.0003

# python3 train_model.py --lr 1e-3 --decay 0.0001   # 2.4203e-07,   8.9623e-05    / 8.2293e-06,     1.2989e-05
# python3 train_model.py --lr 1e-3 --decay 0.001    # 9.6815e-07,   4.1564e-05    / 8.2293e-06,     1.2989e-05
# python3 train_model.py --lr 1e-3 --decay 0.01     # 7.9872e-06,   1.8184e-05?   / 8.2293e-06,     1.2989e-05

# python3 train_model.py --lr 1e-7 --decay 0.0001   # 0.6116,       0.5519        / 0.2487,         0.6817 
# python3 train_model.py --lr 1e-7 --decay 0.001    # 0.6116,       0.5519        / 0.2487,         0.6817
# python3 train_model.py --lr 1e-7 --decay 0.01     # 0.6116,       0.5519        / 0.2487,         0.6817

# python3 train_model.py --lr 6e-7 --decay 0.0001   # 0.7329,       0.4923        / 0.6262,         0.5355
# python3 train_model.py --lr 6e-7 --decay 0.001    # 0.7329,       0.4923        / 0.6262,         0.5355
# python3 train_model.py --lr 6e-7 --decay 0.01     # 0.7329,       0.4923        / 0.6262,         0.5355

# python3 train_model.py --lr 3e-7 --decay 0.0001   # 0.6276,       0.5177        / 0.3404,         0.6146
# python3 train_model.py --lr 3e-7 --decay 0.001    # 0.6276,       0.5177        / 0.3404,         0.6146
# python3 train_model.py --lr 3e-7 --decay 0.01     # 0.6276,       0.5177        / 0.3404,         0.6146

# the first annotation is the accuracy of individual tokenizer, while the second is that of joint tokenizer

# python3 train_model.py --lr 1e-7 --decay 0.0001   # 0.6116,       0.5519        / 0.2487,         0.6817 ***
# python3 train_model.py --lr 3e-7 --decay 0.0001   # 0.6276,       0.5177        / 0.3404,         0.6146 **
# python3 train_model.py --lr 6e-7 --decay 0.0001   # 0.7329,       0.4923        / 0.6262,         0.5355 *
# python3 train_model.py --lr 1e-6 --decay 0.0001   # 0.7097,       0.4603        / 0.7530,         0.4680 
# python3 train_model.py --lr 3e-6 --decay 0.0001   #               0.7354 **     /                 0.4197
# python3 train_model.py --lr 6e-6 --decay 0.0001   #               0.7297 *      /                 0.4197
# python3 train_model.py --lr 1e-5 --decay 0.0001   # 0.2189        0.2169        / 0.4196,         0.4197
# python3 train_model.py --lr 3e-5 --decay 0.0001   #               0.0203        /                 0.0208
# python3 train_model.py --lr 6e-5 --decay 0.0001   #               0.8914 ***    /                 0.0026
# python3 train_model.py --lr 1e-4 --decay 0.0001   # 7.2611e-06,   0.0219        / 0.8444,         0.0003
# python3 train_model.py --lr 3e-4 --decay 0.0001   #               3.8726e-06    /                 0.0009
# python3 train_model.py --lr 6e-4 --decay 0.0001   #               6.2930e-06    /                 0.0013        
# python3 train_model.py --lr 1e-3 --decay 0.0001   # 2.4203e-07,   8.9623e-05    / 8.2293e-06,     1.2989e-05

# python3 train_model.py --lr 1e-7 --decay 0.001    # 0.6116,       0.5519        / 0.2487,         0.6817 ***
# python3 train_model.py --lr 3e-7 --decay 0.001    # 0.6276,       0.5177        / 0.3404,         0.6146 **
# python3 train_model.py --lr 6e-7 --decay 0.001    # 0.7329,       0.4923        / 0.6262,         0.5355 *
# python3 train_model.py --lr 1e-6 --decay 0.001    # 0.7097,       0.4603        / 0.7530,         0.4680
# python3 train_model.py --lr 3e-6 --decay 0.001    #               0.7354 **     /                 0.4197
# python3 train_model.py --lr 6e-6 --decay 0.001    #               0.7297 *      /                 0.4197
# python3 train_model.py --lr 1e-5 --decay 0.001    # 0.2189,       0.2169        / 0.4196,         0.4197
# python3 train_model.py --lr 3e-5 --decay 0.001    #               0.0203        /                 0.0208
# python3 train_model.py --lr 6e-5 --decay 0.001    #               0.8914 ***    /                 0.0026
# python3 train_model.py --lr 1e-4 --decay 0.001    # 7.2611e-06,   0.0219        / 0.8444,         0.0003
# python3 train_model.py --lr 3e-4 --decay 0.001    #               9.6815e-06    /                 0.0009
# python3 train_model.py --lr 6e-4 --decay 0.001    #               6.5350e-06    /                 0.0013
# python3 train_model.py --lr 1e-3 --decay 0.001    # 9.6815e-07,   4.1564e-05    / 8.2293e-06,     1.2989e-05


# python3 train_model.py --lr 1e-7 --decay 0.01     # 0.6116,       0.5519        / 0.2487,         0.6817 ***
# python3 train_model.py --lr 3e-7 --decay 0.01     # 0.6276,       0.5177        / 0.3404,         0.6146 **
# python3 train_model.py --lr 6e-7 --decay 0.01     # 0.7329,       0.4923        / 0.6262,         0.5355 *
# python3 train_model.py --lr 1e-6 --decay 0.01     # 0.7097,       0.4603        / 0.7530,         0.4680
# python3 train_model.py --lr 3e-6 --decay 0.01     #               0.7354 **     /                 0.4197
# python3 train_model.py --lr 6e-6 --decay 0.01     #               0.7297 *      /                 0.4197
# python3 train_model.py --lr 1e-5 --decay 0.01     # 0.2189,       0.2169        / 0.4196,         0.4197
# python3 train_model.py --lr 3e-5 --decay 0.01     #               0.0203        /                 0.0208
# python3 train_model.py --lr 6e-5 --decay 0.01     #               0.8914 ***    /                 0.0026
# python3 train_model.py --lr 1e-4 --decay 0.01     # 7.2611e-06,   0.0219        / 2.4203e-06,     0.0003
# python3 train_model.py --lr 3e-4 --decay 0.01     #               3.8726e-06    /                 0.0009
# python3 train_model.py --lr 6e-4 --decay 0.01     #               7.0190e-06    /                 0.0013
# python3 train_model.py --lr 1e-3 --decay 0.01     # 7.9872e-06,   1.8184e-05?   / 8.2293e-06,     1.2989e-05





