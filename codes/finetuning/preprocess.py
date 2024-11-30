from transformers import AutoTokenizer, GPT2LMHeadModel
import pandas as pd
from datasets import load_dataset
import re, os 
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

data_raw = str(Path().resolve().parent.parent) + os.sep + "data/data_raw"
data_raw_processed = str(Path().resolve().parent.parent) + os.sep + "data/data_raw_processed"
data_it_processed = str(Path().resolve().parent.parent) + os.sep + "data/data_it_processed"
tokenizer_path = str(Path().resolve().parent.parent) + os.sep + "tokenizers"
weight_path = str(Path().resolve().parent.parent) + os.sep + "weights"


# pretrain/preproces.py로  preprocess된 데이터를 한 번 더 preprocess하는 모듈
def add_ineq_sign(list_x):
    digits = {'일': 1, '이': 2, '삼': 3, '사': 4, '오': 5, '육': 6, '칠': 7, '팔': 8, '구': 9}
    units = {'십만': 100000, '만': 10000, '천': 1000, '백': 100, '십': 10}
    list_x = list_x.split(" ")

    list_out = []
    for i in list_x:
        units_only = re.sub(r"[일이삼사오육칠팔구]", "_", i).split("_")
        units_only = [units[i] for i in units_only if i != ""]

        digits_only = re.sub(r"[^일이삼사오육칠팔구]", "_", i).split("_")
        digits_only = [digits[i] for i in digits_only if i != ""]
        
        out = 0
        for i, j in zip(units_only, digits_only):
            out += i * j
        
        list_out.append(out)
    
    ineq_list = []
    for i, j in zip(list_out, list_out[1:]):

        if i <= j:
            ineq_list.append("<")
        if i > j:
            ineq_list.append(">")
    
    out_str = ""
    for i, j in zip(list_x, ineq_list):
        out_str += i + f" {j} "
    out_str += list_x[-1]

    return out_str


def main():

    with open(f"{data_raw_processed}/test_data.csv") as f:
        x = pd.Series([i.strip() for i in f.readlines()][1:]) # 첫 번째 string은 칼럼 네임

    length = int(len(x) * 0.8)
    length2 =  int(len(x) * 0.9)
    xx = x.sample(frac = 1, random_state = 42)

    train_dataset = x[:length]
    valid_dataset = x[length:length2]
    test_dataset = x[length2:]

    train_dataset.to_csv(f"{data_it_processed}/train_data_no_signs.csv", index = False)
    valid_dataset.to_csv(f"{data_it_processed}/valid_data_no_signs.csv", index = False)
    test_dataset.to_csv(f"{data_it_processed}/test_data_no_signs.csv", index = False)

    
    train_dataset.progress_map(add_ineq_sign).to_csv(f"{data_it_processed}/train_data_with_signs.csv", index = False)
    valid_dataset.progress_map(add_ineq_sign).to_csv(f"{data_it_processed}/valid_data_with_signs.csv", index = False)
    test_dataset.progress_map(add_ineq_sign).to_csv(f"{data_it_processed}/test_data_with_signs.csv", index = False)

    # print(train_dataset)
    
    # 아씨 깃헙이 틀렸네
    # print(add_ineq_sign("일십만오만사천칠백 일십만육만삼천이백 일십만오천사십구"))

if __name__ == "__main__":
    main()