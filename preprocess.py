import pandas as pd, numpy as np
from pathlib import Path
import os
from KoreanNumber import num2kr
import torch, torch.nn as nn
from transformers import AutoModel, DataCollatorForLanguageModeling, AutoTokenizer
import json

data_raw = str(Path().resolve()) + os.sep + "data"
data_processed = str(Path().resolve()) + os.sep + "data_processed"


def agg_data(full_cols, target_cols, merge_cols):

    files = os.listdir(data_raw)

    df = pd.DataFrame()
    meta = []
    for file in files:
        business_name = file.split("_")[0]
        full_cols_file = [i.format(business_name) for i in full_cols]
        target_cols_file = [i.format(business_name) for i in target_cols]

        with open(f"{data_raw}/{file}", "r") as f:

            x = pd.DataFrame([i.split("\t") for i in f.readlines()])
            x = x.loc[x.index[::-1]].reset_index(drop=True)

            x.columns = full_cols_file

            if len(df) == 0:
                df = x[target_cols_file]
            else:
                df = df.merge(x[target_cols_file], on=merge_cols)

        meta.append(file)

    return df, meta


def num_to_str(out_file_name, meta_file_name, value_cols):
    df = pd.read_csv(out_file_name)
    with open(meta_file_name, "r") as f:
        meta = [i.strip("\n").split("_")[0] for i in f.readlines()]

    meta = [i.split("_")[0] for i in meta]
    # end_{} 하나만 있으니 일단 [0]
    value_cols_file = [value_cols[0].format(i) for i in meta]
    x = df[value_cols_file].map(lambda x: int(str(x).replace(",", "")))
    x = x.map(lambda x: num2kr.num2kr(x, 1))

    return x

class PriceExpector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PriceExpector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias = True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias = True)

    def forward():
        pass
    


def main():

    full_cols = ["date", "time", "init_{}", "high_{}", "low_{}", "end_{}"] + [
        str(i) for i in list(range(12))
    ]
    target_cols = ["date", "time", "end_{}"]
    merge_cols = ["date", "time"]
    value_cols = ["end_{}"]
    out_file_name = f"{data_processed}/data_aggregated.csv"
    meta_file_name = f"{data_processed}/meta.txt"


    digit_name = ["일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
    unit = ["십", "백", "천", "만", "십만"]
    # zero_point = ["영", "점"] # num2kr이 소숫점을 지원하지 않음(int only) 그래서 나중에 추가해서 다시 해보는 걸로. 
    
    joint_tokens = [i+j for j in unit for i in digit_name] 
    indiv_tokens = digit_name + unit
    # digit_only_tokens = digit_name + zero_point    

    joint_indeces = {j: i for i, j in enumerate(tuple(joint_tokens))}
    indiv_indeces = {j: i for i, j in enumerate(tuple(indiv_tokens))}

    print(joint_indeces)
    print(indiv_indeces)


    # # get df and meta, save them/
    # df, meta = agg_data(full_cols, target_cols, merge_cols)
    # df.to_csv(out_file_name, index=False)
    # with open(meta_file_name, "w") as f:
    #     for i in meta:
    #         f.write(str(i) + "\n")
    # f.close()

    # # preprocess(convert number to string)
    # df = num_to_str(out_file_name, meta_file_name, value_cols)

    

    








if __name__ == "__main__":

    main()
