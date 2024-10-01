import pandas as pd, numpy as np
from pathlib import Path
import os, shutil, random
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


def num_to_str(out_file_name, meta_file_name, value_cols, merge_cols):
    df = pd.read_csv(out_file_name)
    with open(meta_file_name, "r") as f:
        meta = [i.strip("\n").split("_")[0] for i in f.readlines()]

    meta = [i.split("_")[0] for i in meta]
    # end_{} 하나만 있으니 일단 [0]
    value_cols_file = [value_cols[0].format(i) for i in meta]
    x = df[value_cols_file].map(lambda x: int(str(x).replace(",", "")))
    x = x.map(lambda x: num2kr.num2kr(x, 1))
    x = pd.concat([df[merge_cols], x], axis = 1)
    return x


    
def make_custom_tokenizer(old_path, new_path, new_vocab):
    try: os.mkdir(new_path)
    except: pass

    # tokenizer_config.json
    gpt_special_token_idx = "50256"
    special_tokens = dic['added_tokens_decoder'][gpt_special_token_idx]['content']
    print(special_tokens)

    with open(f"{old_path}/tokenizer_config.json", "r") as f:
        dic = json.load(f)
    f.close()
    key = gpt_special_token_idx
    val = dic['added_tokens_decoder'][gpt_special_token_idx]
    dic['added_tokens_decoder'] = {len(new_vocab): val}

    with open(f"{new_path}/tokenizer_config.json", "w") as f:
        f.write(json.dumps(dic, indent = 2, ensure_ascii = False))
    f.close()
    
    # special_tokens_map.json
    shutil.copy(f"{old_path}/special_tokens_map.json", f"{new_path}/special_tokens_map.json")

    # tokenizer.json
    with open(f"{old_path}/tokenizer.json", "r") as f:
        dic = json.load(f)
    f.close()

    dic['added_tokens'] = [
        {
            key: (len(new_vocab) if key == "id" else value) \
                for key, value in dic['added_tokens'][0].items()
        }
    ] # only one token이니까 일단 0

    new_vocab_eos_added = new_vocab[len(new_vocab)] = special_tokens

    dic['model']['vocab'] = new_vocab_eos_added
    with open(f"{new_path}/tokenizer.json", "w") as f:
        f.write(json.dumps(dic, indent = 2, ensure_ascii = False))
    f.close()

    # vocabs
    with open(f"{new_path}/vocab.json", "w") as f:
        f.write(json.dumps(new_vocab_eos_added, indent = 2, ensure_ascii = False))
    f.close()

    # merge rule
    shutil.copy(f"{old_path}/merges.txt", f"{new_path}/merges.txt")
    

    return 

def convert_raw_data_to_training_data(str_file_name, training_file_name, test_file_name, meta_file_name, value_cols):
    
    with open(meta_file_name, "r") as f:
        meta = [i.strip("\n").split("_")[0] for i in f.readlines()]

    meta = [i.split("_")[0] for i in meta]
    # end_{} 하나만 있으니 일단 [0]
    value_cols_file = [value_cols[0].format(i) for i in meta]

    df = pd.read_csv(str_file_name)
    lst2d = []
    lst2d_test = []

    for i in range(len(df) - 80):
        ranges = range(i, i + 80)
        out = df[value_cols_file].loc[ranges].apply(lambda x: " ".join(x), axis = 0).tolist()
        lst2d.append(out)
    
    for i in range(len(df)-80, len(df)):
        ranges = range(i, min(i + 80, len(df)))
        out = df[value_cols_file].loc[ranges].apply(lambda x: " ".join(x), axis = 0).tolist()
        lst2d_test.append(out)

    print(len(lst2d))
    print(len(lst2d_test))

    lst1d = [i for j in lst2d for i in j]
    lst1d_test = [i for j in lst2d_test for i in j]
    
    df = pd.Series(lst1d).sample(frac = 1)
    df_test = pd.Series(lst1d_test).sample(frac = 1)

    train = df.iloc[:round(len(df) * 0.9)]
    test = df.iloc[round(len(df) * 0.9):]

    train.to_csv(training_file_name, index = False)
    test.to_csv(test_file_name, index = False)

    # with open(training_file_name+"test", "w") as f:
    #     for i in lst1d_test:
    #         f.write(f"{i}\n")
    # f.close()



def main():

    full_cols = ["date", "time", "init_{}", "high_{}", "low_{}", "end_{}"] + [
        str(i) for i in list(range(12))
    ]
    target_cols = ["date", "time", "end_{}"]
    merge_cols = ["date", "time"]
    value_cols = ["end_{}"]
    out_file_name = f"{data_processed}/data_aggregated.csv"
    meta_file_name = f"{data_processed}/meta.txt"
    str_file_name = f"{data_processed}/data_string_converted.csv"
    training_file_name = f"{data_processed}/training_data.csv"
    test_file_name = f"{data_processed}/test_data.csv"


    ###### parse data and preprocess then save
    df, meta = agg_data(full_cols, target_cols, merge_cols)
    df.to_csv(out_file_name, index=False)
    with open(meta_file_name, "w") as f:
        for i in meta:
            f.write(str(i) + "\n")
    f.close()
    
    ###### preprocess(convert number to string)
    df = num_to_str(out_file_name, meta_file_name, value_cols, merge_cols)
    df.to_csv(str_file_name, index = False)
    
    
    # ###### prepare tokenizer
    # digit_name = ["일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
    # unit = ["십", "백", "천", "만", "십만"]
    # # zero_point = ["영", "점"] # num2kr이 소숫점을 지원하지 않음(int only) 그래서 나중에 추가해서 다시 해보는 걸로. 
    
    # joint_tokens = [i+j for j in unit for i in digit_name] 
    # indiv_tokens = digit_name + unit
    # # digit_only_tokens = digit_name + zero_point    
    
    # joint_indeces = {j: i for i, j in enumerate(tuple(joint_tokens))}
    # indiv_indeces = {j: i for i, j in enumerate(tuple(indiv_tokens))}

    # checkpoint = "openai-community/gpt2"
    # tokenizer_checkpoint = "tokenizers/tokenizer_checkpoint"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenizer.save_pretrained(tokenizer_checkpoint)

    # joint_indeces_checkpoint = "tokenizers/tokenizer_joint"
    # indiv_indeces_checkpoint = "tokenizers/tokenizer_indiv"
    
    # make_custom_tokenizer(tokenizer_checkpoint, joint_indeces_checkpoint, joint_indeces)
    # make_custom_tokenizer(tokenizer_checkpoint, indiv_indeces_checkpoint, indiv_indeces)

    
    # convert data into training data format(put a single day's data(count: 78) to a list)
    convert_raw_data_to_training_data(str_file_name, training_file_name, test_file_name, meta_file_name, value_cols)
    








if __name__ == "__main__":

    main()
