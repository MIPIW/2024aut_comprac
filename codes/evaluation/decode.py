from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import torch

import pandas as pd
from datasets import load_dataset
import re, os 
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()

weight_path = str(Path().resolve().parent.parent) + os.sep + "weights"
data_it_processed = str(Path().resolve().parent.parent) + os.sep + "data/data_it_processed"

# pretrain/preproces.py로  preprocess된 데이터를 한 번 더 preprocess하는 모듈
def str_to_num(list_x):
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
    
    return list_out

def set_test_trainer(model, tokenizer, test_dataset):

    trainingArgument = TrainingArguments()
    trainer = Trainer(model = model, 
                      tokenizer = tokenizer,
                      eval_dataset=test_dataset)

    return trainer

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint_indiv_it = "/home/hyohyeongjang/2024aut_comprac/weights/model_indiv_finetuned/10-18-10-02-02/checkpoint-5"
    checkpoint_joint_it = "/home/hyohyeongjang/2024aut_comprac/weights/model_joint_finetuned/10-18-10-02-57/checkpoint-5"
    checkpoint_indiv = "/home/hyohyeongjang/2024aut_comprac/weights/model_indiv/10-16-12-13-09/checkpoint-450"
    checkpoint_joint = "/home/hyohyeongjang/2024aut_comprac/weights/model_joint/10-15-02-51-17/checkpoint-50"


    # indiv
    checkpoint = checkpoint_indiv
    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    test_dataset_sign = load_dataset("csv", data_files={"test": f"{data_it_processed}/test_data_with_signs.csv"})
    test_dataset_no_sign = load_dataset("csv", data_files={"test": f"{data_it_processed}/test_data_no_signs.csv"})

    # trainer = set_test_trainer(model, tokenizer, test_dataset_no_sign)
    # trainer.evaluate()
    # trainer = set_test_trainer(model, tokenizer, test_dataset_sign)
    # trainer.evaluate()
    def tokenize_func(text, tokenizer = tokenizer):
        out = []
        for i in range(len(text['0'])):
            out.append(tokenizer(text['0'], return_tensors="pt", padding= "max_length", truncation= True, max_length=1024))
        return {"0": out} # 왜 안 되지 흠 힝
    
    print(test_dataset_sign)
    
    test_dataset_sign_tokenized = test_dataset_sign['test'].map(tokenize_func, batched = True, num_proc=4)
    # test_dataset_no_sign_tokenized = test_dataset_no_sign['test'].map(tokenizer, bathced = True)

    print(test_dataset_sign_tokenized['0'][0])

    generated_ids = model.generate(**test_dataset_sign_tokenized['0'][0], max_new_tokens=128, do_sample=True)
    print(tokenizer.convert_ids_to_tokens(generated_ids))
    # out = tokenizer.batch_decode(generated_ids)[0]
    # print(str_to_num(out))

if __name__ == "__main__":
    main()