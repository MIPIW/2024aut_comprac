import torch, torch.nn as nn
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, GPT2LMHeadModel, AutoConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from trl import  SFTTrainer, DataCollatorForCompletionOnlyLM
import evaluate
from functools import partial
import numpy as np
from datetime import datetime
import os, json
from transformers import set_seed
from argparse import ArgumentParser
from pathlib import Path

weight_path = str(Path().resolve().parent.parent) + os.sep + "weights"
data_preprocess = str(Path().resolve().parent.parent) + os.sep + "data/data_it_processed"

set_seed(42) 



def set_trainer(model, tokenizer, dataset, output_dir, instruction, args):
    
    accuracy = evaluate.load('accuracy')

    def metric(eval_pred, func):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis = -1) # (batch, sequence lenagh, hidden_state)
        filters = labels != -100

        predictions = predictions[filters]
        labels = labels[filters]
        return func.compute(predictions = predictions, references = labels)
    
    # Construct a template as an input for fine-tuning
    def preprocess(data, instruction):
        output_texts = []
    
        for i in range(len(data['0'])):
            output_texts.append(f"# {instruction} % {data['0'][i]}")
        
            
        return output_texts
    
    
    # def tokenize_func(examples, instruction):
    #     return tokenizer(f"{instruction} {examples['0']}", truncation=True, padding=True)

    # training_data = dataset['train'].map(preprocess, batched=True, num_proc = 4)
    # valid_data = dataset['valid'].map(preprocess, batched=True, num_proc = 4)
    # test_data = dataset['test'].map(preprocess, batched=True, num_proc = 4)

    # training_data = training_data.remove_columns(['0'])
    # valid_data = valid_data.remove_columns(['0'])
    # test_data = test_data.remove_columns(['0'])

    training_data = dataset['train']
    valid_data = dataset['valid']
    test_data = dataset['test']
    od = output_dir + os.sep + datetime.strftime(datetime.now(), "%m-%d-%H-%M-%S")
    try: os.mkdir(od)
    except: pass

    trainingarguments = TrainingArguments(
        do_train = True,    
        output_dir = od,                         
        eval_strategy = "steps", # necessary: change to step
        save_strategy = "steps",                         
        eval_steps = 0.25, # necessary: set step
        save_steps = 0.25,
        save_total_limit = 1,
        load_best_model_at_end = True, # necessary: EarlyStoppingCallBack하려면 True여야 함
        metric_for_best_model = "accuracy",
        greater_is_better = True, # necessary: higher metric results better performance # default = True when metric_for_best_model is set
        num_train_epochs = 4,
        seed = 42,
        per_device_train_batch_size = 1024,
        per_device_eval_batch_size = 1024,

        # eval_accumulation_steps = 50,
        learning_rate = args.lr,
        weight_decay = args.decay,
        remove_unused_columns = True
    )


    with open(od+ os.sep + "trainingargs.json", "w") as f: 
        f.write(json.dumps(trainingarguments.to_dict(), indent = 2, ensure_ascii = False))
    f.close()

    response_template = f"# {instruction}"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        trainingarguments,
        train_dataset=training_data,
        eval_dataset=valid_data,
        dataset_text_field="0",
        # formatting_func = partial(preprocess, instruction = instruction),
        compute_metrics=partial(metric, func = accuracy),
        # data_collator = collator,
        tokenizer=tokenizer)

    # trainer = Trainer(
    #     model = model,
    #     args = trainingarguments,
    #     tokenizer = tokenizer,
    #     train_dataset = training_data,
    #     eval_dataset = valid_data,
    #     data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    #     compute_metrics = partial(metric, func = accuracy)
    # )

    return trainer

def main():

    parser = ArgumentParser()
    parser.add_argument("--lr", type = float, required = True)
    parser.add_argument("--decay", type = float, required = True)
    args = parser.parse_args()

    print(args.lr, args.decay)

    
    
    data_path = {"train": f"{data_preprocess}/train_data_no_signs.csv", "valid": f"{data_preprocess}/valid_data_no_signs.csv", "test": f"{data_preprocess}/test_data_no_signs.csv"}
    dataset = load_dataset("csv", data_files = data_path)



    output_dir_indiv = f"{weight_path}/model_indiv_contTrained"
    output_dir_joint = f"{weight_path}/model_joint_contTrained"    

    checkpoint_indiv = "/home/hyohyeongjang/2024aut_comprac/weights/model_indiv/10-16-12-13-09/checkpoint-450"
    checkpoint_joint = "/home/hyohyeongjang/2024aut_comprac/weights/model_joint/10-15-02-51-17/checkpoint-50"

    instruction_indiv = "일 < 일 < 이 < 이 < 삼 < 삼 < 사 < 사 < 오 < 오 < 육 < 육 < 칠 < 칠 < 팔 < 팔 < 구 < 구 < 십 < 십 < 백 < 백 < 천 < 천 < 만 < 만 < 십만 < 십만 > 만 > 천 > 백 > 십 > 구 > 팔 > 칠 > 육 > 오 > 사 > 삼 > 이 > 일"
    instruction_joint = "일 < 일 < 이 < 이 < 삼 < 삼 < 사 < 사 < 오 < 오 < 육 < 육 < 칠 < 칠 < 팔 < 팔 < 구 < 구 < 일십 < 일십 < 이십 < 이십 < 삼십 < 삼십 < 사십 < 사십 < 오십 < 오십 < 육십 < 육십 < 칠십 < 칠십 < 팔십 < 팔십 < 구십 < 구십 < 일백 < 일백 < 이백 < 이백 < 삼백 < 삼백 < 사백 < 사백 < 오백 < 오백 < 육백 < 육백 < 칠백 < 칠백 < 팔백 < 팔백 < 구백 < 구백 < 일천 < 일천 < 이천 < 이천 < 삼천 < 삼천 < 사천 < 사천 < 오천 < 오천 < 육천 < 육천 < 칠천 < 칠천 < 팔천 < 팔천 < 구천 < 구천 < 일만 < 일만 < 이만 < 이만 < 삼만 < 삼만 < 사만 < 사만 < 오만 < 오만 < 육만 < 육만 < 칠만 < 칠만 < 팔만 < 팔만 < 구만 < 구만 < 일십만 < 일십만 > 구만 > 팔만 > 칠만 > 육만 > 오만 > 사만 > 삼만 > 이만 > 일만 > 구천 > 팔천 > 칠천 > 육천 > 오천 > 사천 > 삼천 > 이천 > 일천 > 구백 > 팔백 > 칠백 > 육백 > 오백 > 사백 > 삼백 > 이백 > 일백 > 구십 > 팔십 > 칠십 > 육십 > 오십 > 사십 > 삼십 > 이십 > 일십 > 구 > 팔 > 칠 > 육 > 오 > 사 > 삼 > 이 > 일"

    checkpoint = checkpoint_indiv
    output_dir = output_dir_indiv
    instruction = instruction_indiv

    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_tokens(["<", ">", "#", "%"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    trainer_indiv = set_trainer(model, tokenizer, dataset, output_dir, instruction, args)
    trainer_indiv.train()
    # print(trainer_indiv.evaluate())



    checkpoint = checkpoint_joint
    output_dir = output_dir_joint
    instruction = instruction_joint

    model = GPT2LMHeadModel.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.add_tokens(["<", ">", "#", "%"], special_tokens = True)
    model.resize_token_embeddings(len(tokenizer))
    trainer_joint = set_trainer(model, tokenizer, dataset, output_dir, instruction, args)
    trainer_joint.train()
    # print(trainer_joint.evaluate())

    # make new finetune data and run preprocess
    # convert data _ to either < or >
    # append instruction in front of text
    # train

 

if __name__ == "__main__":
    main()

