import torch, torch.nn as nn
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, GPT2LMHeadModel, AutoConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import evaluate
from functools import partial
import numpy as np
from datetime import datetime
import os, json
from transformers import set_seed
from argparse import ArgumentParser
from pathlib import Path

data_raw = str(Path().resolve().parent.parent) + os.sep + "data/data_raw"
data_processed = str(Path().resolve().parent.parent) + os.sep + "data/data_raw_processed"
tokenizer_path = str(Path().resolve().parent.parent) + os.sep + "tokenizers"
weight_path = str(Path().resolve().parent.parent) + os.sep + "weights"

set_seed(42) 

class PriceExpector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PriceExpector, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias = True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias = True)

    def forward(self, inputs):
        x = self.embedding(inputs['input_ids'])

        x = self.fc1(x)
        x = self.fc2(x)
        return x


def set_trainer(model, tokenizer, dataset, output_dir, args):
    
    accuracy = evaluate.load('accuracy')

    def metric(eval_pred, func):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis = -1) # (batch, sequence lenagh, hidden_state)
        filters = labels != -100

        predictions = predictions[filters]
        labels = labels[filters]
        return func.compute(predictions = predictions, references = labels)
    
    def tokenize_func(examples):
        return tokenizer(examples['0'], truncation=True, padding=True)

    training_data = dataset['train'].map(tokenize_func, batched=True, num_proc = 4)
    valid_data = dataset['valid'].map(tokenize_func, batched=True, num_proc = 4)
    test_data = dataset['test'].map(tokenize_func, batched=True, num_proc = 4)

    training_data = training_data.remove_columns(['0'])
    valid_data = valid_data.remove_columns(['0'])
    test_data = test_data.remove_columns(['0'])

    od = output_dir + os.sep + datetime.strftime(datetime.now(), "%m-%d-%H-%M-%S")
    try: os.mkdir(od)
    except: pass
    trainingarguments = TrainingArguments(
        do_train = True,    
        output_dir = od,                         
        evaluation_strategy = "steps", # necessary: change to step
        save_strategy = "steps",                         
        eval_steps = 50, # necessary: set step
        save_steps = 50,
        save_total_limit = 1,
        load_best_model_at_end = True, # necessary: EarlyStoppingCallBack하려면 True여야 함
        metric_for_best_model = "accuracy",
        greater_is_better = True, # necessary: higher metric results better performance # default = True when metric_for_best_model is set
        num_train_epochs = 10,
        seed = 42,
        per_device_train_batch_size = 1024,
        per_device_eval_batch_size = 1024,

        # eval_accumulation_steps = 50,
        learning_rate = args.lr,
        weight_decay = args.decay,
        remove_unused_columns = False
    )


    with open(od+ os.sep + "trainingargs.json", "w") as f: 
        f.write(json.dumps(trainingarguments.to_dict(), indent = 2, ensure_ascii = False))
    f.close()
    

    trainer = Trainer(
        model = model,
        args = trainingarguments,
        tokenizer = tokenizer,
        train_dataset = training_data,
        eval_dataset = valid_data,
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        compute_metrics = partial(metric, func = accuracy)
    )

    return trainer

def set_config(ggangtong_model_checkpoint, tokenizer):
    print(len(tokenizer) - 1)
    
    config = AutoConfig.from_pretrained(
        ggangtong_model_checkpoint,
        vocab_size = len(tokenizer),
        n_ctx = 1024,
        bos_token_id = tokenizer.bos_token_id,
        eos_token_id = tokenizer.eos_token_id,
        n_embd = 128,
        n_head = 4,
        n_layer = 4,
        n_positions = 1024, 


    )

    return config


def main():

    parser = ArgumentParser()
    parser.add_argument("--lr", type = float, required = True)
    parser.add_argument("--decay", type = float, required = True)
    args = parser.parse_args()

    print(args.lr, args.decay)

    training_data_path = f"{data_processed}/training_data.csv"
    test_data_path = f"{data_processed}/test_data.csv"
    data_path = {"train": f"{data_processed}/training_data.csv", "valid": f"{data_processed}/valid_data.csv", "test": f"{data_processed}/test_data.csv"}
    dataset = load_dataset("csv", data_files = data_path)

    ggangtong_model_checkpoint = "openai-community/gpt2"    

    
    indiv_indeces_checkpoint = f"{tokenizer_path}/tokenizer_indiv_jaeyoon"
    output_dir_indiv = f"{weight_path}/model_indiv"
    tokenizer_indiv = AutoTokenizer.from_pretrained(indiv_indeces_checkpoint)
    tokenizer_indiv.add_special_tokens({"pad_token": "<pad>"}) # Llama3 doesn't have pad_token
    model_config_indiv = set_config(ggangtong_model_checkpoint, tokenizer_indiv)
    model_indiv = GPT2LMHeadModel(model_config_indiv)
    trainer_indiv = set_trainer(model_indiv, tokenizer_indiv, dataset, output_dir_indiv, args)
    trainer_indiv.train()


    joint_indeces_checkpoint = f"{tokenizer_path}/tokenizer_joint_jaeyoon"    
    output_dir_joint = f"{weight_path}/model_joint"    
    tokenizer_joint = AutoTokenizer.from_pretrained(joint_indeces_checkpoint)
    tokenizer_joint.add_special_tokens({"pad_token": "<pad>"}) # Llama3 doesn't have pad_token
    model_config_joint = set_config(ggangtong_model_checkpoint, tokenizer_joint)
    model_joint = GPT2LMHeadModel(model_config_joint)
    trainer_joint = set_trainer(model_joint, tokenizer_joint, dataset, output_dir_joint, args)
    trainer_joint.train()


if __name__ == "__main__":
    main()

    

