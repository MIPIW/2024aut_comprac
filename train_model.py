import torch, torch.nn as nn
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

class PriceExpector(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(PriceExpector, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias = True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias = True)

    def forward(self, inputs):
        pass

def main():

    joint_indeces_checkpoint = "tokenizers/tokenizer_joint/tokenizer.json"
    indiv_indeces_checkpoint = "tokenizers/tokenizer_indiv/tokenizer.json"
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # tokenizer = tokenizer.from_file(joint_indeces_checkpoint)
    tokenizer = tokenizer.from_file(indiv_indeces_checkpoint)


    training_data_path = "data_processed/training_data.csv"
    test_data_path = "data_processed/test_data.csv"
    data_path = {"train": "data_processed/training_data.csv", "test": "data_processed/test_data.csv"}

    # training_data = load_dataset("csv", training_data_path)
    dataset = load_dataset("csv", data_files = data_path)
    print(dataset)



if __name__ == "__main__":
    main()

    

