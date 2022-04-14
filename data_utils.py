import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def encode_data(dataset, dataset_name, tokenizer, max_seq_length=128):
    """
    Featurizes the dataset into input IDs and attention masks for input into a
    transformer-style model.
    """
    if "snli" in dataset_name:
        premise = dataset.premise.values.tolist()
        hypothesis = dataset.hypothesis.values.tolist()
    elif "imdb" in dataset_name or "ag" in dataset_name:
        text = dataset.text.values.tolist()
    
    input_ids = torch.empty((len(dataset), max_seq_length), dtype=torch.long)
    attention_mask = torch.empty((len(dataset), max_seq_length), dtype=torch.long)
    for i in tqdm(range(len(dataset))):
        if "snli" in dataset_name:
            sequence = tokenizer.encode_plus(str(premise[i]), str(hypothesis[i]), return_tensors="pt",
                                             max_length=max_seq_length, padding="max_length", truncation=True)
        elif "imdb" in dataset_name or "ag" in dataset_name:
            sequence = tokenizer.encode_plus(str(text[i]), return_tensors="pt", max_length=max_seq_length, 
                                             padding="max_length", truncation=True)
        else:
            print('Invalid dataset name')
            sys.exit(0)
        input_ids[i] = sequence['input_ids'][0].long()
        attention_mask[i] = sequence['attention_mask'][0].long()

    return input_ids, attention_mask

def extract_labels(dataset):
    """
    Converts labels into numerical labels.
    """
    return dataset.label.astype(int).values.tolist()

class SequenceDataset(Dataset):
    """
    A torch.utils.data.Dataset wrapper for the Sequence dataset.
    """

    def __init__(self, dataframe, dataset_name, tokenizer, max_seq_length=256):
        self.input_ids, self.attention_masks = encode_data(dataframe, dataset_name, tokenizer, max_seq_length)
        self.label_list = extract_labels(dataframe)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        return dict({'input_ids': self.input_ids[i], 'attention_mask':self.attention_masks[i], 
                     'labels': self.label_list[i]})
