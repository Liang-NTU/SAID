import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import random

LABELS_DIR = 'labeling'

name1_pd = pd.read_csv(os.path.join(LABELS_DIR, 'indices1.csv'), encoding='utf-8', index_col=1)
name1_dict = name1_pd.to_dict()['Unnamed: 0']
name_dicts = {1:name1_dict}

name_to_label_1_dict = dict(zip(name1_dict.values(), name1_dict.keys()))
name_to_label_dicts = {1:name_to_label_1_dict}

layer2labels_num = {1:max(name1_dict.keys())+1}

SPECIAL_TOKENS = {"qq_tokens": [f"[unused{i}]" for i in range(1,4)], 
    "qa_tokens": [f"[unused{i}]" for i in range(4,7)],
    "ql_tokens": [f"[unused{i}]" for i in range(8,11)],
    "prompt_tokens": [f"[unused{i}]" for i in range(11,14)]
    }

def split_df(data, ratio=0.8):
    data = data.sample(frac=1).reset_index(drop=True)
    train_data = data.iloc[:int(len(data)*ratio)]
    test_data = data.iloc[int(len(data)*ratio):]
    return train_data, test_data

def get_label(name, level=1):
    return name_to_label_dicts[level][name]

def get_name(label, level=1):
    return name_dicts[level][label]

def get_labels_num(level=1):
    return layer2labels_num[level]

def convert2tokens(texts, tokenizer,head = 60, tail = 60, special_token=True):
    encoded_data = {"input_ids": [], "attention_mask": []}
    max_length = head + tail
    for text in texts:
        # tokenization
        tokens = tokenizer.tokenize(text)
        if special_token:
            # truncate
            if len(tokens) > max_length - 1:
                tokens = tokens[:head]
                if tail > 1:
                    tokens += tokens[(-tail+1):]
            # special tokens
            tokens = ['[CLS]'] + tokens #+ ['[SEP]']
        else:
            # truncate
            if len(tokens) > max_length:
                tokens = tokens[:head] + tokens[-tail:]
        # convert to ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # padding
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        encoded_data['input_ids'].append(input_ids)
        encoded_data['attention_mask'].append(input_mask)
    encoded_data['input_ids'] = torch.tensor(encoded_data['input_ids'])
    encoded_data['attention_mask'] = torch.tensor(encoded_data['attention_mask'])
    return encoded_data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_metrics(pred_df):
    y_true = pred_df['label']
    y_pred = pred_df['prediction']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return results
