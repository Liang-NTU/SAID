import numpy as np
import torch
import pandas as pd
# import polars as pl

from datasets import Dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

import os
import math
import random
import logging
import utils
import argparse

def preprocess_function(examples):
    q1_texts = examples['question']
    a_texts = examples['answer_text']
    q2_texts = examples['pos_question']

    qq_examples = [q1 + " " + " ".join(utils.SPECIAL_TOKENS['qq_tokens']) + " " + q2 for q1,q2 in zip(q1_texts, q2_texts)]
    qa_examples = [q1 + " " + " ".join(utils.SPECIAL_TOKENS['qa_tokens']) + " " + a for q1,a in zip(q1_texts, a_texts)]

    return {'qq_texts': qq_examples, 'qa_texts': qa_examples}


def main(args):
    df = pd.read_csv(args.train_file, lineterminator='\n')
    columns = ['question', 'answer_text', 'pos_question','neg_question','neg_answer']
    df = df[columns]
    df = df.dropna()
    dataset = Dataset.from_pandas(df)
    if args.query_only:
        full_dataset = Dataset.from_dict({'text': dataset['question']})
    else:
        full_dataset = dataset.map(preprocess_function, batched=True, remove_columns=columns)
        qq_dataset = Dataset.from_dict({'text': full_dataset['qq_texts']})
        qa_dataset = Dataset.from_dict({'text': full_dataset['qa_texts']})
        full_dataset = concatenate_datasets([qq_dataset, qa_dataset])

    for seed in [42,3407,3140,4399,2077]:
        full_dataset = full_dataset.shuffle(seed=seed)

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForMaskedLM.from_pretrained(args.base_model)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=args.mlm_ratio
        )

        def tokenize_function(examples):
            return tokenizer(examples['text'], padding="max_length", max_length=512, truncation=True)

        tokenized_datasets = full_dataset.map(tokenize_function, batched=True)

        if args.output is not None:
            output_dir = args.output
        else:
            output_dir = f'./{args.base_model.split("/")[-1]}{"_query" if args.query_only else ""}{seed}'

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="no",
            save_strategy='epoch',
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            report_to=[],
            load_best_model_at_end=False,
            save_total_limit=args.save_limit,
            seed=seed
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets,
            data_collator=data_collator
        )

        trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='google-bert/bert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--train_file', type=str, default='./data/wildchat_pretrain.csv')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--query_only', type=bool, default=False)
    parser.add_argument('--mlm_ratio', type=float, default=0.15)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--save_limit', type=int, default=1)
    args = parser.parse_args()
    main(args)
