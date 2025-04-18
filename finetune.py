import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, AutoModel
from transformers import get_scheduler, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_

import os
import random
import logging
import argparse
import time
from datetime import datetime
import utils

least_epochs_dict = {3: 25, 5: 20, 10: 15, 20: 10, 50: 5}


class CLS(torch.nn.Module):
    def __init__(self, model_path='./pretrained', args=None, num_class=8, emb_dim=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.cls = nn.Linear(emb_dim, 1)
        self.loss_fct = nn.CrossEntropyLoss()

    def load_encoders(self, state_dict):
        state_dict = torch.load(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if 'bert' in k}
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print(f'missing keys: {missing_keys}')
        print(f'unexpected keys: {unexpected_keys}')

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        logits = self.cls(pooler_output)
        return logits

    def prompts_forward(self, input_ids):
        # input_ids: [bs, dim1]
        # prompt_tokens: [dim2]
        # labels_tokens: [num_labels, dim3]
        prompt_tokens = self.prompt_tokens
        labels_tokens = self.labels_tokens
        bs = input_ids.size(0)

        prompt_tokens_expanded = prompt_tokens.unsqueeze(0).expand(bs, -1)  # [bs, dim2]
        input_prompt_concat = torch.cat([input_ids, prompt_tokens_expanded], dim=-1)  # [bs, dim1 + dim2]
        input_prompt_concat_expanded = input_prompt_concat.unsqueeze(1).expand(-1, labels_tokens.size(0),
                                                                               -1)  # [bs, num_labels, dim1 + dim2]
        labels_tokens_expanded = labels_tokens.unsqueeze(0).expand(bs, -1, -1)  # [bs, num_labels, dim3]
        final_input = torch.cat([input_prompt_concat_expanded, labels_tokens_expanded],
                                dim=-1)  # [bs, num_labels, dim1 + dim2 + dim3]
        final_input = final_input.view(-1, final_input.size(-1))  # [bs * num_labels, dim1 + dim2 + dim3]

        outputs = self.bert(input_ids=final_input, attention_mask=(final_input != 0).long())
        # pooler_output = outputs.pooler_output
        last_hidden_state = outputs.last_hidden_state
        # mean pooled output = torch.mean(last hidden state, dim=1)
        mean_pooled_output = last_hidden_state[:,0,:]
        pooler_output = mean_pooled_output
        logits = self.cls(pooler_output)  # logits: [bs * num_labels, 1]
        logits = logits.view(bs, labels_tokens.size(0))
        return logits


def prepare_data(df, tokenizer, batch_size=60, shuffle=True, target_level_name='first_label'):
    def prepare_tokens(df, key, tokenizer):
        tokens = tokenizer(text=df[key].tolist(), padding="max_length", max_length=200, truncation=True,
                           return_tensors='pt')
        input_ids = tokens['input_ids']  # [num, max_len]
        return input_ids

    df = df.dropna().reset_index(drop=True)
    df = df.loc[df[target_level_name] != -1]
    input_ids = prepare_tokens(df, 'question', tokenizer)
    labels = torch.tensor(df[target_level_name].values)
    dataset = TensorDataset(input_ids, labels)
    dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return dataloder


def evaluate(model, dataloader, device=torch.device('cuda')):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in tqdm(dataloader, leave=False):
        input_ids, labels = (b.to(device) for b in batch)
        with torch.no_grad():
            logits = model.prompts_forward(input_ids)
        loss = model.loss_fct(logits, labels)
        loss_val_total += loss.item()
        label_ids = labels.to('cpu').numpy()
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    loss_val_avg = loss_val_total / len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


def finetune(model, train_loader, val_loader, test_loader, optimizer, scheduler, epochs, least_epoch=3,
             device=torch.device('cuda')):
    best_val = 9999
    pred_df = pd.DataFrame()
    early_stop_counter = 0
    for epoch in range(epochs):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(train_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in train_loader:
            input_ids, labels = (b.to(device) for b in batch)
            logits = model.prompts_forward(input_ids)
            loss = model.loss_fct(logits, labels)
            loss_train_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})
            progress_bar.update()

        if epoch > least_epoch:
            model.eval()
            val_loss, predictions, true_vals = evaluate(model, val_loader, device)
            tqdm.write(f'\nEpoch {epoch}')
            loss_train_avg = loss_train_total / len(train_loader)
            tqdm.write(f'Training loss: {loss_train_avg}')
            predictions = np.argmax(predictions, axis=1)
            val_f1 = f1_score(true_vals, predictions, average='weighted')
            frac0 = np.mean(np.array(predictions) == 0)
            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'F1: {val_f1}')
            tqdm.write(f'frac0: {frac0}')

            if val_loss < best_val:
                best_val = val_loss
                test_epoch = epoch
                early_stop_counter = 0
                torch.save(model.state_dict(), 'checkpoint.pth')
            else:
                early_stop_counter += 1
                if early_stop_counter > 4:
                    break

    model.load_state_dict(torch.load('checkpoint.pth'))
    model.eval()
    test_loss, predictions, true_labels = evaluate(model.to(device), test_loader, device)
    pred_df = pd.DataFrame({'prediction': np.argmax(predictions, axis=1), 'label': true_labels})

    return pred_df, test_epoch


def get_fewshot(data, num):
    data = data.sample(frac=1).reset_index(drop=True)
    fewshot = data.groupby('first_label').sample(num)
    fewshot = fewshot[['first_label', 'question']]
    fewshot.reset_index(False)
    return fewshot


def main(args):
    device = torch.device(f'cuda:{args.cuda}')

    lr = args.lr
    checkpoints = [args.checkpoint]
    if not os.path.exists(args.checkpoint):
        print('no checkpoint given, finetune on base model.')
        checkpoints = [None]
    fewshots = [3, 5, 10, 20, 50]
    level = 1
    target_level_names = {1: 'first_label', 2: 'second_label', 3: 'third_label'}
    target_level_name = target_level_names[level]
    num_class = utils.get_labels_num(level)

    random_seeds = [42, 3407, 3140, 4399, 2077]
    save_columns = ['checkpoint', 'seed', 'acc']

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    hard_prompt_tokens = torch.tensor(
        tokenizer.encode('is this query about the following topic?: ', add_special_tokens=False))
    soft_prompt_tokens = torch.tensor(
        tokenizer.encode(" ".join(utils.SPECIAL_TOKENS['prompt_tokens']), add_special_tokens=False))
    labels_tokens = \
    tokenizer.batch_encode_plus([f"{utils.get_name(label, 1)}" for label in range(num_class)], padding=True,
                                add_special_tokens=False, return_tensors="pt")['input_ids']

    data_all = pd.read_csv(os.path.join(args.data_dir, args.finetune_file), lineterminator='\n').sample(frac=1)
    if args.q_colname != 'question' and args.q_colname in data_all.columns:
        data_all = data_all.rename(columns={'text': 'question'})
    data_all = data_all[['question', target_level_name]]
    data_all['question'] = data_all['question'].apply(
        lambda x: x if len(str(x)) < 200 else str(x)[:100] + str(x)[-100:])

    for seed in random_seeds:
        utils.set_seed(seed)
        valset = data_all.sample(args.val_size)
        testset = data_all.sample(args.test_size)
        val_loader = prepare_data(valset, tokenizer, batch_size=args.batch_size, shuffle=False,
                                  target_level_name=target_level_name)
        test_loader = prepare_data(testset, tokenizer, batch_size=args.batch_size, shuffle=False,
                                   target_level_name=target_level_name)

        for shot_num in fewshots:
            trainset = get_fewshot(data_all, shot_num)
            train_loader = prepare_data(trainset, tokenizer, batch_size=args.batch_size, shuffle=True,
                                        target_level_name=target_level_name)

            for state_file in checkpoints:
                # model = CLS(model_path=args.model_path,num_class=num_class)
                if state_file is None:
                    model = CLS(model_path=args.base_model, num_class=num_class)
                elif os.path.isfile(state_file):
                    model = CLS(model_path=args.base_model, num_class=num_class)
                    model.load_encoders(state_file)
                else:
                    model = CLS(model_path=state_file, num_class=num_class)
                model.to(device)
                if args.hard_prompt:
                    model.prompt_tokens = hard_prompt_tokens.to(device)
                else:
                    model.prompt_tokens = soft_prompt_tokens.to(device)
                model.labels_tokens = labels_tokens.to(device)

                optimizer = AdamW(model.parameters(), lr=lr)
                total_steps = len(train_loader) * args.epochs
                scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps * 0.1,
                                                            num_training_steps=total_steps)

                start_time = time.time()
                pred_df, test_epoch = finetune(model, train_loader, val_loader, test_loader, optimizer, scheduler,
                                               args.epochs, least_epoch=least_epochs_dict[shot_num], device=device)
                results = utils.calculate_metrics(pred_df)

                new_record = {
                    'checkpoint': state_file,
                    'fewshot': shot_num,
                    'epoch': test_epoch,
                    'seed': seed,
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1': results['f1'],
                    'time': time.time() - start_time
                }
                date = datetime.now().strftime('%m%d')
                filename = f'./{args.output}{date}.csv'
                new_record = pd.DataFrame(new_record, index=[0])
                if os.path.exists(filename):
                    new_record.to_csv(filename, mode='a+', header=False)
                else:
                    new_record.to_csv(filename, mode='a+', header=True)

                del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='basemodel1')
    # parser.add_argument('--base_model', type=str, default='hfl/chinese-roberta-wwm-ext')
    parser.add_argument('--q_colname', type=str, default='text')
    parser.add_argument('--a_colname', type=str, default='answer_text')
    parser.add_argument('--finetune_file', type=str, default='gpt_data_new_unix.csv')
    parser.add_argument('--output', type=str, default='prompts_finetune')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--cuda', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--val_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--hard_prompt', action='store_true')

    args = parser.parse_args()
    main(args)