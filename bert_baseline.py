import numpy as np
import pandas as pd
import torch
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

def evaluate(model,dataloader,device):
    model.eval()
    loss_val_total = 0
    predictions, true_vals = [], []
    for batch in dataloader:
        inputs = {
                'input_ids':batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'labels':batch[2].to(device)
                }
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        loss = outputs.loss
        loss_val_total += loss.item()
        label_ids = inputs['labels'].to('cpu').numpy()
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    loss_val_avg = loss_val_total/len(dataloader)
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
    return loss_val_avg, predictions, true_vals


def finetune(model, train_loader, val_loader, test_loader, optimizer, scheduler, epochs, device=torch.device('cuda')):
    best_val = 9999
    pred_df = pd.DataFrame()
    model.to(device)
    
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        loss_train_total = 0
        progress_bar = tqdm(train_loader, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
        for batch in train_loader:
            inputs = {
                'input_ids':batch[0].to(device),
                'attention_mask': batch[1].to(device),
                'labels':batch[2].to(device)
                }
            outputs = model(**inputs)
            loss = outputs.loss
            loss_train_total += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
            progress_bar.update()

        model.eval()
        val_loss, predictions, true_vals = evaluate(model, val_loader, device)
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total/len(train_loader)
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
    
    return best_val, pred_df, test_epoch

def prepare_tokens(df, key, tokenizer):
    tokens = tokenizer(text=df[key].tolist(), padding="max_length", max_length=200, truncation=True)
    input_ids =  torch.tensor(tokens['input_ids']) # [num, max_len]
    attention_mask = torch.tensor(tokens['attention_mask'])
    return input_ids, attention_mask

def prepare_data(df, tokenizer, batch_size=60, shuffle=True, target_level_name='first_label'):    
    df = df.dropna().reset_index(drop=True)
    df = df.loc[df[target_level_name] != -1]
    input_ids, attention_mask = prepare_tokens(df, 'text', tokenizer)
    labels = torch.tensor(df[target_level_name].values)
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return dataloder

def get_fewshot(data, num):
    fewshot = data.groupby('first_label').sample(num)
    fewshot = fewshot[['first_label','text']]
    return fewshot

def main(args):
    device = torch.device(f'cuda:{args.cuda}')

    lr=args.lr
    emb_dim = 768

    level = 1
    target_level_names = {1: 'first_label', 2: 'second_label', 3: 'third_label'}
    target_level_name = target_level_names[level]
    num_class = utils.get_labels_num(level)

    random_seeds = [3407,3140,4399,2077,42]
    fewshots = [3, 5, 10, 20, 50]

    data_all = pd.read_csv(os.path.join(args.data_dir, args.finetune_file),lineterminator='\n').sample(frac=1).reset_index(drop=True)
    data_all = data_all[['text',target_level_name]]
    data_all['text'] = data_all['text'].apply(lambda x: x if len(str(x)) < 200 else str(x)[:100] + str(x)[-100:])
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    for seed in random_seeds:
        utils.set_seed(seed)
        traindata = get_fewshot(data_all, 100)
        otherdata = data_all.drop(traindata.index)
        valset = otherdata.sample(args.val_size)
        otherdata = otherdata.drop(valset.index)
        testset = otherdata.sample(args.test_size)
        val_loader = prepare_data(valset, tokenizer, batch_size=args.batch_size, shuffle=False, target_level_name=target_level_name)
        test_loader = prepare_data(testset, tokenizer, batch_size=args.batch_size, shuffle=False, target_level_name=target_level_name)

        for shot_num in fewshots:
            trainset = get_fewshot(traindata, shot_num)
            train_loader = prepare_data(trainset, tokenizer, batch_size=args.batch_size, shuffle=True, target_level_name=target_level_name)

            model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=num_class,ignore_mismatched_sizes=True)
            
            if args.checkpoint is not None and os.path.exists(args.checkpoint):
                if os.path.isfile(args.checkpoint): model.load_state_dict(torch.load(args.checkpoint))
                else: model.from_pretrained(args.checkpoint)
            else:
                print('no checkpoint given, finetune on base model.')
            optimizer = AdamW(model.parameters(), lr=lr)
            total_steps = len(train_loader)*args.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps*0.1, num_training_steps=total_steps)

            start_time = time.time()
            best_val, pred_df, test_epoch = finetune(model, train_loader, val_loader, test_loader, optimizer, scheduler, args.epochs, device=device)
            results = utils.calculate_metrics(pred_df)

            new_record = {
                'checkpoint': args.checkpoint,
                'fewshot': shot_num,
                'epoch': test_epoch,
                'seed': seed,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1': results['f1'],
                'frac0': np.mean(np.array(pred_df['prediction']) == 0),
                'time': time.time()-start_time
            }
            print(np.mean(np.array(pred_df['prediction']) == 0))

            date = datetime.now().strftime('%m%d')
            filename = f'./{args.output}{date}.csv'
            new_record = pd.DataFrame(new_record,index=[0])
            if os.path.exists(filename):
                new_record.to_csv(filename,mode='a+',header=False)
            else:
                new_record.to_csv(filename,mode='a+',header=True)

            del model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='google-bert/bert-base-uncased')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--finetune_file', type=str, default='wildchat.csv')
    parser.add_argument('--output', type=str, default='baseline')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--val_size', type=int, default=1500)
    parser.add_argument('--test_size', type=int, default=8000)
    parser.add_argument('--lr', type=float, default=1e-5)
    

    args = parser.parse_args()
    main(args)