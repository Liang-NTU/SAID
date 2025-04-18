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


class CLS(torch.nn.Module):
    def __init__(self, model_path: str, tokenizer, args=None, num_class=8, emb_dim=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_path)
        self.cls = nn.Linear(emb_dim, 1)
        self.loss_fct = nn.CrossEntropyLoss()

        # Get ID(s) for rqq and rqa from your special tokens
        rqq_ids = tokenizer(" ".join(utils.SPECIAL_TOKENS['qq_tokens']))['input_ids']  # e.g. [x, y, z]
        rqa_ids = tokenizer(" ".join(utils.SPECIAL_TOKENS['qa_tokens']))['input_ids']  # e.g. [a, b, c]
        word_embeddings = self.bert.get_input_embeddings().weight  # [vocab_size, emb_dim]

        # rqq_emb, rqa_emb: trainable embeddings for rqq and rqa tokens
        # shape: [prompt_len, emb_dim] if each list has length=prompt_len
        self.rqq_emb = nn.Parameter(word_embeddings[rqq_ids, :].clone(), requires_grad=True)
        self.rqa_emb = nn.Parameter(word_embeddings[rqa_ids, :].clone(), requires_grad=True)

        # An MLP to generate w1 and w2 based on the query embedding
        # so that each query can have distinct weights for rqq and rqa
        self.mlp_adapter = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)  # ensures w1 + w2 = 1
        )

        # We'll set these externally (in main) for the prompt tokens and label tokens
        self.prompt_tokens = None  # shape: [dim2]
        self.labels_tokens = None  # shape: [num_labels, dim3]

    def load_encoders(self, state_dict):
        # This loads only the "bert" weights from a state_dict (if you want partial init)
        state_dict = torch.load(state_dict)
        state_dict = {k: v for k, v in state_dict.items() if 'bert' in k}
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        print(f'missing keys: {missing_keys}')
        print(f'unexpected keys: {unexpected_keys}')

    def forward(self, input_ids, attention_mask, labels):
        """
        Standard forward pass without prompt logic.
        (Not used in your finetune() but we keep it for completeness.)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output
        logits = self.cls(pooler_output)
        return logits

    def prompts_forward(self, input_ids):
        """
        1) Create final_input by concatenating input_ids + prompt_tokens + labels_tokens
        2) Convert final_input to embeddings
        3) Obtain a query-specific (w1, w2) from MLP
        4) Combine rqq_emb and rqa_emb using w1, w2
        5) Replace the prompt slice in final_input_embeds with that combination
        6) Forward pass through BERT => classification
        """

        # === (A) Build final_input as before ===
        prompt_tokens = self.prompt_tokens      # shape = [dim2], typically a few prompt token IDs
        labels_tokens = self.labels_tokens      # shape = [num_labels, dim3]
        bs = input_ids.size(0)

        # 1) input_ids + prompt_tokens => [bs, dim1 + dim2]
        prompt_tokens_expanded = prompt_tokens.unsqueeze(0).expand(bs, -1)  
        input_prompt_concat = torch.cat([input_ids, prompt_tokens_expanded], dim=-1)  

        # 2) Expand for labels => final_input => [bs * num_labels, (dim1 + dim2 + dim3)]
        input_prompt_concat_expanded = input_prompt_concat.unsqueeze(1).expand(-1, labels_tokens.size(0), -1)
        labels_tokens_expanded = labels_tokens.unsqueeze(0).expand(bs, -1, -1)
        final_input = torch.cat([input_prompt_concat_expanded, labels_tokens_expanded], dim=-1)
        final_input = final_input.view(-1, final_input.size(-1))

        # === (B) Convert final_input to embeddings ===
        final_input_embeds = self.bert.get_input_embeddings()(final_input)

        # === (C) Obtain a query embedding for each example (no prompt, no label) to get (w1, w2) ===
        base_outputs = self.bert(input_ids=input_ids, attention_mask=(input_ids != 0).long())
        query_emb = base_outputs.pooler_output  # [bs, emb_dim]

        # MLP => [bs, 2] => each row => (w1, w2)
        w = self.mlp_adapter(query_emb)  # => [bs, 2]
        w1, w2 = w[:, 0], w[:, 1]        # => each => [bs]

        # === (D) Combine rqq_emb, rqa_emb with w1, w2 ===
        # rqq_emb, rqa_emb => [prompt_len, emb_dim]
        # We'll expand them to [bs * num_labels, prompt_len, emb_dim] so each row can have its own w1, w2
        num_labels = labels_tokens.size(0)
        batch_size_times_labels = final_input.shape[0]  # bs * num_labels

        # Expand rqq_emb => [1, prompt_len, emb_dim] => [bs, prompt_len, emb_dim] => [bs * num_labels, prompt_len, emb_dim]
        rqq_expand = self.rqq_emb.unsqueeze(0).expand(bs, -1, -1)        # [bs, prompt_len, emb_dim]
        rqa_expand = self.rqa_emb.unsqueeze(0).expand(bs, -1, -1)        # [bs, prompt_len, emb_dim]
        rqq_expand = rqq_expand.unsqueeze(1).expand(-1, num_labels, -1, -1).reshape(batch_size_times_labels, -1, rqq_expand.size(-1))
        rqa_expand = rqa_expand.unsqueeze(1).expand(-1, num_labels, -1, -1).reshape(batch_size_times_labels, -1, rqa_expand.size(-1))

        # Similarly, expand w1, w2 => [bs, 1] => [bs, num_labels, 1] => [bs*num_labels, 1]
        w1_expanded = w1.unsqueeze(1).expand(-1, num_labels).reshape(-1)  # => [bs*num_labels]
        w2_expanded = w2.unsqueeze(1).expand(-1, num_labels).reshape(-1)

        # For the final combination, we want => [bs*num_labels, 1, 1] to multiply each token's embedding
        w1_3d = w1_expanded.view(-1, 1, 1)  # [bs*num_labels, 1, 1]
        w2_3d = w2_expanded.view(-1, 1, 1)

        # => [bs*num_labels, prompt_len, emb_dim]
        combination = w1_3d * rqq_expand + w2_3d * rqa_expand

        # === (E) Identify the prompt slice in final_input_embeds and replace it ===
        prompt_len = self.prompt_tokens.size(0)
        label_len  = labels_tokens.size(-1)
        total_appended = prompt_len + label_len

        seq_len = final_input_embeds.size(1)
        prompt_start = seq_len - total_appended
        prompt_end   = prompt_start + prompt_len  # not inclusive

        final_input_embeds[:, prompt_start:prompt_end, :] = combination

        # === (F) BERT forward with replaced embeddings => classification ===
        attention_mask = (final_input != 0).long()
        outputs = self.bert(input_ids=final_input, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output  # => [bs*num_labels, emb_dim]
        logits = self.cls(pooler_output)       # => [bs*num_labels, 1]
        logits = logits.view(bs, num_labels)   # => [bs, num_labels]

        return logits


def prepare_data(df, tokenizer, batch_size=60, shuffle=True, target_level_name='first_label'):
    def prepare_tokens(df, key, tokenizer):
        tokens = tokenizer(text=df[key].tolist(),
                           padding="max_length",
                           max_length=200,
                           truncation=True,
                           return_tensors='pt')
        return tokens['input_ids']

    df = df.dropna().reset_index(drop=True)
    df = df.loc[df[target_level_name] != -1]
    input_ids = prepare_tokens(df, 'text', tokenizer)
    labels = torch.tensor(df[target_level_name].values, dtype=torch.long)
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return dataloader


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


def finetune(model, train_loader, val_loader, test_loader, optimizer, scheduler, epochs, shot_num, seed, checkpoint_name, device=torch.device('cuda')):
    best_val = 9999
    early_stop_counter = 0
    pred_df = pd.DataFrame()

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

            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
            progress_bar.update()

        model.eval()
        val_loss, predictions, true_vals = evaluate(model, val_loader, device)
        tqdm.write(f'\nEpoch {epoch}')
        loss_train_avg = loss_train_total / len(train_loader)
        tqdm.write(f'Training loss: {loss_train_avg}')

        predictions = np.argmax(predictions, axis=1)
        val_f1 = f1_score(true_vals, predictions, average='weighted')
        frac0 = np.mean(predictions == 0)

        tqdm.write(f'Validation loss: {val_loss}')
        tqdm.write(f'F1: {val_f1}')
        tqdm.write(f'frac0: {frac0}')

        # Early-stopping logic
        if val_loss < best_val:
            best_val = val_loss
            test_epoch = epoch
            early_stop_counter = 0
            checkpoint_filename = f'{checkpoint_name}/finetuned/checkpoint_shot{shot_num}_seed{seed}.pth'
            checkpoint_dir = f'{checkpoint_name}/finetuned'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_filename)
        else:
            early_stop_counter += 1
            if early_stop_counter > 9:
                break

    # Load best checkpoint
    checkpoint_filename = f'{checkpoint_name}/finetuned/checkpoint_shot{shot_num}_seed{seed}.pth'
    model.load_state_dict(torch.load(checkpoint_filename))
    model.eval()
    test_loss, predictions, true_labels = evaluate(model.to(device), test_loader, device)
    pred_df = pd.DataFrame({'prediction': np.argmax(predictions, axis=1), 'label': true_labels})

    return pred_df, test_epoch


def get_fewshot(data, num):
    fewshot = data.groupby('first_label').sample(num, random_state=42)
    fewshot = fewshot[['first_label', 'text']]
    return fewshot


def main(args):
    device = torch.device(f'cuda:{args.cuda}')

    lr = args.lr
    checkpoints = []
    print('===============checkpoints==================')
    if os.path.isdir(args.checkpoint):
        # If a directory is provided, list all checkpoint subdirectories
        checkpoints = [
            os.path.join(args.checkpoint, f)
            for f in os.listdir(args.checkpoint)
            if os.path.isdir(os.path.join(args.checkpoint, f))
        ]
        print(checkpoints)
        if not checkpoints:
            print(f"No checkpoint files found in directory: {args.checkpoint}")
            checkpoints = [None]  # Use None to fine-tune on the base model
    else:
        # If a specific file path is provided
        if os.path.exists(args.checkpoint):
            checkpoints = [args.checkpoint]
        else:
            print('No checkpoint given, finetune on base model.')
            checkpoints = [None]

    fewshots = [3, 5, 10, 20]
    level = 1
    target_level_names = {1: 'first_label', 2: 'second_label', 3: 'third_label'}
    target_level_name = target_level_names[level]
    num_class = utils.get_labels_num(level)

    random_seeds = [42, 3407, 3140, 4399, 2077]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Prepare prompt tokens
    hard_prompt_tokens = torch.tensor(tokenizer.encode(
        'is this query about the following topic?: ', 
        add_special_tokens=False
    ))
    labels_tokens = tokenizer.batch_encode_plus(
        [f"{utils.get_name(label,1)}" for label in range(num_class)],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt"
    )['input_ids']

    prompt_token_ids = tokenizer(" ".join(utils.SPECIAL_TOKENS['prompt_tokens']))['input_ids']
    soft_prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long)

    data_all = pd.read_csv(os.path.join(args.data_dir, args.finetune_file), lineterminator='\n').sample(frac=1).reset_index(drop=True)
    data_all = data_all[['text', target_level_name]]
    data_all['text'] = data_all['text'].apply(lambda x: x if len(str(x)) < 200 else str(x)[:100] + str(x)[-100:])

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

            for state_file in checkpoints:
                if state_file is None:
                    model = CLS(model_path=args.base_model, tokenizer=tokenizer, num_class=num_class)
                elif os.path.isfile(state_file):
                    model = CLS(model_path=args.base_model, tokenizer=tokenizer, num_class=num_class)
                    model.load_encoders(state_file)
                else:
                    model = CLS(model_path=state_file, tokenizer=tokenizer, num_class=num_class)
                model.to(device)

                # Choose hard or soft prompt
                if args.hard_prompt:
                    model.prompt_tokens = hard_prompt_tokens.to(device)
                else:
                    model.prompt_tokens = soft_prompt_tokens.to(device)

                model.labels_tokens = labels_tokens.to(device)

                optimizer = AdamW(model.parameters(), lr=lr)
                total_steps = len(train_loader) * args.epochs
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(total_steps * 0.1),
                    num_training_steps=total_steps
                )

                start_time = time.time()
                pred_df, test_epoch = finetune(
                    model, 
                    train_loader, 
                    val_loader, 
                    test_loader, 
                    optimizer, 
                    scheduler, 
                    args.epochs, 
                    shot_num=shot_num, 
                    seed=seed, 
                    checkpoint_name=state_file,
                    device=device
                )
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
    parser.add_argument('--base_model', type=str, default='google-bert/bert-base-uncased')
    parser.add_argument('--finetune_file', type=str, default='wildchat.csv')
    parser.add_argument('--output', type=str, default='result')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/checkpoint')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=60)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--val_size', type=int, default=2000)
    parser.add_argument('--test_size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--hard_prompt', action='store_true')

    args = parser.parse_args()
    main(args)
