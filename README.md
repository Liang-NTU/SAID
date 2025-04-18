# SAID

## environment
python>=3.8

```
pip install -r requirements.txt
```

## data
The wildchat data is collected from [wildchat](https://huggingface.co/datasets/allenai/WildChat). We labeled the data by GPT-4 and human and release it to public for academic use.
Please download from https://entuedu-my.sharepoint.com/:u:/g/personal/liang012_e_ntu_edu_sg/ERL3BUEJIGlAuuNTetOOezEBIeCUnUdwr2rDIVYTrSpsfQ?e=m9ORij 
The gpt_data_new_unix data is used to finetune the pretrained model.
```
unzip data_en.zip
```
## pretrain
pretrain.py: 
```
python pretrain.py [--base_model google-bert/bert-base-uncased] [--train_file ./data/wildchat_pretrain_unix.csv] [--mlm_ratio 0.15] [--batch_size 60] [--epochs 3] [--query_only False] [--output output_dir]
```

## finetune 
finetune.py: finetune model which loads checkpoint if able with given data. test results would be saved in output file. 
```
python finetune.py [--base_model google-bert/bert-base-uncased] [--finetune_file wildchat.csv] [--output output_file_name] [--checkpoint pretrained_model_to_load] [--data_dir data_dir] [--epochs 70] [--lr 1e-5] [--batch_size 5] [--hard_prompt]
```
> notice: it will finetune on base_model if you don't give a checkpoint.
> checkpoint can be state_file loaded by torch.load() or a directory loaded by method AutoModel.from_pretrained().


finetune_Query_adaptive.py: finetune the pretrained model with a query adaptive module.
```
python finetune_Query_adaptive.py [--base_model google-bert/bert-base-uncased] [--finetune_file wildchat.csv] [--output output_file_name] [--checkpoint pretrained_model_to_load] [--data_dir data_dir] [--epochs 70] [--lr 1e-5] [--batch_size 5]
```

baseline_LLM.py: LLM baseline script
```
python baseline_LLM.py [--seed 3140] [--model_path 'nvidia/Llama-3.1-Nemotron-Nano-8B-v1'] [--device 'cuda:3'] [data_path './data/orig_labeled_samples.csv']
```
