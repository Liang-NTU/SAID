from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import utils
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse

def load_llm(model_path='./Llama-3.1-Nemotron-Nano-8B-v1'):
    model_path = model_path
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    return model, tokenizer


def get_results(model, tokenizer, messages, device):
    device = device # the device to load the model onto
    model.to(device)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=4096)
    generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
    # 解码生成的回复
    assistant_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return assistant_response

def get_messages(data_text):
    file_path = "prompt"
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    messages = [
        {"role": "user", "content": 
         f"{{{prompt}}}" \
         f"\n用户输入为: {{{data_text}}}, " \
         f"\n你的输出应该为以下格式: {{ \"label\": label}}" \
         f"\n只允许输出一个label，不要其他内容。" }
        ]
    return messages
    
def get_data(num):
    level = 1
    target_level_names = {1: 'first_label', 2: 'second_label', 3: 'third_label'}
    target_level_name = target_level_names[level]
    num_class = utils.get_labels_num(level)
    data_path = args.data_path
    data_all = pd.read_csv(data_path ,lineterminator='\n')
    data_all = data_all[['text', target_level_name]]
    data_all['text'] = data_all['text'].apply(lambda x: x if len(str(x)) < 200 else str(x)[:100] + str(x)[-100:])
    data_all = data_all.groupby('first_label').sample(frac=1, random_state=num).reset_index(drop=True)
    data_all.dropna()
    return data_all

def get_label_dict(label_text):
    labeling = pd.read_csv('./labeling/indices1.csv')
    label_dict = labeling.to_dict()['Unnamed: 0']
    reversed_dict = {value: key for key, value in label_dict.items()}
    label_num = reversed_dict[label_text]
    return label_num

def get_metrics(data_all):
    y_true = data_all['first_label']  # ground truth
    y_pred = data_all['infer_label']  # inferred label

    # 确保标签列存在
    if 'first_label' not in data_all.columns or 'infer_label' not in data_all.columns:
        raise ValueError("CSV 文件中缺少 'first_label' 或 'infer_label' 列")

    # 计算评估指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')  # 如果是多分类问题二分类问题
    recall = recall_score(y_true, y_pred, average='weighted')        # 如果是二分类问题
    f1 = f1_score(y_true, y_pred, average='weighted')                # 如果是二分类问题

    # 打印结果
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def main(args):
    num = args.seed
    data_all = get_data(num=num)
    data_all_subset = data_all[:]
    
    data_labels = data_all_subset['first_label']
    data_texts = data_all_subset['text']
    data_all_subset['infer_label'] = -1
    data_all_subset['infer_level'] = "无"
    model, tokenizer = load_llm(model_path=args.model_path)
    # 假设 data_texts 是一个包含文本数据的列表
    for i, text in enumerate(data_texts):
        while True:  # 使用 while 循环确保在出错时可以重新尝试
            try:
                print(f"this is the {i}th data")
                message = get_messages(text)
                result = get_results(model, tokenizer, message, device=args.device)

                # 尝试解析 JSON 结果
                result = json.loads(result)
                print(result)
                # 尝试获取标签并更新 data_all
                data_all_subset.loc[i, 'infer_label'] = get_label_dict(result['label'])
                data_all_subset.loc[i, 'infer_level'] = result['label']

                # 如果一切顺利，跳出 while 循环，继续处理下一个 i
                break

            except json.JSONDecodeError as e:
                print(f"JSONDecodeError occurred for {i}th data: {e}. Retrying...")
                print(f"data_all['text'][{i}] is wrong")
                data_all_subset.loc[i, 'infer_label'] = -1
                data_all_subset.loc[i, 'infer_level'] = "无"
                break  # 重新执行当前的 i 和 text

            except KeyError as e:
                print(f"KeyError occurred for {i}th data: {e}. Retrying...")
                print(f"data_all['text'][{i}] is wrong")
                data_all_subset.loc[i, 'infer_label'] = -1
                data_all_subset.loc[i, 'infer_level'] = "无"
                break  # 重新执行当前的 i 和 text

            except Exception as e:
                print(f"Unexpected error occurred for {i}th data: {e}. Retrying...")
                print(f"data_all['text'][{i}] is wrong")
                data_all_subset.loc[i, 'infer_label'] = -1
                data_all_subset.loc[i, 'infer_level'] = "无"
                break  # 捕获其他异常并重新执行    
    data_all_subset.to_csv(f'infer_{num}.csv', index=False)
    
    data_infer = data_all_subset
    get_metrics(data_infer)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3140)
    parser.add_argument('--model_path', type=str, default='nvidia/Llama-3.1-Nemotron-Nano-8B-v1')
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--data_path', type=str, default='./data/orig_labeled_samples.csv')
    args = parser.parse_args()
    main(args)