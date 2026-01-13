import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset


def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='EleutherAI/pythia-12b')
parser.add_argument(
    "--dataset",
    type=str,
    default="WikiMIA_length32",
    choices=[
        "WikiMIA_length32",
        "WikiMIA_length64",
        "WikiMIA_length128",
        "WikiMIA_length32_paraphrased",
        "WikiMIA_length64_paraphrased",
        "WikiMIA_length128_paraphrased",
    ],
)
parser.add_argument('--half', action='store_true')
parser.add_argument('--int8', action='store_true')
args = parser.parse_args()

def load_model(name):
    if args.int8:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            quantization_config=bnb_config,
            dtype="auto",
            return_dict=True,
        )
    elif args.half:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            dtype=torch.bfloat16,
            return_dict=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            return_dict=True,
        )

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name)
    return model, tokenizer

model, tokenizer = load_model(args.model)

if not 'paraphrased' in args.dataset:
    dataset = load_dataset('swj0419/WikiMIA', split=args.dataset)
else:
    dataset = load_dataset('zjysteven/WikiMIA_paraphrased_perturbed', split=args.dataset)
data = convert_huggingface_data_to_list_dic(dataset)

perturbed_dataset = load_dataset(
    'zjysteven/WikiMIA_paraphrased_perturbed', 
    split=args.dataset + '_perturbed'
)
perturbed_data = convert_huggingface_data_to_list_dic(perturbed_dataset)
num_neighbors = len(perturbed_data) // len(data)

def inference(text, model):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    ll = -loss.item() # log-likelihood
    return ll

scores = defaultdict(list)
for i, d in enumerate(tqdm(data, total=len(data), desc='Samples')): 
    text = d['input']
    ll = inference(text, model)

    ll_neighbors = []
    for j in range(num_neighbors):
        text = perturbed_data[i * num_neighbors + j]['input']
        ll_neighbors.append(inference(text, model))
    scores['neighbor'].append(ll - np.mean(ll_neighbors))


def get_metrics(scores, labels):
    fpr_list, tpr_list, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr_list, tpr_list)
    fpr95 = fpr_list[np.where(tpr_list >= 0.95)[0][0]]
    tpr05 = tpr_list[np.where(fpr_list <= 0.05)[0][-1]]
    return auroc, fpr95, tpr05

labels = [d['label'] for d in data] # 1: training, 0: non-training
results = defaultdict(list)
for method, scores in scores.items():
    auroc, fpr95, tpr05 = get_metrics(scores, labels)
    
    results['method'].append(method)
    results['auroc'].append(f"{auroc:.1%}")
    results['fpr95'].append(f"{fpr95:.1%}")
    results['tpr05'].append(f"{tpr05:.1%}")

df = pd.DataFrame(results)
print(df)

save_root = f"results/WikiMIA/{args.dataset}"
os.makedirs(save_root, exist_ok=True)

model_id = args.model.split('/')[-1]
csv_path = os.path.join(save_root, f"{model_id}.csv")

if os.path.isfile(csv_path):
    df.to_csv(csv_path, index=False, mode="a", header=False)
else:
    df.to_csv(csv_path, index=False)