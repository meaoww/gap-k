import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import zlib
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset


def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='EleutherAI/pythia-12b')
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
parser.add_argument("--half", action="store_true")
parser.add_argument("--int8", action="store_true")
parser.add_argument("--window_size", type=int, default=3)
args = parser.parse_args()

RATIO = 0.2

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

def window_lowmean(x: np.ndarray, window: int, ratio: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0

    if x.size < window:
        smoothed = x
    else:
        kernel = np.ones(window, dtype=np.float64) / window
        smoothed = np.convolve(x, kernel, mode="valid")

    k = max(1, int(len(smoothed) * ratio))
    return float(np.mean(np.sort(smoothed)[:k]))

model, tokenizer = load_model(args.model)

if "paraphrased" not in args.dataset:
    dataset = load_dataset("swj0419/WikiMIA", split=args.dataset)
else:
    dataset = load_dataset("zjysteven/WikiMIA_paraphrased_perturbed", split=args.dataset)

data = convert_huggingface_data_to_list_dic(dataset)
scores = defaultdict(list)

for d in tqdm(data, total=len(data), desc="Samples"):
    text = d["input"]
    input_ids_full = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids_full, labels=input_ids_full)

    loss, logits = outputs[:2]
    ll = -loss.item()

    scores["loss"].append(ll)
    scores["zlib"].append(ll / len(zlib.compress(bytes(text, "utf-8"))))

    logits_tv = logits[0, :-1]
    input_ids_shifted = input_ids_full[0, 1:].unsqueeze(-1)

    logits_tv = logits_tv.to(torch.float32)   ## ADD
    probs = F.softmax(logits_tv, dim=-1)
    log_probs = F.log_softmax(logits_tv, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids_shifted).squeeze(-1)

    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    sigma = torch.clamp(sigma, min=1e-8)

    tlogp_np = token_log_probs.detach().cpu().numpy().astype(np.float64)
    k_len = max(1, int(len(tlogp_np) * RATIO))
    scores[f"mink_{RATIO}"].append(float(np.mean(np.sort(tlogp_np)[:k_len])))

    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    mink_plus_np = mink_plus.detach().cpu().numpy().astype(np.float64)
    k_len2 = max(1, int(len(mink_plus_np) * RATIO))
    scores[f"minkpp_{RATIO}"].append(float(np.mean(np.sort(mink_plus_np)[:k_len2])))

    logp_true = token_log_probs
    top1_id = torch.argmax(logits_tv, dim=-1)
    top1_logp = log_probs.gather(dim=-1, index=top1_id.unsqueeze(-1)).squeeze(-1)
    gap = logp_true - top1_logp
    gap_norm = (gap / sigma.sqrt()).detach().cpu().numpy().astype(np.float64)

    scores[f"gapk_{RATIO}"].append(window_lowmean(gap_norm, window=args.window_size, ratio=RATIO))


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

model_id = args.model.split("/")[-1]
csv_path = os.path.join(save_root, f"{model_id}.csv")

if os.path.isfile(csv_path):
    df.to_csv(csv_path, index=False, mode="a", header=False)
else:
    df.to_csv(csv_path, index=False)