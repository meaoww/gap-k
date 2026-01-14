# Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data

## 🔍 Overview
This repository provides the official implementation of **Gap-K%**, a novel and efficient
reference-free Membership Inference Attack (MIA) for detecting pretraining data in
large language models (LLMs).

For experiments on the Neighbor for the MIMIR benchmark, we use the implementation provided here:  
👉 https://github.com/zjysteven/mimir

## ⚙️ Environment
Our experiments are conducted under the following environment:
- Python 3.10
- PyTorch 2.7.1
- CUDA 12.6

After setting up PyTorch, install the remaining dependencies:
```bash
pip install -r requirements.txt
```
## 🔐 Hugging Face Access
Please log in to Hugging Face before running the scripts:
```bash
huggingface-cli login
```

## 📁 Dataset
- WikiMIA original: [swj0419/WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA). 
- WikiMIA paraphrased: [zjysteven/WikiMIA_paraphrased_perturbed](https://huggingface.co/datasets/zjysteven/WikiMIA_paraphrased_perturbed).
- MIMIR [iamgroot42/mimir](https://huggingface.co/datasets/iamgroot42/mimir)

## 🤖 Models
We conduct WikiMIA experiments on a diverse set of large language models:
- Mamba: [state-spaces/mamba-1.4b-hf](https://huggingface.co/state-spaces/mamba-1.4b-hf)
- Pythia: [EleutherAI/pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b), [EleutherAI/pythia-12b](https://huggingface.co/EleutherAI/pythia-12b)
- LLaMA: [huggyllama/llama-13b](https://huggingface.co/huggyllama/llama-13b), [huggyllama/llama-65b](https://huggingface.co/huggyllama/llama-65b)

**Note:** LLaMA-65B is evaluated using INT8 inference.

For MIMIR experiments, we use Pythia [160M](https://huggingface.co/EleutherAI/pythia-160m), [1.4B](https://huggingface.co/EleutherAI/pythia-1.4b), [2.8B](https://huggingface.co/EleutherAI/pythia-2.8b), [6.9B](https://huggingface.co/EleutherAI/pythia-6.9b), and [12B](https://huggingface.co/EleutherAI/pythia-12b).

## 🚀 Running
We provide shell scripts for running all experiments:
- `wikimia.sh`  
  Evaluates Loss, Zlib, Min-K%, Min-K%++, Gap-K% on WikiMIA.

- `wikimia_neighbor.sh`  
  Evaluates Neighbor on WikiMIA.

- `mimir.sh`  
  Evaluates Loss, Zlib, Min-K%, Min-K%++, Gap-K% on MIMIR.

Results are saved to:
```text
results/
```
## 🙏Acknowledgement
This implementation is based on the official codebase of [Min-K%++](https://github.com/zjysteven/mink-plus-plus).