# Gap-K%: Measuring Top-1 Prediction Gap for Detecting Pretraining Data

## Overview

![teaser figure](images/teaser_w_results.png)

We propose a new Membership Inference Attack method named **Gap-K** for detecting pre-training data of LLMs, which achieves SOTA results among reference-free methods. This repo contains the implementation of our method (along with all the baselines) on the [WikiMIA benchmark](https://huggingface.co/datasets/swj0419/WikiMIA). For experiments on the [MIMIR benchmark](https://github.com/iamgroot42/mimir) Neighbor, we use [here](https://github.com/zjysteven/mimir).


## Setup
### Environment
First install torch according to your environment. Then simply install dependencies by `pip install -r requirements.txt`.

Our code is tested with Python 3.10, PyTorch 2.7.1, Cuda 12.6.

### Data
All data splits are hosted on huggingface and will be automatically loaded when running scripts.
- The original WikiMIA is from [🤗swj0419/WikiMIA](https://huggingface.co/datasets/swj0419/WikiMIA). 
- The WikiMIA paraphrased is from [🤗zjysteven/WikiMIA_paraphrased_perturbed](https://huggingface.co/datasets/zjysteven/WikiMIA_paraphrased_perturbed).

## Running
There are four scripts, each of which is self-contained to best facilitate quick reproduction and extension. The meaning of the arguments of each script should be clear from their naming.

- `wikimia.py` will run the Loss, Zlib, Min-K%, Min-K%++, and Gap-K% attack on the WikiMIA dataset (either the original or the paraphrased version) with the specified model.
- `wikimia_neighbor.py` will run the Neighbor attack on the WikiMIA dataset (either the original or the paraphrased version) with the specified model.
- `mimir.py` focus on the MIMIR dataset with the specified model. For this setting only the Loss, Zlib, Min-K%, Min-K%++, and Gap-K% are applicable.

The outputs of these scripts will be a csv file consisting of method results (AUROC and TPR@FPR=5%) stored in the `results` directory, with the filepath indicating the dataset and model. Sample results by running the four scripts are provided in the `results` directory.

## HF paths of evaluated model in the paper
- Mamba: [state-spaces/mamba-1.4b-hf](https://huggingface.co/state-spaces/mamba-1.4b-hf)
- Pythia: [EleutherAI/pythia-6.9b](https://huggingface.co/EleutherAI/pythia-6.9b), [EleutherAI/pythia-12b](https://huggingface.co/EleutherAI/pythia-12b)
- LLaMA: [huggyllama/llama-13b](https://huggingface.co/huggyllama/llama-13b), [huggyllama/llama-65b](https://huggingface.co/huggyllama/llama-65b)


## Acknowledgement
This codebase is adapted from the [official repo](https://github.com/zjysteven/mink-plus-plus) of Min-K%++.