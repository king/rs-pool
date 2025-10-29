# Enhancing Graph Classification Robustness with Singular Pooling

This repository is the official implementation of our paper "Enhancing Graph Classification Robustness with Singular Pooling".


## Requirements
Code is written in Python 3.6 and requires:
- PyTorch
- NetworkX
- Torch Geometric

We additionally note that given our adversarial attacks aim, we approach the GNNs through dense adjacency matrices. In case the user want to use sparse-matrix approach, the attacks should be adapted.


## Setup
The code's folder should be divided into the following subfolders:
- datasets: contains the datasets - Note that if throgh Torch Geometric, the datasets shall be downloaded directly.
- data_splits: contains the train/val/test splits.
- src: contains the main scripts and utils scripts to be used
  - attacks: contains the different attacks considered in the paper (Random, PGD, Genetic)
  - models: Implementation of a GCN and GIN that uses the dense matrices, allowing an adaptation with our considered attacks.

## Datasets
As specified in the paper, we use the graph classification datasets from the TUDataset benchmark:
- Bioinformatics (D&D, PROTEINS, ENZYMES)
- Molecules (NCI1, ER_MD)
- Images (MSRC_9)
- Social Graphs (IMDB-B, REDDIT-B)


All these datasets are part of the torch_geometric datasets and can directly be downloaded when running the code. We additionally note that when available, we use the public train/val/test folds. We provide these splits in "data_splits" folder.


## Training and Evaluation

### Hyper-Parameters

To train and evaluate our proposed RS-Pool and to reproduce the results in the paper, the user should specify the following :

- Dataset : The dataset to be used
- hidden_dimension: The hidden dimension used in the model
- Pooling Method: The pooling strategy to be used (including our proposed RS-Pool).
- value_alpha: As specified in details in Section 6.1, we control the value $\tau$ through a scaling parameter $\alpha$ to ensure better robustness (based on Lemma 5.3).


### Train and Attack

To run for instance the PGD attack of RS-Pool with a GCN backbone (related to Table 1 in our paper) and using the generic hyper-parameters:

```bash
python run_pgd.py --name_dataset DD --pooling rs-pool
```

In case the user want to run the PGD attack on other poolings, such as the "Sum" pool, with the generic hyper-parameters:

```bash
python run_pgd.py --name_dataset DD --pooling sum
```

In addition to the GCN, and in order to run the GIN, with the RS-Pool, with the generic hyper-parameters:

In order to run using the GIN backbone:
```bash
python run_pgd_gin.py --name_dataset DD --pooling rs-pool
```

## Citing

Upon using this repository for your work, or finding our proposed analysis useful for your research, please consider citing our paper [this paper](https://arxiv.org/abs/2412.03783):
```
@inproceedings{
ennadir2025enhancing,
title={Enhancing Graph Classification Robustness with Singular Pooling},
author={Sofiane ENNADIR and Oleg Smirnov and Yassine ABBAHADDOU and Lele Cao and Johannes F. Lutzeyer},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=n1XlwRmF7v}
}
```

For any additional questions/suggestions you might have about the code and/or the proposed analysis, please contact: sofiane.ennadir@king.com
