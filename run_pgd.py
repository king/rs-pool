# Copyright (C) king.com Ltd 2025
# License: Apache 2.0
"""
Main script to run the PGD Attack on the GCN
"""

import argparse
import copy
import numpy as np
from tqdm import tqdm
import os
import random
import pandas as pd

import torch
import torch.nn.functional as F

from src.models.gcn import GCN
from src.models.utils import train_function, test_function
from src.models.utils import classification_loss, normalize_tensor_adj

from datasets.loader import data_loader
from src.attacks.topological_attacks import pgd_attack
from src.utils import set_seed

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_dataset', type=str, default='DD',
                                                            help='Data set')
    parser.add_argument('--model', type=str, default='GCN', help='Model type')
    parser.add_argument('--hidden_dim', type=int, default=32,
                                                            help='Hidden dim')
    parser.add_argument('--pooling', type=str, default='rs-pool',
                                                            help='Pooling Type')
    parser.add_argument('--value_alpha', type=float, default=20,
                                        help='Temperature Value of the RS-Pool')
    args = parser.parse_args()

    # Set to seed
    set_seed(42)

    # Load dataset
    data = data_loader(args.name_dataset)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Parameters
    batch_size = 32
    training_epochs = 101
    lr = 1e-03
    attack_epochs = 100
    budget = 0.3
    fold = 1

    # Let's init the GCN
    model = GCN(data.input_dim, args.hidden_dim,
                data.num_classes, device, pooling=args.pooling,
                value_alpha=args.value_alpha).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = F.nll_loss

    # Train model on the current fold
    model = train_function(model, optimizer, data,
                           fold, device, num_epochs=training_epochs,
                           batch_size=batch_size)

    # Get the test data
    _, _, _, _, _, _, Adj_test, X_test, y_test = data.get_fold_data(fold)

    # Evaluate the model on the clean test set
    test_acc = test_function(model, data, Adj_test,
                             X_test, y_test, device,
                             batch_size=batch_size, verbose=True)
    clean_acc = test_acc.item() if hasattr(test_acc, "item") else test_acc
    print(f"Clean Accuracy: {clean_acc}")

    success = 0
    correct = 0
    for i in tqdm(range(len(Adj_test)), desc=f"Attack budget {budget}"):
        ori_adj, x, y = Adj_test[i], X_test[i], y_test[i]
        n_perturbations = int(ori_adj.sum() // 2 * budget) + 1

        ori_adj = ori_adj.to(device)
        x = x.to(device)
        y = y.to(device)

        # Normalize the original adjacency matrix and predict the class
        adj_norm = normalize_tensor_adj(copy.deepcopy(ori_adj), device=device)
        pred = model.predict(adj_norm, x)

        # Only attack if the original prediction is correct
        if pred.detach().max(1)[1] == y:
            correct += 1
            attacker = pgd_attack(ori_adj, x, y, model, device)
            attacker.attack(attack_epochs, n_perturbations)
            adj_attacked = normalize_tensor_adj(attacker.modified_adj,
                                                                device=device)
            pred_attacked = model.predict(adj_attacked, x)
            # Check if it was mis-classified
            if pred_attacked.detach().max(1)[1] != y:
                success += 1

    # Compute the final metrics of the experiment
    attack_success_rate = success / len(Adj_test)
    acc_rate = correct / len(Adj_test)
    attacked_acc = 1 - ((1 - acc_rate) + attack_success_rate)

    # Print the acc_rate along with the other metrics
    print(f"Budget: {budget} - Success Rate: {attack_success_rate}, \
                    Accuracy Rate: {acc_rate}, Attacked Acc: {attacked_acc}")
