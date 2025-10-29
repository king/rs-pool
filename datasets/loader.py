import os
import torch
import json
import numpy as np
import networkx as nx
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.convert import to_networkx

list_binary = ["IMDB-BINARY", "IMDB-MULTI", "COLLAB", "REDDIT-BINARY", "REDDIT-MULTI-5K"]

class data_loader:
    def __init__(self, name_dataset):
        # Load TUDataset
        dataset = TUDataset(root='./datasets/'+name_dataset, name=name_dataset)
        self.num_examples = len(dataset)
        self.Adj_list = []
        self.y_list = []
        self.X_list = []

        for i in range(self.num_examples):
            G = to_networkx(dataset[i])
            G = nx.to_undirected(G)
            adj = nx.adjacency_matrix(G)
            adj = torch.FloatTensor(adj.toarray())
            self.Adj_list.append(adj)

            if np.isin(name_dataset, list_binary):
                self.X_list.append(torch.FloatTensor(adj.sum(axis=1)).unsqueeze(1))
                self.input_dim = 1
            else:
                self.X_list.append(torch.FloatTensor(dataset[i].x.float()))
                self.input_dim = self.X_list[0].shape[1]

            self.y_list.append(dataset[i].y)

        self.num_classes = len(np.unique(self.y_list))

        # Define split file path
        split_file = './data_splits/' + name_dataset + '_splits.json'

        # Check if the splits file exists; if not, generate and save 10 random folds.
        if not os.path.exists(split_file):
            splits = []
            for _ in range(10):
                # Get a random permutation of indices
                idx = np.random.permutation(self.num_examples)
                n_test = int(np.floor(0.2 * self.num_examples))
                n_val = int(np.floor(0.1 * self.num_examples))

                test_indices = idx[:n_test].tolist()
                val_indices = idx[n_test:n_test+n_val].tolist()
                train_indices = idx[n_test+n_val:].tolist()

                fold_split = {
                    'test': test_indices,
                    'model_selection': [{
                        'train': train_indices,
                        'validation': val_indices
                    }]
                }
                splits.append(fold_split)

            # Ensure the directory exists and save the splits
            os.makedirs(os.path.dirname(split_file), exist_ok=True)
            with open(split_file, 'w') as f:
                json.dump(splits, f)
            self.split = splits
        else:
            # Load the train/validation/test splits from file
            with open(split_file, 'r') as f:
                self.split = json.load(f)

    def get_fold_data(self, int_fold):
        fold = self.split[int_fold]
        test_indices = fold['test']
        train_indices = fold['model_selection'][0]['train']
        val_indices = fold['model_selection'][0]['validation']

        Adj_test = [self.Adj_list[x] for x in test_indices]
        X_test = [self.X_list[x] for x in test_indices]
        y_test = [self.y_list[x] for x in test_indices]

        Adj_train = [self.Adj_list[x] for x in train_indices]
        X_train = [self.X_list[x] for x in train_indices]
        y_train = [self.y_list[x] for x in train_indices]

        Adj_val = [self.Adj_list[x] for x in val_indices]
        X_val = [self.X_list[x] for x in val_indices]
        y_val = [self.y_list[x] for x in val_indices]

        return Adj_train, X_train, y_train, Adj_val, X_val, y_val, Adj_test, X_test, y_test
