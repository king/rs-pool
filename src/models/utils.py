"""
    Utils training file and attack file
"""

import numpy as np
import scipy.sparse as sp
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_tensor_adj(adj, device=torch.device("cpu")):
    """
    Function to normalize the input adjacency matrix
    """
    n = adj.shape[0]
    A = adj + torch.eye(n).to(device)
    D = torch.sum(A, 0)
    D_hat = torch.diag(((D) ** (-0.5)))
    adj_normalized = torch.mm(torch.mm(D_hat, A), D_hat)

    return adj_normalized


def train_function(model, optimizer, data_loader, fold, device,
                    num_epochs=100, batch_size=8, verbose=True, normalize=True):
    """
        Main training function.
            - Takes the number fold
            - Extract the data
            - For each batch, transform the adjacency matrices as block matrix
            - Train the model accordingly

    """
    Adj_train, X_train, y_train, Adj_val, \
     X_val, y_val, Adj_test, X_test, y_test = data_loader.get_fold_data(fold)

    N_train = len(X_train)
    best_val = 0
    for epoch in range(num_epochs):
        model.train()

        train_loss = 0
        correct = 0
        count = 0
        for i in range(0, N_train, batch_size):
            adj_batch = list()
            idx_batch = list()
            y_batch = list()
            features_batch = list()

            for j in range(i, min(N_train, i+batch_size)):

                n = Adj_train[j].shape[0]
                if normalize and model.model_type == "GCN":
                    adj_norm = normalize_tensor_adj(Adj_train[j])
                elif normalize and model.model_type == "Jaccard":
                    clean_adj = dropedge_jaccard_torch(Adj_train[j], X_train[j])
                    adj_norm = normalize_tensor_adj(clean_adj)
                else:
                    adj_norm = Adj_train[j]

                adj_batch.append(adj_norm)

                idx_batch.extend([j-i]*n)
                y_batch.append(y_train[j])
                features_batch.append(X_train[j])

            adj_batch = sp.block_diag(adj_batch)
            adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device).to_dense()

            features_batch = torch.cat(features_batch)
            features_batch = torch.FloatTensor(features_batch).to(device)

            idx_batch = torch.LongTensor(idx_batch).to(device)
            y_batch = torch.LongTensor(y_batch).to(device)

            optimizer.zero_grad()

            output = model(features_batch, adj_batch, idx_batch)

            loss = F.nll_loss(output, y_batch)

            train_loss += loss.item() * output.size(0)
            count += output.size(0)
            preds = output.max(1)[1].type_as(y_batch)
            correct += torch.sum(preds.eq(y_batch).double())
            loss.backward()
            optimizer.step()

        if model.model_type == "Jaccard" and epoch % 30 == 0:
            acc_val = test_function(model, data_loader, \
                                Adj_val, X_val, y_val, device, \
                        batch_size=batch_size, verbose=False, normalize=normalize)
        else:
            acc_val = test_function(model, data_loader, \
                                Adj_val, X_val, y_val, device, \
                        batch_size=batch_size, verbose=False, normalize=normalize)

        # Patience implementation
        if acc_val >= best_val:
            best_val = acc_val
            best_model = copy.deepcopy(model)

        if epoch % 30 == 0 and verbose == True:
            print('Epoch: {:04d}'.format(epoch+1),
                  'loss_train: {:.4f}'.format(train_loss / count),
                  'acc_train: {:.4f}'.format(correct / count),
                  'acc_val: {:.4f}'.format(acc_val))

    return best_model


def test_function(model_local, data_loader, Adj_local, X_local, \
        y_local, device, num_epochs = 100, batch_size=8, verbose=True, normalize=True):
    """
        Main test function.
            - Takes the number fold
            - Extract the data
            - For each batch, like the train function, it transform the
              adjacency matrices as block matrix
    """
    N_test = len(X_local)
    model_local.eval()
    test_loss = 0
    correct = 0
    count = 0
    predicted_list = []
    for i in range(0, N_test, batch_size):
        adj_batch = list()
        idx_batch = list()
        y_batch = list()
        features_batch = list()

        for j in range(i, min(N_test, i+batch_size)):
            n = Adj_local[j].shape[0]
            if normalize and model_local.model_type == "GCN":
                adj_norm = normalize_tensor_adj(Adj_local[j])
            elif normalize and model_local.model_type == "Jaccard":
                clean_adj = dropedge_jaccard_torch(Adj_local[j], X_local[j])
                adj_norm = normalize_tensor_adj(clean_adj)
            else:
                adj_norm = Adj_local[j]

            adj_batch.append(adj_norm)

            idx_batch.extend([j-i]*n)
            y_batch.append(y_local[j])
            features_batch.append(X_local[j])

        adj_batch = sp.block_diag(adj_batch)
        adj_batch = sparse_mx_to_torch_sparse_tensor(adj_batch).to(device).to_dense()

        features_batch = torch.cat(features_batch)
        features_batch = torch.FloatTensor(features_batch).to(device)

        idx_batch = torch.LongTensor(idx_batch).to(device)
        y_batch = torch.LongTensor(y_batch).to(device)

        output = model_local(features_batch, adj_batch, idx_batch)

        count += output.size(0)
        preds = output.max(1)[1].type_as(y_batch)
        predicted_list.extend(preds)
        correct += torch.sum(preds.eq(y_batch).double())

    acc = (correct/N_test)
    if verbose == True:
        print('Accuracy on the test set: {:.4f}'.format(acc))
    return acc


def classification_loss(num_classes):
    """Returns a loss function for classification tasks."""
    if num_classes == 2:
        def loss_fn(x, y, **kwargs):
            return nn.functional.binary_cross_entropy_with_logits(x.squeeze(), y.float(), **kwargs)
    else:
        loss_fn = nn.functional.cross_entropy
    return loss_fn

def number_of_correct_predictions(predictions, labels, num_classes):
    """Sum of predictions with agree with labels. Predictions is given in logits."""
    if num_classes == 2:
        return ((predictions.squeeze() > 0).float() == labels).sum()
    else:
        return (predictions.argmax(axis=1) == labels).sum()
