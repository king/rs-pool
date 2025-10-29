import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from ..models.utils import normalize_tensor_adj

class pgd_attack:
    """
    Implementation of the Topological attack using the PGD.
    ---
    parameters:
        adj : torch.Tensor (adjacency matrix).
        x : graph node features.
        label : previous classification or graph's label
        classifier: victim classifier to be attacked

    """
    def __init__(self, adj, x, label, classifier, device):
        self.adj = adj.to(device)
        self.nnodes = adj.shape[0]
        self.x = x.to(device)
        self.label = label.to(device)
        self.surrogate = classifier

        self.adj_changes = Parameter(torch.FloatTensor(int(self.nnodes*(self.nnodes-1)/2))).to(device)
        self.adj_changes.data.fill_(0)

        self.device = device
        self.loss_type = "CE"
        self.complementary = None

    def attack(self, epochs, n_perturbations, max_time=30, verbose=True):
        """
        Run the attack for a maximum number of epochs or until max_time is exceeded.
        """
        victim_model = self.surrogate
        ori_adj = copy.deepcopy(self.adj)
        label = self.label
        ori_features = copy.deepcopy(self.x)

        victim_model.eval()
        start_time = time.time()

        for t in range(epochs):
            if time.time() - start_time > max_time:
                break

            modified_adj = self.get_modified_adj(ori_adj)

            # If it's a GCN, we should normalize
            if victim_model.model_type == "GCN":
                adj_norm = normalize_tensor_adj(modified_adj, self.device)
            else:
                adj_norm = modified_adj

            output = victim_model.predict(adj_norm, ori_features)
            loss = self._loss(output, label)
            adj_grad = torch.autograd.grad(loss, self.adj_changes)[0]

            # Update adj_changes based on the loss type
            if self.loss_type == 'CE':
                lr = 200 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)
            elif self.loss_type == 'CW':
                lr = 0.1 / np.sqrt(t + 1)
                self.adj_changes.data.add_(lr * adj_grad)

            self.projection(n_perturbations)

        # After the iterative attack, use random sampling as a final step.
        self.random_sample(ori_adj, ori_features, label, n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        self.check_adj_tensor(self.modified_adj)

    def random_sample(self, ori_adj, ori_features, labels, n_perturbations):
        K = 20
        best_loss = -1000
        victim_model = self.surrogate
        victim_model.eval()
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)
                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled, dtype=self.adj_changes.data.dtype, device=self.device))
                modified_adj = self.get_modified_adj(ori_adj)

                # If it's a GCN, we should normalize
                if victim_model.model_type == "GCN":
                    adj_norm = normalize_tensor_adj(modified_adj, self.device)
                else:
                    adj_norm = modified_adj


                output = victim_model.predict(adj_norm, ori_features)
                loss = self._loss(output, labels)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s, dtype=self.adj_changes.data.dtype, device=self.device))

    def get_modified_adj(self, ori_adj):
        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj

        m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m[tril_indices[0], tril_indices[1]] = self.adj_changes
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj
        return modified_adj

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        elif self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000 * onehot).argmax(1)
            margin = output[torch.arange(len(output)), labels] - \
                     output[torch.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
        return loss

    def projection(self, n_perturbations):
        # If the total perturbations exceed the allowed number, use bisection to project.
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=1e-5, max_time_bisection=100)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def bisection(self, a, b, n_perturbations, epsilon, max_time_bisection=100):
        """
        Find the threshold via bisection such that after
        projection the number of changes is n_perturbations.
        """
        def func(x):
            return torch.clamp(self.adj_changes - x, 0, 1).sum() - n_perturbations

        miu = a
        start_time = time.time()
        while (b - a) >= epsilon:
            if time.time() - start_time > max_time_bisection:
                break
            miu = (a + b) / 2
            if func(miu) == 0.0:
                break
            if func(miu) * func(a) < 0:
                b = miu
            else:
                a = miu
        return miu

    def check_adj_tensor(self, adj):
        """Check if the modified adjacency is symmetric, unweighted, with zero diagonal."""
        assert torch.abs(adj - adj.t()).sum() == 0, "Input graph is not symmetric"
        assert adj.min() == 0, "Min value should be 0!"
        diag = adj.diag()
        assert diag.max() == 0, "Diagonal should be 0!"
        assert diag.min() == 0, "Diagonal should be 0!"
