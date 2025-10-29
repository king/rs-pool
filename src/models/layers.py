# Copyright (C) king.com Ltd 2025
# License: Apache 2.0
# Inspired and Adapted and inspired from the original GCN and GIN papers.

"""
Main implementation of the different layers that are used in the code base
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class convClass(nn.Module):
    def __init__(self, input_dim , output_dim, activation):
        super(convClass, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.Tensor(self.output_dim, self.input_dim))
        self.activation = activation
        self.reset_parameters()

    def forward(self, x, adj):
        x = F.linear(x, self.weight)
        return self.activation(torch.mm(adj, x))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

class GINConv(nn.Module):
    def __init__(self, input_dim, output_dim, activation, eps=0.0, train_eps=False):
        """
        GIN convolution layer.

        Args:
            input_dim (int): Dimension of the input node features.
            output_dim (int): Dimension of the output node features.
            activation (callable): Activation function (e.g. nn.ReLU()).
            eps (float): Initial value for epsilon.
            train_eps (bool): If True, epsilon becomes a learnable parameter.
        """
        super(GINConv, self).__init__()
        # Two-layer MLP as described in the GIN paper.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.activation = activation
        if train_eps:
            self.eps = Parameter(torch.Tensor([eps]))
        else:
            self.eps = eps
        self.train_eps = train_eps

    def forward(self, x, adj):
        """
        Forward propagation using dense adjacency.

        Args:
            x (torch.Tensor): Node features, shape [N, input_dim].
            adj (torch.Tensor): Dense adjacency matrix (without self-loops), shape [N, N].

        Returns:
            torch.Tensor: Updated node features, shape [N, output_dim].
        """
        # If adj is expected to have no self-loops, add an identity matrix.
        I = torch.eye(x.size(0), device=x.device)
        # Aggregate neighbor features.
        neighbor_sum = torch.mm(adj, x)
        # Compute (1 + eps)*x + neighbor aggregate.
        out = (1 + self.eps) * x + neighbor_sum
        # Pass through the MLP and activation.
        out = self.mlp(out)
        return self.activation(out)

if __name__ == "__main__":
    pass
