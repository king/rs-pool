# Copyright (C) king.com Ltd 2025
# License: Apache 2.0
# Adapted and inspired from the original GCN paper (Semi-Supervised
# Classification with Graph

"""
Main implementation of the GCN model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .layers import convClass

class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) with various pooling strategies.
    As implented by original work: "Semi-Supervised Classification with Graph
    Convolutional Networks".
    ---
    Args:
        input_dim (int): Dimension of input node features.
        hidden_dim (int): Hidden layer dimension.
        output_dim (int): Number of output classes.
        device (torch.device): Device on which to run the model.
        dropout (float): Dropout probability (if used later).
        pooling (str): Pooling strategy ('sum', 'avg', 'max', 'rs-pool',
                                        'attn_avg', 'sag', 'topk', 'asa',
                                        'pan', 'sort').
        eps (float): Initial epsilon for the GIN convolution layers.
        train_eps (bool): If True, epsilon is learned during training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: torch.device,
        dropout: float = 0.5,
        pooling: str = "rs-pool",
        value_alpha: float = 1.0,
        num_iterations: int = 2,
        k_ratio: float = 0.5):
        super().__init__()
        self.device = device
        self.activation = nn.ReLU()

        # Convolutional backbone
        self.conv1 = convClass(input_dim, hidden_dim, activation=self.activation)
        self.conv2 = convClass(hidden_dim, hidden_dim, activation=self.activation)
        self.dropout_layer = nn.Dropout(p=dropout)

        # Pooling config
        self.pooling = pooling.lower()
        self.value_alpha = value_alpha
        self.num_iterations = num_iterations # Num of iterative power iterations
        self.k_ratio = min(max(k_ratio, 0.0), 1.0)
        print("Using pooling method:", self.pooling)

        # Parameterised pooling setups
        if self.pooling == "attn_avg":
            self.att_weight = Parameter(torch.Tensor(hidden_dim))
            nn.init.xavier_uniform_(self.att_weight.unsqueeze(0))

        elif self.pooling == "sag":
            self.sag_query = Parameter(torch.Tensor(hidden_dim))
            nn.init.xavier_uniform_(self.sag_query.unsqueeze(0))

        elif self.pooling == "topk":
            self.topk_query = Parameter(torch.Tensor(hidden_dim))
            nn.init.xavier_uniform_(self.topk_query.unsqueeze(0))

        elif self.pooling == "asa":
            self.asa_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        elif self.pooling == "sort":
            self.sort_query = Parameter(torch.Tensor(hidden_dim))
            nn.init.xavier_uniform_(self.sort_query.unsqueeze(0))
            self.sort_scale = Parameter(torch.ones(1))


        # ReadOut function
        self.lin = nn.Linear(hidden_dim, output_dim)
        self.model_type = "GCN"

    # ---------------------------------------------------------------------
    def _topk(self, x_graph: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the TopK Pooling
            --> Select top‑k nodesthen sum their features.
        """
        num_nodes = x_graph.size(0)
        k = max(1, int(self.k_ratio * num_nodes))
        _, topk_idx = torch.topk(scores, k)
        return torch.sum(x_graph[topk_idx], dim=0)

    # ---------------------------------------------------------------------
    def _svd_pool(self, x_graph: torch.Tensor) -> torch.Tensor:
        """
        Implementation of our proposed RS-Pool, which consists of approximating
        the dominant singular vector via power iterations then rescale.
        """
        vec = torch.sum(x_graph, dim=0)
        vec = vec / (torch.norm(vec) + 1e-8)
        for _ in range(self.num_iterations):
            vec = torch.matmul(x_graph.t(), torch.matmul(x_graph, vec))
            vec = vec / (torch.norm(vec) + 1e-8)
        singular_value = torch.norm(torch.matmul(x_graph, vec))
        return vec * (singular_value / self.value_alpha)

    # ---------------------------------------------------------------------
    def forward(self, x_in: torch.Tensor, adj: torch.Tensor, idx: torch.Tensor):
        """
        Forward pass for batched graphs using dense adjacency.
        Note that the adjacency are supposed to be normalized.

        Args:
            x_in (torch.Tensor): Node features for all graphs.
            adj (torch.Tensor): Normalized Dense adjacency matrix for all graphs.
            idx (torch.Tensor): Graph index for each node.

        Returns:
            torch.Tensor: Log-softmax outputs for each graph.
        """
        x = self.conv1(x_in, adj)
        x = self.conv2(x, adj)
        x = self.dropout_layer(x)

        batch_size = int(torch.max(idx).item()) + 1
        pooled_out = []

        for graph_id in range(batch_size):
            node_indices = (idx == graph_id).nonzero(as_tuple=False).view(-1)
            x_graph = x[node_indices]
            adj_sub = adj[node_indices][:, node_indices]

            if self.pooling == "sum":
                pooled_vec = torch.sum(x_graph, dim=0)

            elif self.pooling == "avg":
                pooled_vec = torch.mean(x_graph, dim=0)

            elif self.pooling == "max":
                pooled_vec, _ = torch.max(x_graph, dim=0)

            elif self.pooling == "rs-pool":
                pooled_vec = self._svd_pool(x_graph)

            elif self.pooling == "attn_avg":
                scores = torch.matmul(x_graph, self.att_weight)
                attn = F.softmax(scores, dim=0)
                pooled_vec = torch.sum(x_graph * attn.unsqueeze(1), dim=0)

            elif self.pooling == "sag":
                scores = torch.tanh(torch.matmul(x_graph, self.sag_query))
                attn = F.softmax(scores, dim=0)
                pooled_vec = torch.sum(x_graph * attn.unsqueeze(1), dim=0)

            elif self.pooling == "topk":
                scores = torch.matmul(x_graph, self.topk_query)
                pooled_vec = self._topk(x_graph, scores)

            elif self.pooling == "asa":
                scores = self.asa_mlp(x_graph).squeeze(1)
                pooled_vec = self._topk(x_graph, scores)

            elif self.pooling == "pan":
                centrality = torch.sum(adj_sub, dim=1)
                pooled_vec = self._topk(x_graph, centrality)

            elif self.pooling == "sort":
                scores = torch.matmul(x_graph, self.sort_query) * self.sort_scale
                pooled_vec = self._topk(x_graph, scores)  # same k_ratio rule

            else:
                raise ValueError(f"Unsupported pooling type: {self.pooling}")

            pooled_out.append(pooled_vec)

        pooled_out = torch.stack(pooled_out, dim=0)
        out = self.lin(pooled_out)
        return F.log_softmax(out, dim=1)

    # ---------------------------------------------------------------------
    def predict(self, adj: torch.Tensor, x_in: torch.Tensor):
        """
        Inference for a single graph.
        This is needed for the attack

        Args:
            adj (torch.Tensor): Dense adjacency matrix.
            x_in (torch.Tensor): Node features.
        """
        x = self.conv1(x_in, adj)
        x = self.conv2(x, adj)
        x = self.dropout_layer(x)

        if self.pooling == "sum":
            pooled_vec = torch.sum(x, dim=0)
        elif self.pooling == "avg":
            pooled_vec = torch.mean(x, dim=0)
        elif self.pooling == "max":
            pooled_vec, _ = torch.max(x, dim=0)
        elif self.pooling == "rs-pool":
            pooled_vec = self._svd_pool(x)
        elif self.pooling == "attn_avg":
            scores = torch.matmul(x, self.att_weight)
            attn = F.softmax(scores, dim=0)
            pooled_vec = torch.sum(x * attn.unsqueeze(1), dim=0)
        elif self.pooling == "sag":
            scores = torch.tanh(torch.matmul(x, self.sag_query))
            attn = F.softmax(scores, dim=0)
            pooled_vec = torch.sum(x * attn.unsqueeze(1), dim=0)
        elif self.pooling == "topk":
            scores = torch.matmul(x, self.topk_query)
            pooled_vec = self._topk(x, scores)
        elif self.pooling == "asa":
            scores = self.asa_mlp(x).squeeze(1)
            pooled_vec = self._topk(x, scores)
        elif self.pooling == "pan":
            centrality = torch.sum(adj, dim=1)
            pooled_vec = self._topk(x, centrality)
        elif self.pooling == "sort":
            scores = torch.matmul(x, self.sort_query) * self.sort_scale
            pooled_vec = self._topk(x, scores)
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        pooled_vec = pooled_vec.unsqueeze(0)
        out = self.lin(pooled_vec)
        return F.log_softmax(out, dim=1)
