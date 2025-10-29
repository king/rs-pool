import copy
import numpy as np
import torch
from ..models.utils import normalize_tensor_adj
import time
def random_sample_flip(adj, budget, add_edge_only=False, remove_edge_only=False, preserve_disconnected_components=False):
    """
    Randomly sample candidate edge flips.

    Args:
        adj: torch.Tensor (adjacency matrix).
        budget: number of edge flips to propose.
        add_edge_only: if True, only propose to add an edge (i.e. where entry == 0).
        remove_edge_only: if True, only propose to remove an edge (i.e. where entry == 1).
        preserve_disconnected_components: if True, avoid flips that disconnect the graph (not implemented here).

    Returns:
        List of tuples (i, j) with i < j.
    """
    n = adj.shape[0]
    edges = []
    for _ in range(budget):
        while True:
            i = np.random.randint(0, n)
            j = np.random.randint(0, n)
            if i == j:
                continue
            if i > j:
                i, j = j, i  # ensure i < j for uniqueness
            current_val = adj[i, j].item()
            if add_edge_only and current_val == 1:
                continue
            if remove_edge_only and current_val == 0:
                continue
            edges.append((i, j))
            break
    return edges

def apply_edge_flips(adj, edges, mode='flip'):
    """
    Apply the candidate edge flips to a copy of the adjacency matrix.

    Args:
        adj: torch.Tensor (original adjacency).
        edges: list of tuples (i, j) to flip.
        mode: 'flip' (flip existing: if edge exists remove it, else add it),
              'add' (only add edges), or 'remove' (only remove edges).

    Returns:
        new_adj: perturbed adjacency matrix (torch.Tensor).
    """
    new_adj = adj.clone()
    for (i, j) in edges:
        if mode == 'flip':
            new_adj[i, j] = 1 - new_adj[i, j]
            new_adj[j, i] = 1 - new_adj[j, i]
        elif mode == 'add':
            new_adj[i, j] = 1
            new_adj[j, i] = 1
        elif mode == 'remove':
            new_adj[i, j] = 0
            new_adj[j, i] = 0
    return new_adj

class RandomFlip:
    def __init__(self, classifier: torch.nn.Module, device, mode: str = 'flip',
                 preserve_disconnected_components=False, **kwargs):
        """
        Baseline random attack that randomly flips edges for an untargeted attack.

        Args:
            classifier: Model with a predict(adj, x) method.
            device: Torch device (e.g., 'cuda' or 'cpu') for running the model.
            mode: 'flip', 'add', 'remove', or 'rewire' (rewire is treated as flip).
            preserve_disconnected_components: if True, avoid perturbations that disconnect the graph.
        """
        self.classifier = classifier
        self.device = device
        assert mode in ['flip', 'add', 'remove', 'rewire'], f"Mode {mode} not recognized!"
        self.mode = mode
        self.preserve_disconnected_components = preserve_disconnected_components


    def attack(self, adj: torch.Tensor, x: torch.Tensor, label: torch.Tensor, budget: int, max_queries: int):
        """
        Perform the random edge-flip attack for an untargeted setting.

        Args:
            adj: Original adjacency matrix (torch.Tensor).
            x: Node feature matrix (torch.Tensor).
            label: Graph label (torch.Tensor); assumed to be a single label.
            budget: Number of edge flips to perform per candidate.
            max_queries: Maximum number of perturbation attempts.

        Returns:
            adv_adj: The adversarial (perturbed) adjacency matrix (torch.Tensor) if the attack succeeded
                     (i.e. if the classifier's prediction changes from the true label), otherwise None.
        """
        start_time = time.time()
        for query in range(max_queries):
            # Sample candidate edges.
            if self.mode == 'rewire':
                edges = random_sample_flip(adj, budget, preserve_disconnected_components=self.preserve_disconnected_components)
            else:
                add_edge_only = (self.mode == 'add')
                remove_edge_only = (self.mode == 'remove')
                edges = random_sample_flip(adj, budget, add_edge_only=add_edge_only,
                                           remove_edge_only=remove_edge_only,
                                           preserve_disconnected_components=self.preserve_disconnected_components)
            # Apply the sampled edge flips.
            perturbed_adj = apply_edge_flips(adj, edges, mode=self.mode)

            # Normalize the perturbed adjacency matrix for the GCN.
            if self.classifier.model_type == "GCN":
                perturbed_adj_norm = normalize_tensor_adj(perturbed_adj, device=self.device).to(self.device)
            else:
                perturbed_adj_norm = perturbed_adj.to(self.device)

            with torch.no_grad():
                predictions = self.classifier.predict(perturbed_adj_norm, x)

            # For an untargeted attack, succeed if the prediction changes from the true label.
            if predictions.detach().max(1)[1] != label:
                return perturbed_adj

        return None  # Return None if no successful attack was found within max_queries.
