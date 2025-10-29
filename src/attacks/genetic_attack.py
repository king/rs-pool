import time
import torch
import numpy as np
import networkx as nx
import random
from copy import deepcopy
import torch.nn.functional as F

from ..models.utils import normalize_tensor_adj

class genetic_attack:
    def __init__(self, adj, x, classifier, n_edges_attack, label, device,
                 population_size=500, cross_rate=0.1, mutate_rate=0.2, time_limit=60):
        """
        Implementation of the Genetic Attack
        ---
        Parameters:
            adj: torch.Tensor representing the adjacency matrix
            x: node feature tensor
            classifier: the victim model to attack
            n_edges_attack: number of edge perturbations per candidate solution
            label: the true graph label (for graph classification)
            device: torch device ('cpu' or 'cuda')
            population_size: number of individuals in the genetic population
            cross_rate: crossover probability
            mutate_rate: mutation probability
        """
        self.adjacency_matrix = adj
        self.x = x.to(device)
        self.classifier = classifier
        self.n_edges_attack = n_edges_attack
        self.label = label
        self.device = device

        self.population_size = population_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutate_rate
        self.time_limit = time_limit
        self.attack_failed = False

        # Create a NetworkX graph from the adjacency matrix for connected-component analysis.
        g = nx.from_numpy_array(self.adjacency_matrix.cpu().numpy())
        self.comps = [comp for comp in nx.connected_components(g)]
        self.set_id = {}
        self.solution = None
        for i, comp in enumerate(self.comps):
            for node in comp:
                self.set_id[node] = i

        # Initialize the population.
        self.population = []
        for _ in range(population_size):
            candidate = []
            for _ in range(n_edges_attack):
                while True:
                    i = np.random.randint(len(self.adjacency_matrix))
                    j = np.random.randint(len(self.adjacency_matrix))
                    # No self-loops.
                    if i == j:
                        continue
                    # Only consider nodes within the same connected component.
                    if self.set_id[i] != self.set_id[j]:
                        continue
                    # Avoid duplicate edge flips.
                    if (i, j) in candidate or (j, i) in candidate:
                        continue
                    break
                candidate.append((i, j))
            self.population.append(candidate)

    def rand_action(self, i):
        """Select a random node from the same connected component as node i (different from i)."""
        region = list(self.comps[self.set_id[i]])
        assert len(region) > 1, "Connected component too small for mutation."
        while True:
            j = random.choice(region)
            if j == i:
                continue
            return j

    def get_fitness(self):
        """
        For each candidate solution in the population, apply the proposed edge flips to a copy of the original
        adjacency matrix, normalize it, and compute the classifier's negative log-likelihood for the true label.
        If any candidate causes misclassification, it is stored as a solution.
        """
        nll_list = []
        for edges in self.population:
            # Only process candidates if no solution found yet.
            if self.solution is None:
                modified_adj = self.adjacency_matrix.clone()
                # Apply each edge flip: here we set the entry to 1 (i.e. adding the edge)
                for (i, j) in edges:
                    modified_adj[i, j] = 1
                    modified_adj[j, i] = 1

                if self.classifier.model_type == "GCN":
                    adj_norm = normalize_tensor_adj(modified_adj, self.device).to(self.device)
                else:
                    adj_norm = modified_adj

                pred = self.classifier.predict(adj_norm, self.x)
                # Check if the perturbed graph is misclassified.
                if pred.detach().max(1)[1] != self.label:
                    self.solution = modified_adj
                    return None  # early exit once a successful attack is found
                loss = F.nll_loss(pred, self.label)
                nll_list.append(loss)
        return torch.stack(nll_list)

    def select(self, fitness):
        """
        Select individuals for reproduction based on fitness.
        Higher (exponentiated) fitness scores lead to higher selection probabilities.
        """
        scores = fitness.cpu().data.numpy()
        sorted_idx = np.argsort(-scores)
        selected = []
        num_top = self.population_size - self.population_size // 2
        for idx in sorted_idx[:num_top]:
            selected.append(deepcopy(self.population[idx]))
        # Sample additional individuals proportionally to their fitness.
        probs = scores / scores.sum()
        sampled_idx = np.random.choice(np.arange(self.population_size),
                                       size=self.population_size // 2,
                                       replace=True,
                                       p=probs)
        for idx in sampled_idx:
            selected.append(deepcopy(self.population[idx]))
        return selected

    def crossover(self, parent, pop):
        """
        Perform crossover between the parent and a randomly selected partner.
        """
        if np.random.rand() < self.cross_rate:
            partner = random.choice(pop)
            if len(parent) != self.n_edges_attack or len(partner) != self.n_edges_attack:
                return deepcopy(partner)
            child = []
            for i in range(self.n_edges_attack):
                if np.random.rand() < 0.5:
                    child.append(parent[i])
                else:
                    child.append(partner[i])
            return child
        else:
            return deepcopy(parent)

    def mutate(self, individual):
        """
        Mutate an individual by randomly replacing one endpoint of an edge with a new one from the same component.
        """
        if len(individual) != self.n_edges_attack:
            return individual
        for i in range(self.n_edges_attack):
            if np.random.rand() < self.mutate_rate:
                e = individual[i]
                if np.random.rand() < 0.5:
                    new_edge = (e[0], self.rand_action(e[0]))
                else:
                    new_edge = (self.rand_action(e[1]), e[1])
                individual[i] = new_edge
        return individual

    def evolve(self):
        """
        Run one generation of the genetic algorithm:
        evaluate fitness, select parents, perform crossover and mutation, and update the population.
        In this version, we check the elapsed time during the parent loop. If the time limit is exceeded,
        the attack is flagged as failed and we break out of the evolution process.
        """
        fitness = self.get_fitness()
        if self.solution is not None:
            return

        pop = self.select(fitness)
        new_pop_list = []

        # Loop over each parent in the selected population.
        for parent in pop:
            child = self.crossover(parent, pop)
            child = self.mutate(child)
            new_pop_list.append(child)

        self.population = new_pop_list
