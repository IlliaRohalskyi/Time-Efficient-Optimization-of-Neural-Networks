"""
This module provides functionality to optimize the weights of a neural network model
using Particle Swarm Optimization (PSO).
Functions:
    optimize_with_pso(model, dataloader, max_iter=100, n_particles=50):
        Optimize the weights of a neural network model using PSO.
Example usage:
    model = YourNeuralNetworkModel()
    dataloader = YourDataLoader()
    optimized_model = optimize_with_pso(model, dataloader)
"""

import numpy as np
import torch

from src.utility import fitness_function, unflatten_weights


class PSO:  # pylint: disable=R0903
    """
    Particle Swarm Optimization (PSO) algorithm implementation.
    Attributes:
        positions (np.ndarray): Current positions of the particles.
        velocities (np.ndarray): Current velocities of the particles.
        pbest_positions (np.ndarray): Best known positions of the particles.
        pbest_scores (np.ndarray): Best known scores of the particles.
        gbest_position (np.ndarray): Best known position of the swarm.
        gbest_score (float): Best known score of the swarm.
        bounds (tuple): Bounds for the search space as (lower, upper).
    Methods:
        __init__(n_particles, dim, bounds=None): Initializes the PSO optimizer.
        optimize(fitness_func, max_iter): Optimizes the given fitness function.
    """

    def __init__(self, n_particles, dim, bounds=None):
        """
        Initialize the PSO optimizer.
        Args:
            n_particles (int): Number of particles in the swarm.
            dim (int): Dimensionality of the search space.
            bounds (tuple, optional): Bounds for the search space as (lower, upper).
        """
        self.positions = np.random.randn(n_particles, dim)
        self.velocities = np.random.randn(n_particles, dim) * 0.1
        self.pbest_positions = self.positions.copy()
        self.pbest_scores = np.full(n_particles, np.inf)
        self.gbest_position = None
        self.gbest_score = np.inf
        self.bounds = bounds

    def optimize(self, fitness_func, max_iter):
        """
        Optimize using the PSO algorithm.
        Args:
            fitness_func (callable): The fitness function to minimize.
            max_iter (int): Maximum number of iterations.
        Returns:
            np.ndarray: The best position (weights) found by the swarm.
        """
        for iteration in range(max_iter):
            for i, _ in enumerate(self.positions):
                fitness = fitness_func(self.positions[i])
                if fitness < self.pbest_scores[i]:
                    self.pbest_scores[i] = fitness
                    self.pbest_positions[i] = self.positions[i]
                if fitness < self.gbest_score:
                    self.gbest_score = fitness
                    self.gbest_position = self.positions[i]

            r1, r2 = np.random.rand(2)
            self.velocities = (
                0.729 * self.velocities
                + 2.05 * r1 * (self.pbest_positions - self.positions)
                + 2.05 * r2 * (self.gbest_position - self.positions)
            )
            self.positions += self.velocities
            if self.bounds:
                self.positions = np.clip(self.positions, *self.bounds)

            print(f"Iteration {iteration + 1} - Best Fitness: {self.gbest_score}")
            if self.gbest_score < 0.1:
                break
        return self.gbest_position


def optimize_with_pso(
    model, dataloader, task_type="classification", max_iter=100, n_particles=50
):
    """
    Optimize the weights of a neural network model using Particle Swarm Optimization (PSO).
    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.
        max_iter (int, optional): Maximum number of iterations for the optimization process.
        n_particles (int, optional): Number of particles in the swarm. Default is 50.
    Returns:
        torch.nn.Module: The optimized neural network model.
    """
    criterion = torch.nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    pso = PSO(n_particles, n_params)

    def fitness_wrapper(weights):
        return fitness_function(weights, model, dataloader, criterion, task_type)

    best_weights = pso.optimize(fitness_wrapper, max_iter)
    unflatten_weights(model, best_weights)
    return model
