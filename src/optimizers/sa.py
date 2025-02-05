"""
This module provides functionality to optimize the weights of a neural network
model using Simulated Annealing (SA).
Functions:
    optimize_with_sa(model, train_loader, val_loader, initial_temp=1.0, final_temp=0.01,
        cooling_rate=0.95, max_iter=1000):
        Optimize the weights of a neural network model using SA.
Example usage:
    model = YourNeuralNetworkModel()
    train_loader = YourTrainDataLoader()
    val_loader = YourValDataLoader()
    optimized_model = optimize_with_sa(model, train_loader, val_loader)
"""

import numpy as np
import torch

from src.utility import (
    fitness_function,
    flatten_weights,
    generate_neighbor,
    unflatten_weights,
)


def optimize_with_sa(  # pylint: disable=R0913, R0914, R0917
    model,
    train_loader,
    val_loader,
    initial_temp=0.03,
    final_temp=1e-20,
    cooling_rate=0.9,
    max_iter=2,
):
    """
    Optimize the weights of a neural network model using Simulated Annealing (SA).
    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader providing the validation data.
        initial_temp (float, optional): Initial temperature for the SA algorithm.
        final_temp (float, optional): Final temperature for the SA algorithm.
        cooling_rate (float, optional): Cooling rate for the SA algorithm.
        max_iter (int, optional): Maximum number of iterations.
    Returns:
        torch.nn.Module: The optimized neural network model.
    """
    criterion = torch.nn.CrossEntropyLoss()
    current_weights = flatten_weights(model)
    current_loss = fitness_function(current_weights, model, train_loader)
    best_weights = current_weights.copy()
    best_loss = current_loss

    temp = initial_temp
    iteration = 0

    while temp > final_temp and iteration < max_iter:
        neighbor_weights = generate_neighbor(current_weights, temp)
        neighbor_loss = fitness_function(neighbor_weights, model, train_loader)
        delta_loss = neighbor_loss - current_loss

        if delta_loss < 0 or np.random.random() < np.exp(-delta_loss / temp):
            current_weights = neighbor_weights
            current_loss = neighbor_loss
            if current_loss < best_loss:
                best_weights = current_weights.copy()
                best_loss = current_loss

        temp *= cooling_rate
        iteration += 1
        print(
            f"Iteration {iteration}, Temperature: {temp:.4f}, Current Loss: {current_loss:.4f}, Best Loss: {best_loss:.4f}"
        )

        # Validation phase
        val_loss = fitness_function(best_weights, model, val_loader)
        print(f"Iteration {iteration}, Validation Loss: {val_loss:.4f}")

    unflatten_weights(model, best_weights)
    return model
