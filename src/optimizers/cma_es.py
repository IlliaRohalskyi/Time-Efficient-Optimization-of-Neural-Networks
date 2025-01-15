"""
This module provides functionality to optimize the weights of a neural network
model using the Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
Functions:
    optimize_with_cma(model, train_loader, val_loader, max_generations=10, population_size=20, sigma=0.1):
        Optimize the weights of a neural network model using CMA-ES.
Example usage:
    model = YourNeuralNetworkModel()
    train_loader = YourTrainDataLoader()
    val_loader = YourValDataLoader()
    optimized_model = optimize_with_cma(model, train_loader, val_loader)
"""

import cma
import torch

from src.utility import fitness_function, flatten_weights, unflatten_weights


def optimize_with_cma(
    model,
    train_loader,
    val_loader,
    max_generations=5,
    population_size=5,
    sigma=0.1,
):
    """
    Optimize the weights of a neural network model using the
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        train_loader (torch.utils.data.DataLoader): DataLoader providing the training data.
        val_loader (torch.utils.data.DataLoader): DataLoader providing the validation data.
        max_generations (int, optional): Maximum number of generations for optimization process.
        population_size (int, optional): The population size for the CMA-ES algorithm.
        sigma (float, optional): The initial standard deviation for the CMA-ES algorithm.
    Returns:
        torch.nn.Module: The optimized neural network model.
    """

    initial_weights = flatten_weights(model)
    es = cma.CMAEvolutionStrategy(initial_weights, sigma, {"popsize": population_size})

    for generation in range(max_generations):
        solutions = es.ask()
        fitnesses = [
            fitness_function(weights, model, train_loader)
            for weights in solutions
        ]
        es.tell(solutions, fitnesses)

        best_fitness = min(fitnesses)
        print(
            f"Generation {generation + 1}/{max_generations} - Best Fitness: {best_fitness}"
        )
        if best_fitness < 0.1:
            print(f"Converged at generation {generation + 1}")
            break

        # Validation phase
        best_weights = es.result.xbest
        unflatten_weights(model, best_weights)
        val_loss = fitness_function(best_weights, model, val_loader)
        print(f"Generation {generation + 1}/{max_generations}, Validation Loss: {val_loss}")

    unflatten_weights(model, es.result.xbest)
    return model