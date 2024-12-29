"""
Utility functions for neural network optimization.
This module provides utility functions for neural network optimization, including
functions to get the project root directory, flatten and unflatten model weights,
evaluate the fitness of a model, and generate neighboring solutions.
"""

import os

import numpy as np
import torch


def get_root():
    """
    Get Project Root Directory.

    This function determines the root directory of a project based on the location of the script.
    It navigates upwards in the directory tree until it finds the setup.py file.

    Returns:
        str: The absolute path of the project's root directory.
    """
    script_path = os.path.abspath(__file__)

    # Navigate upwards in the directory tree until you find the setup.py file
    while not os.path.exists(os.path.join(script_path, "setup.py")):
        script_path = os.path.dirname(script_path)

    return script_path


def flatten_weights(model):
    """
    Flattens the trainable parameters of a PyTorch model into a single 1D NumPy array.
    Args:
        model (torch.nn.Module):
            The neural network model whose parameters are to be flattened.
    Returns:
        numpy.ndarray:
            A 1D array containing all trainable parameters of the model.
    """

    return np.concatenate(
        [param.detach().cpu().numpy().flatten() for param in model.parameters()]
    )


def unflatten_weights(model, weights):
    """
    Reconstructs the model parameters from a flattened array of weights.
    Args:
        model (torch.nn.Module): The PyTorch model whose parameters will be set.
        weights (list or ndarray): A flattened representation of all model parameters
            in the exact order taken from model.parameters().
    Returns:
        None
        This function updates the model parameters in-place.
    """

    idx = 0
    for param in model.parameters():
        num_params = param.numel()
        param.data = torch.tensor(
            weights[idx : idx + num_params].reshape(param.shape), dtype=torch.float32
        )
        idx += num_params


def fitness_function(weights, model, dataloader, criterion):
    """
    Evaluate the fitness of a neural network model
    by computing the average loss over the provided dataset.
    Args:
        weights (iterable): Flattened weights to be unflattened and assigned to the model.
        model (torch.nn.Module): The neural network model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the dataset for evaluation.
        criterion (torch.nn.modules.loss._Loss): Compute the loss between outputs and targets.
    Returns:
        float: The average loss over all batches in the dataloader.
    """

    unflatten_weights(model, weights)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def generate_neighbor(weights, temperature):
    """
    Generate a neighboring solution by adding a perturbation to the weights.
    Args:
        weights (numpy.ndarray): Current weights of the neural network.
        temperature (float): Standard deviation of the normal distribution for perturbation.
    Returns:
        numpy.ndarray: New weights after perturbation.
    """

    perturbation = np.random.normal(0, temperature, size=weights.shape)
    return weights + perturbation


def optimize_backprop(model, dataloader, optimizer, max_iter=1000):
    """
    Run a standard backprop training loop on the given model.

    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): An initialized optimizer instance.
        max_iter (int, optional): Maximum number of epochs/iterations to train.

    Returns:
        torch.nn.Module: The trained model.
    """
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(max_iter):
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
