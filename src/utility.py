"""
Utility functions for neural network optimization.
This module provides utility functions for neural network optimization, including
functions to get the project root directory, flatten and unflatten model weights,
evaluate the fitness of a model, and generate neighboring solutions.
"""

import os
import warnings

import numpy as np
import torch
from torch import nn


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


def fitness_function(weights, model, dataloader):
    """
    Calculates loss (fitness) for classification.

    Args:
        weights (list): List of model weights.
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.

    Returns:
        float: Average loss over the dataloader.
    """
    device = next(model.parameters()).device
    idx = 0
    for param in model.parameters():
        num_params = param.numel()
        param.data = torch.tensor(
            weights[idx : idx + num_params].reshape(param.shape),
            dtype=torch.float32,
            device=device,
        )
        idx += num_params

    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            data, target = batch
            data, target = data.to(device), target.to(device)
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


def optimize_backprop(model, train_loader, val_loader, optimizer, max_iter=5):
    """
    Universal backprop training loop for classification.
    Includes validation after each epoch.
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(max_iter):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            data, target = batch
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                data, target = batch
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(
            f"Epoch {epoch + 1}/{max_iter}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
        )

    return model
