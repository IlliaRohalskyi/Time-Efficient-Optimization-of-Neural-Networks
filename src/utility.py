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


def fitness_function(model, dataloader, task_type):
    """
    Calculates loss (fitness) for classification, segmentation, or detection.
    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the dataset.
        task_type (str): One of 'classification', 'segmentation', 'detection'.
    Returns:
        float: Average loss over the dataloader.
    """
    model.eval()
    total_loss = 0.0
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()  # Default for classification/segmentation

    with torch.no_grad():
        for batch in dataloader:
            if task_type == "classification":
                # Typical batch: (data, targets)
                data, target = batch
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                total_loss += loss.item()

            elif task_type == "segmentation":
                # Typical batch: (data, segmentation_mask)
                data, mask = batch
                data, mask = data.to(device), mask.long().to(device)
                # Segmentation models often return a dict or tensor
                output = model(data)
                # If model returns dict (e.g., DeepLab), get 'out'
                if isinstance(output, dict) and "out" in output:
                    output = output["out"]
                # Use cross-entropy by default for segmentation masks
                loss = criterion(output, mask)
                total_loss += loss.item()

            elif task_type == "detection":
                # Typical batch for detection: (images, targets), where targets have bboxes, labels, etc.
                images, targets = batch
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # Detection models usually return loss dict
                model_out = model(
                    images, targets
                )  # e.g. FasterRCNN returns {'loss_classifier', 'loss_box_reg', ...}
                if isinstance(model_out, dict):
                    loss = sum(model_out.values())  # sum of all detection losses
                else:
                    # In case the model returns something else or just predictions
                    # default to zero or handle differently
                    warnings.warn(
                        "Model output is not a dictionary of losses. Defaulting loss to zero."
                    )
                    loss = torch.tensor(0.0, device=device)
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


def optimize_backprop(
    model, dataloader, optimizer, task_type, max_iter=1000
):
    """
    Universal backprop training loop for classification, segmentation, or detection.
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()  # Default for classification/segmentation
    model.train()

    for _ in range(max_iter):
        for batch in dataloader:
            if task_type == "classification":
                data, target = batch
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)

            elif task_type == "segmentation":
                data, mask = batch
                data, mask = data.to(device), mask.long().to(device)
                outputs = model(data)
                if isinstance(outputs, dict) and "out" in outputs:
                    outputs = outputs["out"]
                loss = criterion(outputs, mask)

            elif task_type == "detection":
                images, targets = batch
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                if isinstance(loss_dict, dict):
                    loss = sum(loss_dict.values())
                else:
                    loss = torch.tensor(0.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model
