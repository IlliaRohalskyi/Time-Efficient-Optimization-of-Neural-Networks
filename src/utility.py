"""
Utility functions for neural network optimization.
This module provides utility functions for neural network optimization, including
functions to get the project root directory, flatten and unflatten model weights,
evaluate the fitness of a model, and generate neighboring solutions.
"""

import os
import time

import GPUtil
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.datasets.cv import get_cifar10_dataloader, get_cifar100_dataloader
from src.models.cv import EfficientNetB0, MobileNetV2, ResNet18


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


def optimize_backprop(  # pylint: disable=R0913, R0914, R0917, R0915
    model,
    train_loader,
    val_loader,
    optimizer,
    max_iter=5,
    dataset_name="",
    model_name="",
    optimizer_name="",
):
    """
    Universal backprop training loop for classification with metrics logging.
    """
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    # Initialize metrics storage
    metrics = []

    for epoch in range(max_iter):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []

        for batch in train_loader:
            data, target = batch
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward(create_graph=True)
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(target.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                data, target = batch
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(target.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        # Get memory usage
        gpu = GPUtil.getGPUs()[0]
        vram_used = gpu.memoryUsed
        ram_used = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Time for epoch
        epoch_time = time.time() - epoch_start_time

        # Store metrics
        metrics.append(
            {
                "Dataset": dataset_name,
                "Model": model_name,
                "Optimizer": optimizer_name,
                "Epoch": epoch + 1,
                "Time(s)": epoch_time,
                "VRAM(MB)": vram_used,
                "RAM(MB)": ram_used,
                "Train_Loss": train_loss,
                "Train_Acc": train_acc,
                "Train_F1": train_f1,
                "Val_Loss": val_loss,
                "Val_Acc": val_acc,
                "Val_F1": val_f1,
            }
        )

        print(f"Epoch {epoch + 1}/{max_iter}")
        print(f"Time: {epoch_time:.2f}s, VRAM: {vram_used}MB, RAM: {ram_used:.1f}MB")
        print(
            f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}"
        )
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    df = pd.DataFrame(metrics)
    excel_path = os.path.join(get_root(), "optimization_metrics.xlsx")

    if not pd.io.common.file_exists(excel_path):
        df.to_excel(excel_path, index=False)
    else:
        with pd.ExcelWriter(excel_path, mode="a", if_sheet_exists="overlay") as writer:
            existing_df = pd.read_excel(excel_path)
            pd.concat([existing_df, df]).to_excel(writer, index=False)

    return model


def get_dataloaders_and_models(batch_size):
    """
    Returns a list of data loaders for classification datasets and a list of classification models.
    Args:
        batch_size (int): The batch size to be used for the data loaders.
    Returns:
        tuple: A tuple containing:
            - classification_dataloaders (list): A list of tuples where each tuple contains a data
              loader and the corresponding dataset name.
            - classification_models (list): A list of classification model classes.
    """

    classification_dataloaders = [
        (get_cifar10_dataloader(batch_size=batch_size), "CIFAR10"),
        (get_cifar100_dataloader(batch_size=batch_size), "CIFAR100"),
    ]

    classification_models = [
        ResNet18,
        MobileNetV2,
        EfficientNetB0,
    ]
    return classification_dataloaders, classification_models


def run_optimizer(
    model,
    train_loader,
    val_loader,
    optimizer,
    max_iter,
    dataset_name,
    model_name,
    optimizer_name,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Run the optimizer with the given parameters.
    """
    return optimize_backprop(
        model,
        train_loader,
        val_loader,
        optimizer,
        max_iter,
        dataset_name,
        model_name,
        optimizer_name,
    )
