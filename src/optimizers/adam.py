"""
This module provides an Adam optimization function for neural network training.
"""

from torch.optim import Adam

from src.utility import optimize_backprop


def optimize_with_adam(model, dataloader, task_type="classification", lr=0.001, max_iter=1000):
    """
    Train a model using the Adam optimizer.

    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.
        lr (float, optional): Learning rate for Adam.
        max_iter (int, optional): Maximum number of training iterations.

    Returns:
        torch.nn.Module: The trained model.
    """
    optimizer = Adam(model.parameters(), lr=lr)
    return optimize_backprop(model, dataloader, optimizer, task_type, max_iter)
