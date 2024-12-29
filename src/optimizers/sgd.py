"""
This module provides an SGD optimization function for neural network training.
"""

from torch.optim import SGD

from src.utility import optimize_backprop


def optimize_with_sgd(model, dataloader, lr=0.01, max_iter=1000):
    """
    Train a model using the SGD optimizer.

    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        dataloader (torch.utils.data.DataLoader): DataLoader providing the training data.
        lr (float, optional): Learning rate for SGD.
        max_iter (int, optional): Maximum number of training iterations.

    Returns:
        torch.nn.Module: The trained model.
    """
    optimizer = SGD(model.parameters(), lr=lr)
    return optimize_backprop(model, dataloader, optimizer, max_iter)
