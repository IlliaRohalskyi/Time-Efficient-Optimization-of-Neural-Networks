"""
This module provides RangerQH optimization for neural network training.
"""

import torch_optimizer as optim

from src.utility import run_optimizer


def optimize_with_ranger_qh(
    model,
    train_loader,
    val_loader,
    optimizer_name="RangerQH",
    model_name="Unknown",
    dataset_name="Unknown",
    lr=0.001,
    max_iter=50,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, duplicate-code
    """
    Train a model using the RangerQH optimizer.
    """
    print(
        f"\nStarting {optimizer_name} optimization for {model_name} on {dataset_name}"
    )
    optimizer = optim.RangerQH(model.parameters(), lr=lr)
    return run_optimizer(
        model,
        train_loader,
        val_loader,
        optimizer,
        max_iter,
        dataset_name,
        model_name,
        optimizer_name,
    )
