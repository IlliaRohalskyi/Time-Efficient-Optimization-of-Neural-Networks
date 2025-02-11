"""
This module provides AdaBelief optimization for neural network training.
"""

import torch_optimizer as optim

from src.utility import run_optimizer


def optimize_with_adabelief(
    model,
    train_loader,
    val_loader,
    optimizer_name="AdaBelief",
    model_name="Unknown",
    dataset_name="Unknown",
    lr=0.001,
    max_iter=50,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, duplicate-code
    """
    Train a model using the AdaBelief optimizer.
    """
    print(
        f"\nStarting {optimizer_name} optimization for {model_name} on {dataset_name}"
    )
    optimizer = optim.AdaBelief(model.parameters(), lr=lr)
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
