"""
This module provides an Adam optimization function for neural network training.
"""

from torch.optim import Adam

from src.utility import run_optimizer


def optimize_with_adam(
    model,
    train_loader,
    val_loader,
    optimizer_name="Adam",
    model_name="Unknown",
    dataset_name="Unknown",
    lr=0.001,
    betas=(0.9, 0.999),
    max_iter=50,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, duplicate-code
    """
    Train a model using the Adam optimizer.
    """
    print(
        f"\nStarting {optimizer_name} optimization for {model_name} on {dataset_name}"
    )
    optimizer = Adam(model.parameters(), lr=lr, betas=betas)
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
