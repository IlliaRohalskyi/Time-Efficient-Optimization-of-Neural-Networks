"""
This module provides AdamW optimization for neural network training.
"""

from torch.optim import AdamW

from src.utility import run_optimizer


def optimize_with_adamw(
    model,
    train_loader,
    val_loader,
    optimizer_name="AdamW",
    model_name="Unknown",
    dataset_name="Unknown",
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    max_iter=50,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, duplicate-code
    """
    Train a model using the AdamW optimizer.
    """
    print(
        f"\nStarting {optimizer_name} optimization for {model_name} on {dataset_name}"
    )
    optimizer = AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
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
