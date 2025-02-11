"""
This module provides common utility functions for running experiments on neural network models
using various optimizers on classification datasets.
"""

import torch


def get_num_classes(dataloader):
    """
    Returns the number of classes in the dataset associated with the given dataloader.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.

    Returns:
        int: Number of classes in the dataset.
    """
    dataset = dataloader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return len(dataset.classes)


def run_experiment(
    model,
    train_loader,
    test_loader,
    optimizer_fn,
    optimizer_name,
    model_name,
    dataset_name,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Runs an experiment with the given model, data loaders, and optimizer function.

    Args:
        model (torch.nn.Module): The neural network model to be trained and evaluated.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        optimizer_fn (callable): A function that initializes and returns the optimizer.
        optimizer_name (str): The name of the optimizer being used.
        model_name (str): The name of the model being used.
        dataset_name (str): The name of the dataset being used.

    Returns:
        None
    """
    optimizer_fn(
        model,
        train_loader,
        test_loader,
        optimizer_name=optimizer_name,
        model_name=model_name,
        dataset_name=dataset_name,
    )


def run_experiments(classification_dataloaders, classification_models, optimizers):
    """
    Runs a series of experiments with the given classification data loaders, models, and optimizers.

    Args:
        classification_dataloaders (list): List of tuples containing training and testing
          data loaders.
        classification_models (list): List of neural network model classes.
        optimizers (list): List of tuples containing optimizer names and functions.

    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    for model_class in classification_models:
        model_name = model_class.__name__

        for (train_loader, test_loader), dataset_name in classification_dataloaders:
            num_classes = get_num_classes(train_loader)

            for opt_name, opt_fn in optimizers:
                print("\nRunning experiment:")
                print(f"Model: {model_name}")
                print(f"Dataset: {dataset_name}")
                print(f"Optimizer: {opt_name}")

                model = model_class(num_classes=num_classes, pretrained=False).to(
                    device
                )
                run_experiment(
                    model,
                    train_loader,
                    test_loader,
                    opt_fn,
                    optimizer_name=opt_name,
                    model_name=model_name,
                    dataset_name=dataset_name,
                )
