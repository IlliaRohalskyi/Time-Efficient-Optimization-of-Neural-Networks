"""
This module runs experiments on neural network models using various optimization algorithms without 
gradient computation.
Functions:
    get_num_classes(dataloader):
        Returns the number of classes in the dataset associated with the given dataloader.
    run_experiment(model, train_loader, test_loader, optimizer_fn):
        Runs a single experiment using the specified model, dataloaders, and optimizer function.
    run_no_grad_experiments():
        Runs experiments on classification models using different optimization algorithms without 
        gradient computation.
Main Execution:
    If this script is run as the main module, it sets the device to GPU if available, otherwise 
    CPU, and runs the no-grad experiments.
"""

import torch

from src.optimizers.adam import optimize_with_adam
from src.optimizers.cma_es import optimize_with_cma
from src.optimizers.pso import optimize_with_pso
from src.optimizers.sa import optimize_with_sa
from src.optimizers.sgd import optimize_with_sgd
from src.utility import get_dataloaders_and_models


def get_num_classes(dataloader):
    """
    Returns the number of classes in the dataset associated with the given dataloader.
    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader containing the dataset.
    Returns:
        int: The number of classes in the dataset.
    """

    dataset = dataloader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return len(dataset.classes)


def run_experiment(model, train_loader, test_loader, optimizer_fn):
    """
    Runs an experiment by applying the given optimizer function to the model using the
      provided data loaders.
    Args:
        model (torch.nn.Module): The neural network model to be optimized.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        test_loader (torch.utils.data.DataLoader): DataLoader for the testing dataset.
        optimizer_fn (function): A function that takes the model, train_loader, and test_loader
          as arguments and performs optimization.
    Returns:
        None
    """

    optimizer_fn(model, train_loader, test_loader)


def run_no_grad_experiments():
    """
    Run a series of experiments on classification models without gradient computation.
    This function initializes dataloaders and models for classification tasks, then iterates
    over a list of optimization algorithms to train and evaluate each model. The results
    of each experiment are printed to the console.
    The following optimization algorithms are used:
    - Simulated Annealing (SA)
    - Particle Swarm Optimization (PSO)
    - Covariance Matrix Adaptation (CMA)
    - Stochastic Gradient Descent (SGD)
    - Adam
    The function performs the following steps:
    1. Retrieves dataloaders and models for classification tasks.
    2. Iterates over each model class.
    3. For each model, iterates over each combination of training and testing dataloaders.
    4. For each combination of dataloaders, iterates over each optimization algorithm.
    5. Prints the model name and optimizer name.
    6. Initializes the model and runs the experiment using the specified optimizer.
    Note: The models are initialized without pre-trained weights and moved to the specified device.
    Args:
        None
    Returns:
        None
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    classification_dataloaders, classification_models = get_dataloaders_and_models(
        batch_size=1024
    )

    optimizers = [
        ("SA", optimize_with_sa),
        ("PSO", optimize_with_pso),
        ("CMA", optimize_with_cma),
        ("SGD", optimize_with_sgd),
        ("Adam", optimize_with_adam),
    ]

    for model_class in classification_models:
        model_name = model_class.__name__

        for (train_loader, test_loader), _ in classification_dataloaders:
            num_classes = get_num_classes(train_loader)

            for opt_name, opt_fn in optimizers:
                print(f"[classification] Model={model_name}, Optimizer={opt_name}")
                model = model_class(num_classes=num_classes, pretrained=False).to(
                    device
                )
                run_experiment(model, train_loader, test_loader, opt_fn)


if __name__ == "__main__":
    run_no_grad_experiments()
