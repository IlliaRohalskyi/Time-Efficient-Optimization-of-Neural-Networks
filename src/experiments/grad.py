"""
This script runs experiments to optimize neural network models using various optimizers on 
classification datasets.
Modules:
    - torch: PyTorch library for tensor computations and neural networks.
    - src.optimizers: Custom modules for different optimization algorithms (AdaHessian, Adam, 
      AdamW, Nesterov, RangerQH, Shampoo, Yogi, SGD).
    - src.experiments.common: Common utility functions for running experiments.
    - src.utility: Custom utility functions for fitness evaluation, weight flattening, and 
      retrieving dataloaders and models.
Functions:
    - run_new_experiments(): Runs a series of experiments on different classification models using 
      various optimizers.
"""

from src.experiments.common import run_experiments
from src.optimizers.adahessian import optimize_with_adahessian
from src.optimizers.adam import optimize_with_adam
from src.optimizers.adamw import optimize_with_adamw
from src.optimizers.nag import optimize_with_nesterov
from src.optimizers.ranger_qh import optimize_with_ranger_qh
from src.optimizers.sgd import optimize_with_sgd
from src.optimizers.shampoo import optimize_with_shampoo
from src.optimizers.yogi import optimize_with_yogi
from src.utility import get_dataloaders_and_models


def run_new_experiments():
    """
    Runs a series of experiments to evaluate different optimizers on various classification models
    and datasets.

    The function performs the following steps:
    1. Retrieves classification data loaders and models.
    2. Defines a list of optimizers to be tested.
    3. Iterates over each model and dataset combination.
    4. For each combination, iterates over each optimizer.
    5. Prints the current experiment details.
    6. Initializes the model and runs the experiment with the specified optimizer.

    Optimizers tested:
    - AdaHessian
    - AdamW
    - Nesterov
    - RangerQH
    - Shampoo
    - Yogi
    - SGD
    - Adam

    The results of each experiment are printed to the console.

    Returns:
        None
    """
    classification_dataloaders, classification_models = get_dataloaders_and_models(
        batch_size=512
    )

    optimizers = [
        ("AdaHessian", optimize_with_adahessian),
        ("AdamW", optimize_with_adamw),
        ("Nesterov", optimize_with_nesterov),
        ("RangerQH", optimize_with_ranger_qh),
        ("Shampoo", optimize_with_shampoo),
        ("Yogi", optimize_with_yogi),
        ("SGD", optimize_with_sgd),
        ("Adam", optimize_with_adam),
    ]

    run_experiments(classification_dataloaders, classification_models, optimizers)


if __name__ == "__main__":
    run_new_experiments()
