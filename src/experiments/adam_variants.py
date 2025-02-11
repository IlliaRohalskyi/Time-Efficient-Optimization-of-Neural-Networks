"""
This script runs experiments to compare different variants of the Adam optimizer on 
classification datasets.
Modules:
    - torch: PyTorch library for tensor computations and neural networks.
    - src.optimizers: Custom modules for different Adam variants (Adam, AdamW, Adamax, AdaBelief, 
      RAdam, QHAdam).
    - src.experiments.common: Common utility functions for running experiments.
    - src.utility: Custom utility functions for fitness evaluation, weight flattening, and 
      retrieving dataloaders and models.
Functions:
    - run_adam_variants_experiments(): Runs a series of experiments on different classification 
      models using various Adam variants.
"""

from src.experiments.common import run_experiments
from src.optimizers.adabelief import optimize_with_adabelief
from src.optimizers.adam import optimize_with_adam
from src.optimizers.adamax import optimize_with_adamax
from src.optimizers.adamw import optimize_with_adamw
from src.optimizers.qh_adam import optimize_with_qhadam
from src.optimizers.radam import optimize_with_radam
from src.utility import get_dataloaders_and_models


def run_adam_variants_experiments():
    """
    Runs a series of experiments to evaluate different variants of the Adam optimizer on
    various classification models and datasets.

    The function performs the following steps:
    1. Retrieves classification data loaders and models.
    2. Defines a list of Adam variants to be tested.
    3. Iterates over each model and dataset combination.
    4. For each combination, iterates over each Adam variant.
    5. Prints the current experiment details.
    6. Initializes the model and runs the experiment with the specified Adam variant.

    Adam variants tested:
    - Adam
    - AdamW
    - Adamax
    - AdaBelief
    - RAdam
    - QHAdam

    The results of each experiment are printed to the console.

    Returns:
        None
    """
    classification_dataloaders, classification_models = get_dataloaders_and_models(
        batch_size=512
    )

    optimizers = [
        ("Adam", optimize_with_adam),
        ("AdamW", optimize_with_adamw),
        ("Adamax", optimize_with_adamax),
        ("AdaBelief", optimize_with_adabelief),
        ("RAdam", optimize_with_radam),
        ("QHAdam", optimize_with_qhadam),
    ]

    run_experiments(classification_dataloaders, classification_models, optimizers)


if __name__ == "__main__":
    run_adam_variants_experiments()
