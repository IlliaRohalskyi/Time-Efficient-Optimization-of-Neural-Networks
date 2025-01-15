import torch
from torch.utils.data import DataLoader

# Models [src/models/cv.py]
from src.models.cv import (
    ResNet18,
    MobileNetV2,
    EfficientNetB0,
)
# Datasets [src.datasets/cv.py]
from src.datasets.cv import (
    get_cifar10_dataloader,
    get_cifar100_dataloader,
)
# Optimizers [src/optimizers/*.py]
from src.optimizers.sgd import optimize_with_sgd
from src.optimizers.adam import optimize_with_adam
from src.optimizers.pso import optimize_with_pso
from src.optimizers.cma_es import optimize_with_cma
from src.optimizers.sa import optimize_with_sa

# Universal fitness function [src/utility.py]
from src.utility import fitness_function

def get_num_classes(dataloader):
    dataset = dataloader.dataset
    if isinstance(dataset, torch.utils.data.Subset):
        dataset = dataset.dataset
    return len(dataset.classes)

def run_experiment(model, train_loader, test_loader, optimizer_fn):
    optimizer_fn(model, train_loader, test_loader)
    final_train_loss = fitness_function(model, train_loader)
    final_test_loss = fitness_function(model, test_loader)
    return final_train_loss, final_test_loss

def run_cv_experiments():
    classification_dataloaders = [
        get_cifar10_dataloader(batch_size=1024),
        get_cifar100_dataloader(batch_size=1024),
    ]

    classification_models = [
        ResNet18,
        MobileNetV2,
        EfficientNetB0,
    ]

    optimizers = [
        ("PSO", optimize_with_pso),
        ("CMA", optimize_with_cma),
        ("SA", optimize_with_sa),
        ("SGD", optimize_with_sgd),
        ("Adam", optimize_with_adam),
    ]

    for model_class in classification_models:
        model_name = model_class.__name__

        for train_loader, test_loader in classification_dataloaders:
            num_classes = get_num_classes(train_loader)

            for opt_name, opt_fn in optimizers:
                print(f"[classification] Model={model_name}, Optimizer={opt_name}")
                model = model_class(num_classes=num_classes, pretrained=False).to(device)
                t_loss, v_loss = run_experiment(model, train_loader, test_loader, opt_fn)
                print(f"Train Loss: {t_loss}, Test Loss: {v_loss}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    run_cv_experiments()