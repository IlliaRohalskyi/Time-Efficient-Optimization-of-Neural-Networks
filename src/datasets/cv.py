"""
This module provides DataLoader functions for CIFAR-10 and CIFAR-100 datasets.
"""

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_cifar10_dataloader(batch_size=32, split_ratio=0.8):
    """
    Returns train and test DataLoaders for the CIFAR-10 dataset with an 80/20 split.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        split_ratio (float, optional): Ratio of training data. Defaults to 0.8.

    Returns:
        tuple: (train_loader, test_loader) DataLoaders for the CIFAR-10 dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    train_len = int(split_ratio * len(dataset))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader

def get_cifar100_dataloader(batch_size=32, split_ratio=0.8):
    """
    Returns train and test DataLoaders for the CIFAR-100 dataset with an 80/20 split.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        split_ratio (float, optional): Ratio of training data. Defaults to 0.8.

    Returns:
        tuple: (train_loader, test_loader) DataLoaders for the CIFAR-100 dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
    train_len = int(split_ratio * len(dataset))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader