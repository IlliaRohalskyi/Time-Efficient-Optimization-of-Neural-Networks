"""
This module provides DataLoader functions for various computer vision datasets.
The datasets included are:
1. MNIST
2. Fashion-MNIST
3. CIFAR-10
4. CIFAR-100
5. ImageNet (subset)
6. COCO (Object Detection)
7. Pascal VOC (Object Detection and Segmentation)
8. Cityscapes (Image Segmentation)
9. ADE20K (Image Segmentation)
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import CocoDetection, VOCDetection, VOCSegmentation
from torchvision.transforms import functional as F

def get_mnist_dataloader(batch_size=32, train=True):
    """
    Returns a DataLoader for the MNIST dataset.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        train (bool, optional): If True, returns the training set. Otherwise, returns the test set. Defaults to True.

    Returns:
        DataLoader: DataLoader for the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_fashion_mnist_dataloader(batch_size=32, train=True):
    """
    Returns a DataLoader for the Fashion-MNIST dataset.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        train (bool, optional): If True, returns the training set. Otherwise, returns the test set. Defaults to True.

    Returns:
        DataLoader: DataLoader for the Fashion-MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.FashionMNIST(root='data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_cifar10_dataloader(batch_size=32, train=True):
    """
    Returns a DataLoader for the CIFAR-10 dataset.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        train (bool, optional): If True, returns the training set. Otherwise, returns the test set. Defaults to True.

    Returns:
        DataLoader: DataLoader for the CIFAR-10 dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    dataset = datasets.CIFAR10(root='data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_cifar100_dataloader(batch_size=32, train=True):
    """
    Returns a DataLoader for the CIFAR-100 dataset.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        train (bool, optional): If True, returns the training set. Otherwise, returns the test set. Defaults to True.

    Returns:
        DataLoader: DataLoader for the CIFAR-100 dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    dataset = datasets.CIFAR100(root='data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_imagenet_dataloader(batch_size=32, train=True):
    """
    Returns a DataLoader for the ImageNet dataset (subset).

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        train (bool, optional): If True, returns the training set. Otherwise, returns the validation set. Defaults to True.

    Returns:
        DataLoader: DataLoader for the ImageNet dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    split = 'train' if train else 'val'
    dataset = datasets.ImageNet(root='data', split=split, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_coco_dataloader(batch_size=32, train=True):
    """
    Returns a DataLoader for the COCO dataset (Object Detection).

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        train (bool, optional): If True, returns the training set. Otherwise, returns the validation set. Defaults to True.

    Returns:
        DataLoader: DataLoader for the COCO dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    split = 'train2017' if train else 'val2017'
    dataset = CocoDetection(root=f'data/coco/{split}', annFile=f'data/coco/annotations/instances_{split}.json', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_voc_dataloader(batch_size=32, year='2012', train=True, task='detection'):
    """
    Returns a DataLoader for the Pascal VOC dataset.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        year (str, optional): Year of the dataset (2007 or 2012). Defaults to '2012'.
        train (bool, optional): If True, returns the training set. Otherwise, returns the validation set. Defaults to True.
        task (str, optional): Task type ('detection' or 'segmentation'). Defaults to 'detection'.

    Returns:
        DataLoader: DataLoader for the Pascal VOC dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_set = 'train' if train else 'val'
    if task == 'detection':
        dataset = VOCDetection(root='data', year=year, image_set=image_set, download=True, transform=transform)
    else:
        dataset = VOCSegmentation(root='data', year=year, image_set=image_set, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_cityscapes_dataloader(batch_size=32, train=True):
    """
    Returns a DataLoader for the Cityscapes dataset (Image Segmentation).

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        train (bool, optional): If True, returns the training set. Otherwise, returns the validation set. Defaults to True.

    Returns:
        DataLoader: DataLoader for the Cityscapes dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    split = 'train' if train else 'val'
    dataset = datasets.Cityscapes(root='data', split=split, mode='fine', target_type='semantic', transform=transform, target_transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def get_ade20k_dataloader(batch_size=32, train=True):
    """
    Returns a DataLoader for the ADE20K dataset (Image Segmentation).

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        train (bool, optional): If True, returns the training set. Otherwise, returns the validation set. Defaults to True.

    Returns:
        DataLoader: DataLoader for the ADE20K dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    split = 'training' if train else 'validation'
    dataset = datasets.ADE20K(root='data', split=split, transform=transform, target_transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)