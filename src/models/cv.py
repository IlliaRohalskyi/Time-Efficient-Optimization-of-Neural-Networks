"""
This module provides implementations of various computer vision models
for image classification tasks.
"""

from torch import nn
from torchvision import models


class ResNet18(nn.Module):
    """
    ResNet18 model for image classification.
    """

    def __init__(self, pretrained=True, num_classes=100):
        """
        Initializes the ResNet18 model.

        Args:
            pretrained (bool): If True, use pretrained weights.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 model for image classification.
    """

    def __init__(self, pretrained=True, num_classes=100):
        """
        Initializes the MobileNetV2 model.

        Args:
            pretrained (bool): If True, use pretrained weights.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)


class EfficientNetB0(nn.Module):
    """
    EfficientNetB0 model for image classification.
    """

    def __init__(self, pretrained=True, num_classes=100):
        """
        Initializes the EfficientNetB0 model.

        Args:
            pretrained (bool): If True, use pretrained weights.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)
