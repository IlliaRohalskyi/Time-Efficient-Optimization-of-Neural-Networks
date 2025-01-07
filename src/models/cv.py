"""
This module provides implementations of various computer vision models
for image classification, segmentation, and object detection tasks.
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


class UNet(nn.Module):
    """
    UNet model for image segmentation.
    """

    def __init__(self, pretrained=True, num_classes=1):
        """
        Initializes the UNet model.

        Args:
            pretrained (bool): If True, use pretrained weights.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.encoder = models.resnet18(pretrained=pretrained)
        self.encoder_layers = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class DeepLabV3(nn.Module):
    """
    DeepLabV3 model for image segmentation.
    """

    def __init__(self, pretrained=True, num_classes=21):
        """
        Initializes the DeepLabV3 model.

        Args:
            pretrained (bool): If True, use pretrained weights.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)["out"]


class FasterRCNN(nn.Module):
    """
    Faster R-CNN model for object detection.
    """

    def __init__(self, pretrained=True, num_classes=91):
        """
        Initializes the Faster R-CNN model.

        Args:
            pretrained (bool): If True, use pretrained weights.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
        in_feats = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (
            models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
        )

    def forward(self, x, targets=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            targets (dict, optional): Ground truth boxes and labels.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x, targets)


class SSD(nn.Module):
    """
    SSD model for object detection.
    """

    def __init__(self, pretrained=True, num_classes=91):
        """
        Initializes the SSD model.

        Args:
            pretrained (bool): If True, use pretrained weights.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.model = models.detection.ssd300_vgg16(pretrained=pretrained)
        in_feats = self.model.head.classification_head.conv_cls[0].in_channels
        self.model.head.classification_head.num_classes = num_classes
        self.model.head.classification_head.conv_cls = nn.Conv2d(
            in_feats, num_classes * 4, kernel_size=3, padding=1
        )

    def forward(self, x, targets=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.
            targets (dict, optional): Ground truth boxes and labels.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x, targets)


class YOLOv3(nn.Module):
    """
    YOLOv3 model for object detection.
    """

    def __init__(self, pretrained=True, num_classes=80):
        """
        Initializes the YOLOv3 model.

        Args:
            pretrained (bool): If True, use pretrained weights.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        self.detect_head = nn.Conv2d(512, num_classes * 5, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.backbone(x)
        return self.detect_head(x)
