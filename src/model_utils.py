"""
Shared utilities and constants for the brain MRI tumor detector.

This module contains common functions and constants used across
training, inference, and model comparison.
"""

from typing import Dict

import torch.nn as nn
from torchvision import models

# Model file naming convention
MODEL_FILES: Dict[str, str] = {
    "resnet18": "resnet18_brain_mri.pth",
    "vgg16": "vgg16_brain_mri.pth",
    "densenet121": "densenet121_brain_mri.pth",
    "efficientnet_b0": "efficientnet_b0_brain_mri.pth",
    "mobilenet_v2": "mobilenet_v2_brain_mri.pth"
}

# Training hyperparameters
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_NUM_EPOCHS = 10
DEFAULT_BATCH_SIZE = 16

# Image preprocessing constants
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# MRI validation constants (for inference.py)
# Typical MRI images have grayscale intensities in mid-range (80-120)
MRI_GRAYSCALE_MIN = 80
MRI_GRAYSCALE_MAX = 120


def get_model_architecture(model_name: str, num_classes: int = 1) -> nn.Module:
    """
    Get model architecture by name.

    This function provides a centralized way to create model architectures
    with the correct output layer modifications for binary classification.

    Args:
        model_name: Name of the model (resnet18, vgg16, densenet121, 
                    efficientnet_b0, mobilenet_v2)
        num_classes: Number of output classes (1 for binary classification 
                     with BCEWithLogitsLoss)

    Returns:
        PyTorch model with modified final layer

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def get_model_architecture_without_weights(model_name: str, num_classes: int = 1) -> nn.Module:
    """
    Get model architecture without pretrained weights (for loading checkpoints).

    Args:
        model_name: Name of the model
        num_classes: Number of output classes

    Returns:
        PyTorch model with modified final layer (no pretrained weights)

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model
