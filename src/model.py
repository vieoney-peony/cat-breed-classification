"""Model definitions for cat breed classification."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchsummary import summary

LIST_BACKBONES = models.list_models(module=torchvision.models)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


class CatBreedClassifier(nn.Module):
    """Cat breed classification model using pre-trained backbone."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "alexnet",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
    ):
        """
        Initialize the model.

        Args:
            num_classes: Number of classes to predict
            backbone: Model backbone (e.g. shufflenetv2, mobilenetv2)
            pretrained: Whether to use pre-trained weights
            dropout_rate: Dropout rate for classifier head
        """
        super().__init__()

        if backbone not in LIST_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. Available backbones: {LIST_BACKBONES}"
            )
        # Load the backbone model
        if pretrained:
            self.backbone = models.get_model(backbone, weights="DEFAULT")
        else:
            self.backbone = models.get_model(backbone, weights=None)

        feature_dim = (
            self.backbone.fc.in_features
            if hasattr(self.backbone, "fc")
            else self.backbone.classifier[1].in_features
        )

        if hasattr(self.backbone, "fc"):
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            self.backbone.classifier[-1] = nn.Identity()

        # Create classifier head
        self.classifier = nn.Linear(feature_dim, num_classes)

        logger.info(f"Created {backbone} model with {num_classes} classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Class logits
        """
        features = self.backbone(x)
        return self.classifier(features)


def create_model(num_classes: int, model_config: Dict[str, Any] = None) -> nn.Module:
    """
    Create a cat breed classifier model.

    Args:
        num_classes: Number of output classes
        model_config: Model configuration dictionary

    Returns:
        Initialized model
    """
    if model_config is None:
        model_config = {}

    backbone = model_config.get("backbone", "shufflenet_v2")
    pretrained = model_config.get("pretrained", True)
    dropout_rate = model_config.get("dropout_rate", 0.5)

    return CatBreedClassifier(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        dropout_rate=dropout_rate,
    )


def save_model(
    model: nn.Module,
    save_path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    loss: float = 0.0,
    accuracy: float = 0.0,
    metadata: Optional[Dict[str, Any]] = None,
    save_as_state_dict: bool = False,
) -> None:
    """
    Save model checkpoint.

    Args:
        model: Model to save
        save_path: Path to save the checkpoint
        optimizer: Optimizer state to save
        epoch: Current epoch
        loss: Current validation loss
        accuracy: Current validation accuracy
        metadata: Additional metadata to save
    """
    if save_as_state_dict:
        save_dict = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": loss,
            "val_accuracy": accuracy,
        }

        if optimizer is not None:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()

        if metadata is not None:
            save_dict["metadata"] = metadata

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the model
        torch.save(save_dict, save_path)
        logger.info(f"Model saved to {save_path}")
    else:
        # Save the entire model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model, save_path)
        logger.info(f"Model saved to {save_path} as a complete model")


def load_model(
    path: Union[str, Path],
    num_classes: int,
    model_config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load model from checkpoint.

    Args:
        path: Path to the checkpoint
        num_classes: Number of output classes
        model_config: Model configuration

    Returns:
        Tuple of (model, checkpoint_data)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint
    if model_config is None:
        model = torch.load(path, map_location=torch.device(device), weights_only=False)
    else:
        checkpoint = torch.load(path, map_location=torch.device(device))
        # Create model
        model = create_model(num_classes=num_classes, model_config=model_config)
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Loaded model from {path}")

    return model


if __name__ == "__main__":
    # Example usage
    model = create_model(
        num_classes=10, model_config={"backbone": "mobilenet_v2", "pretrained": True}
    )

    # Print model summary
    summary(model, input_size=(3, 224, 224), device=device)

    # Save and load example
    save_model(model, "./checkpoints/model.pth")
    loaded_model = load_model("./checkpoints/model.pth", num_classes=10)
