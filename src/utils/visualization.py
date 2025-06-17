"""Visualization utilities for the cat breed classification project."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


def setup_logger(
    log_file: Optional[Union[str, Path]] = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger for the project.

    Args:
        log_file: Path to the log file
        level: Logging level

    Returns:
        Configured logger
    """
    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file is provided
    if log_file is not None:
        log_dir = Path(log_file).parent
        os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    figsize: tuple = (10, 8),
    title: str = "Confusion Matrix",
) -> Figure:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()

    return fig


def plot_learning_curves(
    history: Dict[str, List[float]], figsize: tuple = (12, 4)
) -> Figure:
    """
    Plot learning curves from training history.

    Args:
        history: Dictionary containing training metrics
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot training and validation loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].grid(True)

    # Plot training and validation accuracy
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    return fig


def print_classification_report(
    y_true: List[int], y_pred: List[int], class_names: List[str]
) -> Dict[str, Any]:
    """
    Print and return classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names

    Returns:
        Classification report as dictionary
    """
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    # Print the report as a formatted table
    df_report = pd.DataFrame(report).transpose()
    print(df_report)

    return report


class ProgressBar:
    """Simple progress bar for visualization."""

    def __init__(self, total: int, desc: str = "", bar_length: int = 30):
        """
        Initialize progress bar.

        Args:
            total: Total number of items
            desc: Description of the progress bar
            bar_length: Length of the progress bar
        """
        self.total = total
        self.desc = desc
        self.bar_length = bar_length
        self.current = 0

    def update(self, current: int) -> None:
        """
        Update the progress bar.

        Args:
            current: Current progress
        """
        self.current = current
        progress = min(1.0, current / self.total)

        # Calculate bar
        filled_length = int(self.bar_length * progress)
        bar = "█" * filled_length + "░" * (self.bar_length - filled_length)

        # Print progress bar
        print(
            f"\r{self.desc}: [{bar}] {current}/{self.total} ({progress * 100:.1f}%)",
            end="",
        )

        # Print newline when complete
        if progress >= 1.0:
            print()


def visualize_predictions(
    images: torch.Tensor,
    true_labels: List[int],
    pred_labels: List[int],
    class_names: List[str],
    n_samples: int = 16,
    title: str = "Model Predictions",
) -> Figure:
    """
    Visualize model predictions.

    Args:
        images: Batch of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        n_samples: Number of samples to visualize
        title: Plot title

    Returns:
        Matplotlib figure
    """
    # Limit number of samples
    n_samples = min(n_samples, len(images))
    images = images[:n_samples]
    true_labels = true_labels[:n_samples]
    pred_labels = pred_labels[:n_samples]

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Plot images in a grid
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_samples == 1 else axes

    for i in range(n_samples):
        axes[i].imshow(images[i].permute(1, 2, 0).cpu().numpy())

        # Color based on correctness
        color = "green" if true_labels[i] == pred_labels[i] else "red"

        axes[i].set_title(
            f"True: {class_names[true_labels[i]]}\nPred: {class_names[pred_labels[i]]}",
            color=color,
        )
        axes[i].axis("off")

    # Hide empty subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    return fig
