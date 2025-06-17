"""Training module for cat breed classification."""

import datetime
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.data_utils import print_training_config
from utils.visualization import plot_learning_curves

logger = logging.getLogger(__name__)


class CatBreedTrainer:
    """Trainer class for cat breed classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = None,
        optimizer: torch.optim.Optimizer = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        checkpoint_dir: Union[str, Path] = "checkpoints",
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            lr_scheduler: Learning rate scheduler
            device: Device to use for training
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Use cross-entropy loss by default
        self.criterion = criterion or nn.CrossEntropyLoss()

        # Use AdamW optimizer by default
        self.optimizer = optimizer or optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=1e-4
        )
        self.lr_scheduler = lr_scheduler
        self.device = device if torch.cuda.is_available() else "cpu"

        # Set up checkpoint directory with backbone name and timestamp
        backbone_name = getattr(model, "backbone_name", type(model).__name__)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir = self.base_checkpoint_dir / f"{backbone_name}_{timestamp}"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Store the best model path for easy reference
        self.best_model_path = self.checkpoint_dir / "best_state.pth"
        self.last_model_path = self.checkpoint_dir / "last_state.pth"

        # Move model to device
        self.model.to(self.device)

        # Initialize training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }

        logger.info(
            f"Trainer initialized with {len(train_loader.dataset)} training samples and "
            f"{len(val_loader.dataset)} validation samples, using device: {self.device}"
        )

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {"loss": total_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
            )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.

        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating")

            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # Update progress bar
                pbar.set_postfix(
                    {"loss": total_loss / (pbar.n + 1), "acc": 100.0 * correct / total}
                )

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def fit(
        self,
        epochs: int,
        early_stopping_patience: int = 10,
        save_best_only: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_best_only: Whether to save only the best model

        Returns:
            Training history
        """
        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0  # Create training configuration dictionary
        training_config = {
            "backbone": getattr(self.model, "backbone_name", type(self.model).__name__),
            "pretrained": True,  # Assuming pretrained is used
            "dropout_rate": 0.5,  # Default value, could be extracted from model if needed
            "num_classes": (
                self.model.classifier.out_features
                if hasattr(self.model, "classifier")
                else None
            ),
            "optimizer": self.optimizer.__class__.__name__,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "weight_decay": self.optimizer.param_groups[0].get("weight_decay", 0),
            "scheduler": self.lr_scheduler.__class__.__name__
            if self.lr_scheduler
            else None,
            "epochs": epochs,
            "batch_size": self.train_loader.batch_size,
            "early_stopping": early_stopping_patience,
            "device": self.device,
            "training_start_time": datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "checkpoint_dir": str(self.checkpoint_dir),
        }

        # Save training configuration as YAML
        config_path = self.checkpoint_dir / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(training_config, f, default_flow_style=False)
        logger.info(f"Training configuration saved to {os.path.relpath(config_path)}")

        # Print training configuration
        print_training_config(training_config)

        logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            logger.info(f"Epoch {epoch}/{epochs} - Starting training...")
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            logger.info(
                f"Epoch {epoch}/{epochs} - {epoch_time:.2f}s - "
                f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.2f}% - "
                f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.2f}% - "
                f"lr: {current_lr:.6f}"
            )  # Save checkpoint
            improved = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                improved = True
                patience_counter = 0
                # Save best model
                self.save_checkpoint(
                    self.best_model_path,
                    epoch=epoch,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    is_best=True,
                )
                logger.info(
                    f"New best model saved with validation loss: {val_loss:.4f}"
                )
            else:
                patience_counter += 1

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # Always save the latest model
            self.save_checkpoint(
                self.last_model_path,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
                is_best=False,
            )

            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {early_stopping_patience} epochs)"
                )
                break

        total_time = time.time() - start_time
        logger.info(
            f"Training completed in {total_time:.2f}s - "
            f"Best val_loss: {best_val_loss:.4f} - Best val_acc: {best_val_acc:.2f}%"
        )  # Plot learning curves at the end of training
        fig = plot_learning_curves(self.history)
        fig.savefig(self.checkpoint_dir / "learning_curves.png")

        # Save final class names if available
        if hasattr(self.train_loader.dataset, "class_names"):
            import json

            with open(self.checkpoint_dir / "class_names.json", "w") as f:
                json.dump(self.train_loader.dataset.class_names, f)

        logger.info(
            f"Training artifacts saved to {os.path.relpath(self.checkpoint_dir)}"
        )
        return self.history

    def save_checkpoint(
        self,
        path: Union[str, Path],
        epoch: int,
        val_loss: float,
        val_acc: float,
        is_best: bool = False,
    ) -> None:
        """
        Save a model checkpoint.

        Args:
            path: Path to save the checkpoint
            epoch: Current epoch
            val_loss: Validation loss
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "history": self.history,
        }

        if self.lr_scheduler:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {os.path.relpath(path)}")

    def load_checkpoint(self, path: Union[str, Path] = None) -> int:
        """
        Load a model checkpoint.

        Args:
            path: Path to the checkpoint. If None, will try to load the best model.

        Returns:
            Epoch number of the loaded checkpoint
        """
        if path is None:
            # Look for best model in the checkpoint directory
            if hasattr(self, "best_model_path") and os.path.exists(
                self.best_model_path
            ):
                path = self.best_model_path
            else:
                # Look in the checkpoint directory for any model files
                checkpoint_files = list(self.checkpoint_dir.glob("*.pth"))
                if not checkpoint_files:
                    logger.error(
                        "No checkpoint files found in the checkpoint directory"
                    )
                    return 0
                path = checkpoint_files[0]  # Use first available checkpoint

        # Ensure path is a Path object
        path = Path(path)

        # Check if the file exists
        if not path.exists():
            logger.error(f"Checkpoint file not found: {path}")
            return 0

        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state if it exists
        if "scheduler_state_dict" in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load history if it exists
        if "history" in checkpoint:
            self.history = checkpoint["history"]

        epoch = checkpoint.get("epoch", 0)
        val_loss = checkpoint.get("val_loss", "N/A")
        val_acc = checkpoint.get("val_acc", "N/A")
        logger.info(
            f"Loaded checkpoint from {os.path.relpath(path)} (epoch {epoch}, val_loss: {val_loss}, val_acc: {val_acc})"
        )

        return epoch
