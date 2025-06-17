"""Evaluation module for cat breed classification."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from utils.visualization import plot_confusion_matrix, print_classification_report

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class for cat breed classification."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: str = "cuda",
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Device to use for evaluation
            output_dir: Directory to save results
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir) if output_dir else None

        # Move model to device
        self.model.to(self.device)
        self.model.eval()

        if self.output_dir:
            self.output_dir.mkdir(exist_ok=True, parents=True)

        logger.info(
            f"Evaluator initialized with {len(test_loader.dataset)} test samples "
            f"using device: {self.device}"
        )

    def evaluate(self, class_names: List[str]) -> Dict[str, Any]:
        """
        Evaluate the model.

        Args:
            class_names: List of class names

        Returns:
            Dictionary containing evaluation metrics
        """
        all_targets = []
        all_predictions = []
        all_probabilities = []
        total_time = 0
        num_samples = len(self.test_loader.dataset)

        logger.info(f"Starting evaluation on {num_samples} samples")

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Measure inference time
                start_time = time.time()
                outputs = self.model(inputs)
                batch_time = time.time() - start_time
                total_time += batch_time

                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = outputs.max(1)

                # Collect results
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # Calculate average inference time
        avg_inference_time = total_time / num_samples

        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, average="weighted"
        )

        # Create confusion matrix
        cm_fig = plot_confusion_matrix(all_targets, all_predictions, class_names)

        # Print classification report
        class_report = print_classification_report(
            all_targets, all_predictions, class_names
        )

        # Top-k accuracy
        top3_accuracy = self._calculate_topk_accuracy(
            all_probabilities, all_targets, k=3
        )
        top5_accuracy = self._calculate_topk_accuracy(
            all_probabilities, all_targets, k=5
        )

        # Per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(
            all_targets, all_predictions, all_probabilities, class_names
        )

        # Create results dictionary
        results = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "top3_accuracy": float(top3_accuracy),
            "top5_accuracy": float(top5_accuracy),
            "avg_inference_time_ms": float(avg_inference_time * 1000),
            "per_class_metrics": per_class_metrics,
            "class_report": class_report,
        }

        # Print summary
        logger.info("Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Top-3 Accuracy: {top3_accuracy:.4f}")
        logger.info(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        logger.info(
            f"Average Inference Time: {avg_inference_time * 1000:.2f} ms/sample"
        )

        # Save results if output directory is provided
        if self.output_dir:
            results_path = self.output_dir / "evaluation_results.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)

            # Save confusion matrix plot
            cm_path = self.output_dir / "confusion_matrix.png"
            cm_fig.savefig(cm_path)

            logger.info(f"Evaluation results saved to {self.output_dir}")

        return results

    def _calculate_topk_accuracy(
        self, probabilities: np.ndarray, targets: np.ndarray, k: int = 5
    ) -> float:
        """
        Calculate top-k accuracy.

        Args:
            probabilities: Prediction probabilities
            targets: True labels
            k: k value for top-k accuracy

        Returns:
            Top-k accuracy
        """
        batch_size = targets.shape[0]
        top_k_predictions = np.argsort(-probabilities, axis=1)[:, :k]

        # Check if target is in top-k predictions for each sample
        correct = 0
        for i in range(batch_size):
            if targets[i] in top_k_predictions[i]:
                correct += 1

        return correct / batch_size

    def _calculate_per_class_metrics(
        self,
        targets: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.

        Args:
            targets: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            class_names: List of class names

        Returns:
            Dictionary of per-class metrics
        """
        # Calculate precision, recall, f1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions
        )

        # Create per-class metrics dictionary
        per_class_metrics = {}
        for i, class_name in enumerate(class_names):
            per_class_metrics[class_name] = {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }

        return per_class_metrics
