"""Test module for cat breed classification."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import load_model

logger = logging.getLogger(__name__)


class CatBreedPredictor:
    """Predictor class for cat breed classification."""

    def __init__(
        self,
        model_path: Union[str, Path],
        class_names: List[str],
        device: str = "cuda",
        img_size: int = 224,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to the model checkpoint
            class_names: List of class names
            device: Device to use for inference
            img_size: Input image size
        """
        self.class_names = class_names
        self.device = device if torch.cuda.is_available() else "cpu"
        self.img_size = img_size

        # Load model
        self.model, _ = load_model(model_path, num_classes=len(class_names))
        self.model.to(self.device)
        self.model.eval()

        # Set up transforms
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        logger.info(f"Predictor initialized using device: {self.device}")

    def predict_image(
        self, image: Union[str, Path, np.ndarray, Image.Image], top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Predict breed for an image.

        Args:
            image: Input image (path, array, or PIL Image)
            top_k: Return top-k predictions

        Returns:
            Dictionary containing prediction results
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Transform image
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Record inference time
        start_time = time.time()

        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Get top-k predictions
        top_probs, top_indices = torch.topk(
            probabilities, k=min(top_k, len(self.class_names))
        )
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # Create prediction results
        predictions = [
            {"class": self.class_names[idx], "score": float(prob)}
            for idx, prob in zip(top_indices, top_probs)
        ]

        results = {
            "predictions": predictions,
            "top_prediction": predictions[0]["class"],
            "top_score": predictions[0]["score"],
            "inference_time_ms": float(inference_time),
        }

        return results

    def predict_and_visualize(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 5,
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        """
        Predict and visualize results.

        Args:
            image: Input image (path, array, or PIL Image)
            top_k: Return top-k predictions
            figsize: Figure size

        Returns:
            Matplotlib figure with visualization
        """
        # Load image for display
        if isinstance(image, (str, Path)):
            display_img = cv2.imread(str(image))
            display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(display_img)
        elif isinstance(image, Image.Image):
            pil_img = image
            display_img = np.array(pil_img)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Make prediction
        results = self.predict_image(pil_img, top_k=top_k)
        predictions = results["predictions"]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot image
        ax1.imshow(display_img)
        ax1.set_title(f"Prediction: {results['top_prediction']}")
        ax1.axis("off")

        # Plot prediction bars
        classes = [p["class"] for p in predictions]
        scores = [p["score"] for p in predictions]

        y_pos = np.arange(len(classes))
        ax2.barh(y_pos, scores, align="center")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(classes)
        ax2.set_xlabel("Probability")
        ax2.set_xlim(0, 1)
        ax2.set_title(f"Top-{len(predictions)} Predictions")

        # Add inference time text
        plt.figtext(
            0.5,
            0.01,
            f"Inference time: {results['inference_time_ms']:.2f} ms",
            ha="center",
        )

        plt.tight_layout()

        return fig

    def process_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        display: bool = False,
        fps: Optional[int] = None,
        show_fps: bool = True,
    ) -> Optional[str]:
        """
        Process a video file.

        Args:
            video_path: Path to the video file
            output_path: Path to save the output video
            display: Whether to display the processed video
            fps: Output video FPS (uses input video FPS if None)
            show_fps: Whether to show FPS on the video

        Returns:
            Path to the output video if output_path is provided
        """
        if output_path is None and not display:
            raise ValueError("Either output_path or display must be specified")

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = fps or int(cap.get(cv2.CAP_PROP_FPS))

        # Create video writer if output_path is provided
        if output_path:
            output_path = str(output_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process video
        frame_count = 0
        processing_times = []

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            start_time = time.time()

            # Make prediction
            results = self.predict_image(frame, top_k=3)
            predictions = results["predictions"]

            # Draw prediction on frame
            prediction_text = (
                f"{predictions[0]['class']}: {predictions[0]['score']:.2f}"
            )
            cv2.putText(
                frame,
                prediction_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Draw top-3 predictions
            for i, pred in enumerate(predictions[:3]):
                text = f"{pred['class']}: {pred['score']:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (10, 70 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Calculate and show FPS
            proc_time = time.time() - start_time
            processing_times.append(proc_time)

            if show_fps and len(processing_times) > 0:
                avg_time = sum(processing_times[-30:]) / min(len(processing_times), 30)
                fps_text = f"FPS: {1.0 / avg_time:.1f}"
                cv2.putText(
                    frame,
                    fps_text,
                    (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # Write frame to output video
            if output_path:
                writer.write(frame)

            # Display frame
            if display:
                cv2.imshow("Video", frame)

                # Exit on 'q' press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        # Clean up
        cap.release()
        if output_path:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        # Calculate stats
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            logger.info(f"Processed {frame_count} frames")
            logger.info(f"Average processing time: {avg_time * 1000:.2f} ms/frame")
            logger.info(f"Average FPS: {1.0 / avg_time:.2f}")

        return output_path if output_path else None

    def run_webcam(self, camera_id: int = 0) -> None:
        """
        Run prediction on webcam feed.

        Args:
            camera_id: Camera ID (usually 0 for built-in webcam)
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        processing_times = []
        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1
            start_time = time.time()

            # Make prediction
            results = self.predict_image(frame, top_k=3)
            predictions = results["predictions"]

            # Draw prediction on frame
            prediction_text = (
                f"{predictions[0]['class']}: {predictions[0]['score']:.2f}"
            )
            cv2.putText(
                frame,
                prediction_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Draw top-3 predictions
            for i, pred in enumerate(predictions[:3]):
                text = f"{pred['class']}: {pred['score']:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (10, 70 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Calculate and show FPS
            proc_time = time.time() - start_time
            processing_times.append(proc_time)

            if len(processing_times) > 0:
                avg_time = sum(processing_times[-30:]) / min(len(processing_times), 30)
                fps_text = f"FPS: {1.0 / avg_time:.1f}"
                cv2.putText(
                    frame,
                    fps_text,
                    (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            # Display frame
            cv2.imshow("Webcam", frame)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

        # Calculate stats
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            logger.info(f"Processed {frame_count} frames")
            logger.info(f"Average processing time: {avg_time * 1000:.2f} ms/frame")
            logger.info(f"Average FPS: {1.0 / avg_time:.2f}")
