"""
Pipeline for cat breed and emotion detection with animal tracking.

UNIFIED PIPELINE FLOW (Applied to ALL input types):
======================================================

1. INPUT PROCESSING
   - Image: Single image file
   - Video: Frame-by-frame processing
   - Webcam: Real-time frame processing

2. ANIMAL DETECTION/TRACKING
   - MMPose (if available): Advanced animal detection with pose estimation
   - Fallback: Whole image assumed to contain cat

3. BOUNDING BOX EXTRACTION
   - Extract detected animal bounding boxes
   - Filter for cats/animals above confidence threshold
   - Validate bbox coordinates within image bounds

4. FRAME CROPPING
   - Crop image regions based on detected bboxes
   - Each detected animal gets its own cropped frame

5. BREED CLASSIFICATION
   - Run breed classification model on each cropped frame
   - Output: breed label + confidence score

6. EMOTION CLASSIFICATION
   - Run emotion classification model on each cropped frame
   - Output: emotion label + confidence score

7. RESULT VISUALIZATION
   - Draw bounding boxes around detected animals
   - Add refined labels (inside/above/below bbox as appropriate)
   - Semi-transparent backgrounds for optimal readability

This ensures consistent processing regardless of input type.
All images/frames follow the same 7-step pipeline flow.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from test import CatBreedPredictor
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# Import local modules
from model import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MMPose imports (following the official demo pattern)
try:
    from mmdet.apis import inference_detector, init_detector

    MMDET_AVAILABLE = True
except ImportError:
    logger.warning("MMDet not available")
    MMDET_AVAILABLE = False

try:
    from mmpose.apis import inference_topdown
    from mmpose.apis import init_model as init_pose_estimator
    from mmpose.evaluation.functional import nms
    from mmpose.utils import adapt_mmdet_pipeline

    MMPOSE_AVAILABLE = True
except ImportError:
    logger.warning("MMPose not available.")
    MMPOSE_AVAILABLE = False


class CatAnalysisPipeline:
    """
    Pipeline for detecting, tracking and analyzing cats in images/videos.

    Pipeline Flow:
    1. Input Processing (Image/Video/Webcam)
    2. Animal Detection/Tracking (MMPose or fallback to full image)
    3. Bounding Box Extraction (from detection results)
    4. Frame Cropping (based on detected bboxes)
    5. Breed Classification (on cropped cat frames)
    6. Emotion Classification (on cropped cat frames)
    7. Result Visualization (with refined label positioning)

    This ensures consistent processing across all input types.
    """

    def __init__(
        self,
        breed_model_path: str,
        emotion_model_path: str,
        breed_classes: List[str],
        emotion_classes: List[str],
        device: str = "cuda",
        img_size: int = 224,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the pipeline.

        Args:
            breed_model_path: Path to trained breed classification model
            emotion_model_path: Path to trained emotion classification model
            breed_classes: List of breed class names
            emotion_classes: List of emotion class names
            device: Device to use for inference
            img_size: Input image size for models
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.img_size = img_size
        self.confidence_threshold = confidence_threshold

        # Load breed classification model
        logger.info(f"Loading breed model from {breed_model_path}")
        self.breed_model = load_model(
            path=breed_model_path,
            num_classes=len(breed_classes),
            model_type="breed",
            model_config={"backbone": "mobilenet_v2", "pretrained": False},
        )
        self.breed_model.eval()
        self.breed_model.to(device)
        self.breed_classes = breed_classes

        # Load emotion classification model
        logger.info(f"Loading emotion model from {emotion_model_path}")
        self.emotion_model = load_model(
            path=emotion_model_path,
            num_classes=len(emotion_classes),
            model_type="emotion",
            model_config={"backbone": "mobilenet_v2", "pretrained": False},
        )
        self.emotion_model.eval()
        self.emotion_model.to(device)
        self.emotion_classes = emotion_classes

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Initialize animal detection model if both MMDet and MMPose are available
        if MMPOSE_AVAILABLE and MMDET_AVAILABLE:
            self._init_detection_models()
        else:
            self.detector = None
            self.pose_estimator = None

    def _init_detection_models(self):
        """Initialize MMDet and MMPose detection models following official demo pattern."""
        try:
            if not (MMDET_AVAILABLE and MMPOSE_AVAILABLE):
                logger.warning("Both MMDet and MMPose are required for animal detection")
                self.detector = None
                self.pose_estimator = None
                return

            # Use COCO-trained detector for animal detection (class 15 = cat)
            det_config = "src/dependencies/mmpose/demo/mmdetection_cfg/rtmdet_m_8xb32-300e_coco.py"
            det_checkpoint = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"

            # Use animal pose estimator
            pose_config = "src/dependencies/mmpose/configs/animal_2d_keypoint/topdown_heatmap/animalpose/td-hm_hrnet-w32_8xb64-210e_animalpose-256x256.py"
            pose_checkpoint = "https://download.openmmlab.com/mmpose/animal/hrnet/hrnet_w32_animalpose_256x256-1aa7f075_20210426.pth"

            # Initialize detector following the official demo pattern
            self.detector = init_detector(det_config, det_checkpoint, device=self.device)
            self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

            # Initialize pose estimator
            self.pose_estimator = init_pose_estimator(
                pose_config, pose_checkpoint, device=self.device
            )

            # Animal detection parameters
            self.det_cat_id = 15  # COCO cat class
            self.bbox_thr = self.confidence_threshold
            self.nms_thr = 0.3

            logger.info("Animal detection and pose estimation models initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize detection models: {e}")
            self.detector = None
            self.pose_estimator = None

    def detect_animals(self, image: np.ndarray) -> List[Dict]:
        """
        Step 2: Detect/Track animals in the image using MMPose or fallback.

        This is the core detection step that runs for ALL input types:
        - For images: Single detection per image
        - For videos: Detection per frame
        - For webcam: Real-time detection per frame

        Args:
            image: Input image as numpy array

        Returns:
            List of detection dictionaries with bounding boxes and scores
            Format: [{"bbox": [x1, y1, x2, y2], "score": float, "label": str}]
        """
        if not MMDET_AVAILABLE or self.detector is None:
            # Fallback: assume the whole image contains a cat
            h, w = image.shape[:2]
            return [{"bbox": [0, 0, w, h], "score": 1.0, "label": "cat"}]

        try:
            # Run detection following the official MMPose demo pattern
            det_result = inference_detector(self.detector, image)
            pred_instance = det_result.pred_instances.cpu().numpy()

            # Extract bboxes with scores
            bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)

            # Filter for cats (class 15 in COCO) above threshold
            bboxes = bboxes[
                np.logical_and(
                    pred_instance.labels == self.det_cat_id, pred_instance.scores > self.bbox_thr
                )
            ]

            # Apply NMS to remove overlapping detections
            if len(bboxes) > 0:
                bboxes = bboxes[nms(bboxes, self.nms_thr)]

            # Convert to our format
            detections = []
            for bbox in bboxes:
                if len(bbox) >= 5:  # x1, y1, x2, y2, score
                    detections.append(
                        {"bbox": bbox[:4].tolist(), "score": float(bbox[4]), "label": "cat"}
                    )

            # If using pose estimation, run pose detection for additional context
            if MMPOSE_AVAILABLE and self.pose_estimator is not None and len(detections) > 0:
                try:
                    pose_bboxes = np.array([det["bbox"] for det in detections])
                    pose_results = inference_topdown(self.pose_estimator, image, pose_bboxes)
                    # Store pose results for potential future use
                    for i, detection in enumerate(detections):
                        if i < len(pose_results):
                            detection["pose_keypoints"] = pose_results[i]
                except Exception as e:
                    logger.debug(f"Pose estimation failed: {e}")

            return detections

        except Exception as e:
            logger.warning(f"Animal detection failed: {e}")
            # Fallback: assume the whole image contains a cat
            h, w = image.shape[:2]
            return [{"bbox": [0, 0, w, h], "score": 1.0, "label": "cat"}]

    def classify_breed_and_emotion(self, image_crop: np.ndarray) -> Tuple[str, float, str, float]:
        """
        Steps 5-6: Classify breed and emotion from a cropped cat image.

        This runs on EVERY detected bounding box from the detection step.
        The same classification logic applies to all input types.

        Args:
            image_crop: Cropped cat image as numpy array (from detected bbox)

        Returns:
            Tuple of (breed_label, breed_confidence, emotion_label, emotion_confidence)
        """
        # Convert to PIL and apply transforms
        pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Breed classification
            breed_outputs = self.breed_model(input_tensor)
            breed_probs = F.softmax(breed_outputs, dim=1)
            breed_confidence, breed_idx = torch.max(breed_probs, 1)
            breed_label = self.breed_classes[breed_idx.item()]
            breed_conf = breed_confidence.item()

            # Emotion classification
            emotion_outputs = self.emotion_model(input_tensor)
            emotion_probs = F.softmax(emotion_outputs, dim=1)
            emotion_confidence, emotion_idx = torch.max(emotion_probs, 1)
            emotion_label = self.emotion_classes[emotion_idx.item()]
            emotion_conf = emotion_confidence.item()

        return breed_label, breed_conf, emotion_label, emotion_conf

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Complete pipeline processing for a single image/frame.

        This method implements the full pipeline flow:
        1. Input: Single image/frame
        2. Detection: Animal detection/tracking
        3. Bbox Extraction: Get bounding boxes from detections
        4. Cropping: Extract cat regions based on bboxes
        5. Breed Classification: Classify breed for each detected cat
        6. Emotion Classification: Classify emotion for each detected cat
        7. Visualization: Draw results with refined label positioning

        Used by: image processing, video frame processing, webcam frame processing

        Args:
            image: Input image as numpy array

        Returns:
            Tuple of (annotated_image, detection_results)
        """
        # Step 2: Detect animals using MMPose or fallback
        detections = self.detect_animals(image)

        results = []
        annotated_image = image.copy()

        # Step 3: Process each detected bounding box
        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = map(int, bbox)

            # Step 3a: Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Step 4: Crop the detected region for classification
            crop = image[y1:y2, x1:x2]

            if crop.size > 0:
                # Steps 5-6: Classify breed and emotion on cropped frame
                breed_label, breed_conf, emotion_label, emotion_conf = (
                    self.classify_breed_and_emotion(crop)
                )

                # Store results for this detection
                result = {
                    "bbox": [x1, y1, x2, y2],
                    "detection_score": detection["score"],
                    "breed": breed_label,
                    "breed_confidence": breed_conf,
                    "emotion": emotion_label,
                    "emotion_confidence": emotion_conf,
                }
                results.append(result)

                # Step 7: Visualization - Draw bounding box and refined labels
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Create label text
                breed_text = f"{breed_label} ({breed_conf:.2f})"
                emotion_text = f"{emotion_label} ({emotion_conf:.2f})"

                # Calculate text sizes for proper positioning
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                breed_size = cv2.getTextSize(breed_text, font, font_scale, thickness)[0]
                emotion_size = cv2.getTextSize(emotion_text, font, font_scale, thickness)[0]

                # Calculate label background dimensions
                max_text_width = max(breed_size[0], emotion_size[0])
                line_height = max(breed_size[1], emotion_size[1])
                padding = 8
                label_height = (2 * line_height) + (3 * padding)  # Two lines + padding
                label_width = max_text_width + padding

                # Get image and bounding box dimensions
                img_height, img_width = annotated_image.shape[:2]
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                # Determine optimal label position
                if bbox_width >= label_width + 10 and bbox_height >= label_height + 10:
                    # Labels fit inside the bounding box - place at top-left
                    label_x = x1 + 5
                    label_y = y1 + 5
                    bg_color = (0, 0, 0)  # Black background for inside box
                    text_color = (255, 255, 255)  # White text for inside box
                    use_transparency = True

                    # Draw semi-transparent background
                    overlay = annotated_image.copy()
                    cv2.rectangle(
                        overlay,
                        (label_x, label_y),
                        (label_x + label_width, label_y + label_height),
                        bg_color,
                        -1,
                    )
                    cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)

                elif y1 >= label_height + 10:
                    # Place labels above the bounding box
                    label_x = x1
                    label_y = y1 - label_height - 5
                    bg_color = (0, 255, 0)  # Green background for outside box
                    text_color = (0, 0, 0)  # Black text for outside box
                    use_transparency = False

                    # Ensure label doesn't go outside image bounds
                    if label_x + label_width > img_width:
                        label_x = img_width - label_width
                    if label_x < 0:
                        label_x = 0

                    # Draw solid background
                    cv2.rectangle(
                        annotated_image,
                        (label_x, label_y),
                        (label_x + label_width, label_y + label_height),
                        bg_color,
                        -1,
                    )

                else:
                    # Place labels below the bounding box if not enough space above
                    label_x = x1
                    label_y = y2 + 5
                    bg_color = (0, 255, 0)  # Green background for outside box
                    text_color = (0, 0, 0)  # Black text for outside box
                    use_transparency = False

                    # Ensure label doesn't go outside image bounds
                    if label_x + label_width > img_width:
                        label_x = img_width - label_width
                    if label_x < 0:
                        label_x = 0
                    if label_y + label_height > img_height:
                        label_y = img_height - label_height

                    # Draw solid background
                    cv2.rectangle(
                        annotated_image,
                        (label_x, label_y),
                        (label_x + label_width, label_y + label_height),
                        bg_color,
                        -1,
                    )

                # Draw text labels
                text_x = label_x + padding // 2
                breed_y = label_y + line_height + padding
                emotion_y = breed_y + line_height + padding

                cv2.putText(
                    annotated_image,
                    breed_text,
                    (text_x, breed_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )
                cv2.putText(
                    annotated_image,
                    emotion_text,
                    (text_x, emotion_y),
                    font,
                    font_scale,
                    text_color,
                    thickness,
                )

        return annotated_image, results

    def process_video(
        self, video_path: str, output_path: Optional[str] = None, display: bool = True
    ) -> Optional[str]:
        """
        Process a video file using the same pipeline flow as single images.

        Pipeline Flow (applied to each frame):
        1. Input: Video frame
        2-7. Same as process_image() for each frame

        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display the video while processing

        Returns:
            Path to output video if saved, None otherwise
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return None

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set up video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logger.info(f"Processing video: {total_frames} frames at {fps} FPS")

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame using the same 7-step pipeline flow as single images
                annotated_frame, results = self.process_image(frame)

                # Write frame if output is specified
                if writer:
                    writer.write(annotated_frame)

                # Display frame if requested
                if display:
                    cv2.imshow("Cat Analysis Pipeline", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if frame_count % 30 == 0:  # Log progress every 30 frames
                    elapsed = time.time() - start_time
                    fps_current = frame_count / elapsed
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% | FPS: {fps_current:.1f}")

        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        logger.info(f"Video processing completed in {elapsed:.2f}s")

        return output_path

    def process_webcam(self, camera_id: int = 0):
        """
        Process webcam feed in real-time using the same pipeline flow.

        Pipeline Flow (applied to each frame):
        1. Input: Webcam frame
        2-7. Same as process_image() for each frame in real-time

        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return

        logger.info("Starting webcam processing. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame using the same 7-step pipeline flow (real-time)
                annotated_frame, results = self.process_image(frame)

                # Display frame
                cv2.imshow("Cat Analysis Pipeline - Webcam", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


def load_class_names(class_names_path: str) -> List[str]:
    """Load class names from JSON file."""
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            return json.load(f)
    else:
        logger.warning(f"Class names file not found: {class_names_path}")
        return []


def main():
    """
    Main function for running the unified cat analysis pipeline.

    ALL commands (image, video, webcam) use the same 7-step pipeline flow:
    Detection → Bbox Extraction → Cropping → Breed Classification → Emotion Classification → Visualization

    The only difference between commands is the input source:
    - image: Single image file processing
    - video: Frame-by-frame video processing
    - webcam: Real-time webcam frame processing
    """
    parser = argparse.ArgumentParser(description="Cat Analysis Pipeline")

    # Required arguments
    parser.add_argument("command", choices=["image", "video", "webcam"], help="Processing mode")
    parser.add_argument("--breed-model", required=True, help="Path to breed classification model")
    parser.add_argument(
        "--emotion-model", required=True, help="Path to emotion classification model"
    )
    parser.add_argument("--breed-classes", required=True, help="Path to breed class names JSON")
    parser.add_argument(
        "--emotion-classes", required=True, help="Path to emotion class names JSON"
    )

    # Optional arguments
    parser.add_argument("--input", help="Input image/video path (required for image/video modes)")
    parser.add_argument("--output", help="Output path for processed video")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID for webcam mode")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use"
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--confidence", type=float, default=0.5, help="Detection confidence threshold"
    )

    args = parser.parse_args()

    # Load class names
    breed_classes = load_class_names(args.breed_classes)
    emotion_classes = load_class_names(args.emotion_classes)

    if not breed_classes or not emotion_classes:
        logger.error("Failed to load class names")
        sys.exit(1)

    # Initialize pipeline
    pipeline = CatAnalysisPipeline(
        breed_model_path=args.breed_model,
        emotion_model_path=args.emotion_model,
        breed_classes=breed_classes,
        emotion_classes=emotion_classes,
        device=args.device,
        img_size=args.img_size,
        confidence_threshold=args.confidence,
    )

    # Run processing based on command
    if args.command == "image":
        if not args.input or not os.path.exists(args.input):
            logger.error("Input image path is required and must exist")
            sys.exit(1)

        # Load and process image
        image = cv2.imread(args.input)
        if image is None:
            logger.error(f"Failed to load image: {args.input}")
            sys.exit(1)

        annotated_image, results = pipeline.process_image(image)

        # Save or display result
        if args.output:
            cv2.imwrite(args.output, annotated_image)
            logger.info(f"Result saved to {args.output}")
        else:
            cv2.imshow("Cat Analysis Result", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Print results
        for i, result in enumerate(results):
            logger.info(
                f"Detection {i+1}: Breed={result['breed']} ({result['breed_confidence']:.2f}), "
                f"Emotion={result['emotion']} ({result['emotion_confidence']:.2f})"
            )

    elif args.command == "video":
        if not args.input or not os.path.exists(args.input):
            logger.error("Input video path is required and must exist")
            sys.exit(1)

        output_path = pipeline.process_video(
            video_path=args.input, output_path=args.output, display=True
        )

        if output_path:
            logger.info(f"Processed video saved to {output_path}")

    elif args.command == "webcam":
        pipeline.process_webcam(camera_id=args.camera_id)


if __name__ == "__main__":
    main()
