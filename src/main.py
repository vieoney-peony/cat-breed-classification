"""Main entry point for cat breed classification."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from test import CatBreedPredictor

import torch
import torch.optim as optim
import yaml

from evaluate import Evaluator
from model import create_model, load_model
from trainer import CatBreedTrainer
from utils.data_utils import get_data_loaders
from utils.visualization import setup_logger

# Set up paths
ROOT_DIR = Path(__file__).parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR.parent / "data" / "processed"))
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Create logger
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Cat Breed Classification")

    # Main command
    parser.add_argument(
        "command",
        choices=["train", "test", "evaluate", "predict", "video", "webcam"],
        help="Command to execute",
    )  # Common arguments
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to a YAML configuration file (e.g., training_config.yaml). Command line arguments will override these settings.",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR), help="Path to data directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(CHECKPOINT_DIR),
        help="Directory to save/load checkpoints",
    )
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv2",
        help="Model backbone architecture",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument("--log-file", type=str, help="Log file path")

    # Training arguments
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )

    # Testing/evaluation arguments
    parser.add_argument(
        "--output-dir", type=str, help="Output directory for evaluation results"
    )

    # Prediction arguments
    parser.add_argument("--input", type=str, help="Input image or video path")
    parser.add_argument("--output", type=str, help="Output path for processed video")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera ID for webcam")

    return parser.parse_args()


def load_and_update_config(args):
    """
    Load configuration from YAML file if specified and update args.
    Command-line arguments take precedence over YAML configuration.

    Args:
        args: Command-line arguments

    Returns:
        Updated args
    """
    if not args.config_path:
        return args

    config_path = Path(args.config_path)
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return args

    try:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if not config:
            logger.warning(f"Empty or invalid configuration file: {config_path}")
            return args

        # Create a dictionary from args
        args_dict = vars(args)

        # Track which arguments were explicitly set on the command line
        default_parser = argparse.ArgumentParser()
        default_parser.add_argument(
            "command",
            choices=["train", "test", "evaluate", "predict", "video", "webcam"],
        )
        default_parser.add_argument("--config-path", type=str)
        default_parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
        default_parser.add_argument(
            "--checkpoint-dir", type=str, default=str(CHECKPOINT_DIR)
        )
        default_parser.add_argument("--model-path", type=str)
        default_parser.add_argument("--backbone", type=str, default="mobilenetv2")
        default_parser.add_argument("--batch-size", type=int, default=32)
        default_parser.add_argument("--num-workers", type=int, default=4)
        default_parser.add_argument("--img-size", type=int, default=224)
        default_parser.add_argument(
            "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
        )
        default_parser.add_argument("--log-file", type=str)
        default_parser.add_argument("--epochs", type=int, default=30)
        default_parser.add_argument("--lr", type=float, default=0.001)
        default_parser.add_argument("--weight-decay", type=float, default=1e-4)
        default_parser.add_argument("--patience", type=int, default=10)
        default_parser.add_argument("--output-dir", type=str)
        default_parser.add_argument("--input", type=str)
        default_parser.add_argument("--output", type=str)
        default_parser.add_argument("--camera-id", type=int, default=0)

        default_args = default_parser.parse_args([args.command])
        default_arg_values = vars(default_args)

        # Map config keys to argument names
        config_to_arg_map = {
            "backbone": "backbone",
            "batch_size": "batch_size",
            "learning_rate": "lr",
            "weight_decay": "weight_decay",
            "epochs": "epochs",
            "early_stopping": "patience",
            "device": "device",
            "img_size": "img_size",
            "num_workers": "num_workers",
            "dropout_rate": None,  # Ignore - not a CLI argument
            "pretrained": None,  # Ignore - not a CLI argument
            "num_classes": None,  # Ignore - not a CLI argument
            "optimizer": None,  # Ignore - not a CLI argument
            "scheduler": None,  # Ignore - not a CLI argument
        }

        # Update args with values from config, but only if not explicitly set in command line
        for config_key, arg_name in config_to_arg_map.items():
            if arg_name is None:
                continue

            if config_key in config and args_dict.get(
                arg_name
            ) == default_arg_values.get(arg_name):
                logger.debug(
                    f"Setting {arg_name} = {config[config_key]} from config file"
                )
                args_dict[arg_name] = config[config_key]

        # Print the configuration used
        logger.info("Using configuration:")
        for key, value in args_dict.items():
            if key != "command":
                logger.info(f"  {key}: {value}")

        return args
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return args


def train(args):
    """Train a model."""
    logger.info(f"Starting training with {args.backbone} backbone")

    # Load data
    data_loaders = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    train_loader = data_loaders["train"]
    val_loader = data_loaders["val"]
    class_names = data_loaders["class_names"]

    # Create model
    model_config = {"backbone": args.backbone, "pretrained": True, "dropout_rate": 0.5}

    model = create_model(num_classes=len(class_names), model_config=model_config)

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5, min_lr=1e-6
    )

    # Create trainer
    trainer = CatBreedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    # Train model
    history = trainer.fit(
        epochs=args.epochs, early_stopping_patience=args.patience, save_best_only=True
    )  # No need to save class names and model separately as it's handled by the trainer
    logger.info(f"Training completed, models saved to {trainer.checkpoint_dir}")


def evaluate(args):
    """Evaluate a model."""
    # If model_path is provided, use it directly
    if args.model_path and os.path.exists(args.model_path):
        model_path = args.model_path
        model_dir = Path(model_path).parent
    else:
        # Otherwise, find the latest model checkpoint directory
        checkpoint_dir = Path(args.checkpoint_dir)
        model_dirs = sorted(
            [d for d in checkpoint_dir.glob("*_*") if d.is_dir()],
            key=os.path.getmtime,
            reverse=True,
        )

        if not model_dirs:
            logger.error(f"No model checkpoint directories found in {checkpoint_dir}")
            sys.exit(1)

        # Use the latest directory
        model_dir = model_dirs[0]
        model_path = model_dir / "best_state.pth"

        if not model_path.exists():
            model_path = model_dir / "last_state.pth"

        if not model_path.exists():
            logger.error(f"No model checkpoint files found in {model_dir}")
            sys.exit(1)

    logger.info(f"Using model checkpoint: {model_path}")

    # Load class names
    class_names_path = model_dir / "class_names.json"
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        # If class names not found, try to determine from data directory
        class_names = [
            d.name
            for d in Path(args.data_dir).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        class_names.sort()

    # Load data
    data_loaders = get_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    test_loader = data_loaders["test"]

    # Load model
    model, checkpoint = load_model(path=model_path, num_classes=len(class_names))

    # Create evaluator
    output_dir = args.output_dir or Path(args.checkpoint_dir) / "evaluation"
    os.makedirs(output_dir, exist_ok=True)

    evaluator = Evaluator(
        model=model, test_loader=test_loader, device=args.device, output_dir=output_dir
    )

    # Evaluate model
    results = evaluator.evaluate(class_names=class_names)

    # Print key metrics
    logger.info("Evaluation Summary:")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"F1 Score: {results['f1_score']:.4f}")
    logger.info(f"Top-3 Accuracy: {results['top3_accuracy']:.4f}")
    logger.info(
        f"Average Inference Time: {results['avg_inference_time_ms']:.2f} ms/sample"
    )


def predict(args):
    """Run prediction on a single image."""
    if not args.input:
        logger.error("Input image path is required")
        sys.exit(1)

    image_path = args.input
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)

    # If model_path is provided, use it directly
    if args.model_path and os.path.exists(args.model_path):
        model_path = args.model_path
        model_dir = Path(model_path).parent
    else:
        # Otherwise, find the latest model checkpoint directory
        checkpoint_dir = Path(args.checkpoint_dir)
        model_dirs = sorted(
            [d for d in checkpoint_dir.glob("*_*") if d.is_dir()],
            key=os.path.getmtime,
            reverse=True,
        )

        if not model_dirs:
            logger.error(f"No model checkpoint directories found in {checkpoint_dir}")
            sys.exit(1)

        # Use the latest directory
        model_dir = model_dirs[0]
        model_path = model_dir / "best_state.pth"

        if not model_path.exists():
            model_path = model_dir / "last_state.pth"

        if not model_path.exists():
            logger.error(f"No model checkpoint files found in {model_dir}")
            sys.exit(1)

    logger.info(f"Using model checkpoint: {model_path}")

    # Load class names
    class_names_path = model_dir / "class_names.json"
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        # If class names not found, try to determine from data directory
        class_names = [
            d.name
            for d in Path(args.data_dir).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        class_names.sort()

    # Create predictor
    predictor = CatBreedPredictor(
        model_path=model_path,
        class_names=class_names,
        device=args.device,
        img_size=args.img_size,
    )

    # Make prediction
    fig = predictor.predict_and_visualize(image_path)

    # Save or show result
    if args.output:
        fig.savefig(args.output)
        logger.info(f"Prediction visualization saved to {args.output}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


def process_video(args):
    """Process a video file."""
    if not args.input:
        logger.error("Input video path is required")
        sys.exit(1)

    video_path = args.input
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)

    # If model_path is provided, use it directly
    if args.model_path and os.path.exists(args.model_path):
        model_path = args.model_path
        model_dir = Path(model_path).parent
    else:
        # Otherwise, find the latest model checkpoint directory
        checkpoint_dir = Path(args.checkpoint_dir)
        model_dirs = sorted(
            [d for d in checkpoint_dir.glob("*_*") if d.is_dir()],
            key=os.path.getmtime,
            reverse=True,
        )

        if not model_dirs:
            logger.error(f"No model checkpoint directories found in {checkpoint_dir}")
            sys.exit(1)

        # Use the latest directory
        model_dir = model_dirs[0]
        model_path = model_dir / "best_state.pth"

        if not model_path.exists():
            model_path = model_dir / "last_state.pth"

        if not model_path.exists():
            logger.error(f"No model checkpoint files found in {model_dir}")
            sys.exit(1)

    logger.info(f"Using model checkpoint: {model_path}")

    # Load class names
    class_names_path = model_dir / "class_names.json"
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        # If class names not found, try to determine from data directory
        class_names = [
            d.name
            for d in Path(args.data_dir).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        class_names.sort()

    # Create predictor
    predictor = CatBreedPredictor(
        model_path=model_path,
        class_names=class_names,
        device=args.device,
        img_size=args.img_size,
    )

    # Process video
    output_path = args.output
    display = output_path is None

    output = predictor.process_video(
        video_path=video_path, output_path=output_path, display=display
    )

    if output:
        logger.info(f"Processed video saved to {output}")


def run_webcam(args):
    """Run prediction on webcam feed."""
    # If model_path is provided, use it directly
    if args.model_path and os.path.exists(args.model_path):
        model_path = args.model_path
        model_dir = Path(model_path).parent
    else:
        # Otherwise, find the latest model checkpoint directory
        checkpoint_dir = Path(args.checkpoint_dir)
        model_dirs = sorted(
            [d for d in checkpoint_dir.glob("*_*") if d.is_dir()],
            key=os.path.getmtime,
            reverse=True,
        )

        if not model_dirs:
            logger.error(f"No model checkpoint directories found in {checkpoint_dir}")
            sys.exit(1)

        # Use the latest directory
        model_dir = model_dirs[0]
        model_path = model_dir / "best_state.pth"

        if not model_path.exists():
            model_path = model_dir / "last_state.pth"

        if not model_path.exists():
            logger.error(f"No model checkpoint files found in {model_dir}")
            sys.exit(1)

    logger.info(f"Using model checkpoint: {model_path}")

    # Load class names
    class_names_path = model_dir / "class_names.json"
    if os.path.exists(class_names_path):
        with open(class_names_path, "r") as f:
            class_names = json.load(f)
    else:
        # If class names not found, try to determine from data directory
        class_names = [
            d.name
            for d in Path(args.data_dir).iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        class_names.sort()

    # Create predictor
    predictor = CatBreedPredictor(
        model_path=model_path,
        class_names=class_names,
        device=args.device,
        img_size=args.img_size,
    )

    # Run webcam
    try:
        predictor.run_webcam(camera_id=args.camera_id)
    except KeyboardInterrupt:
        logger.info("Webcam feed stopped by user")


def main():
    """Main entry point."""
    args = parse_args()
    args = load_and_update_config(args)

    # Set up logging
    setup_logger(args.log_file)

    # Print info
    logger.info(f"Running command: {args.command}")
    logger.info(f"Device: {args.device}")

    # Run command
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "video":
        process_video(args)
    elif args.command == "webcam":
        run_webcam(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
