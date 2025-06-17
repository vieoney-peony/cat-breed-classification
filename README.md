# Cat Breed Classification

A modular, concise PyTorch image classification pipeline for cat breeds, supporting configurable backbones and external configuration files.

## Features

-   Support for various backbone architectures (MobileNetV2, ShuffleNetV2, EfficientNet, etc.)
-   Pre-trained ImageNet weights for fast convergence
-   Data augmentation for improved generalization
-   Training with validation-based early stopping and learning rate scheduling
-   Comprehensive evaluation metrics and visualizations
-   Real-time inference on images, videos, or webcam feed
-   Configuration via YAML files with CLI overrides
-   Robust checkpointing and experiment tracking

## Project Structure

```
├── config/                 # Configuration files
├── data/                   # Dataset directory
│   └── processed/          # Processed dataset (class folders)
├── docs/                   # Documentation
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
│   ├── checkpoints/        # Model checkpoints
│   ├── utils/              # Utility functions
│   │   ├── data_utils.py   # Data loading and processing utilities
│   │   └── visualization.py # Visualization utilities
│   ├── model.py            # Model architecture definitions
│   ├── trainer.py          # Training loop and checkpointing
│   ├── evaluate.py         # Model evaluation
│   ├── main.py             # CLI entry point
│   └── test.py             # Inference and deployment
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/username/cat-breed-classification.git
    cd cat-breed-classification
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Preparation

The dataset should be organized as follows:

```
data/processed/
├── Abyssinian/
│   ├── Abyssinian_1.jpg
│   ├── Abyssinian_2.jpg
│   └── ...
├── Bengal/
│   ├── Bengal_1.jpg
│   └── ...
└── ...
```

## Usage

### Training

Train a model with default settings:

```bash
python src/main.py train
```

Train with custom settings:

```bash
python src/main.py train --backbone shufflenetv2 --batch-size 64 --epochs 50 --lr 0.0005
```

Train using a configuration file:

```bash
python src/main.py train --config-path config/training_config.yaml
```

### Evaluation

Evaluate a trained model:

```bash
python src/main.py evaluate --model-path src/checkpoints/mobilenetv2_20230615_123456/best_state.pth
```

If no model path is provided, the latest model checkpoint will be used:

```bash
python src/main.py evaluate
```

### Prediction

Run inference on a single image:

```bash
python src/main.py predict --input path/to/image.jpg
```

### Video Processing

Process a video file:

```bash
python src/main.py video --input path/to/video.mp4 --output path/to/output.mp4
```

### Webcam

Run real-time inference on webcam feed:

```bash
python src/main.py webcam
```

## Configuration Files

The system supports YAML configuration files. Command-line arguments override configuration file settings.

Example configuration:

```yaml
# Model parameters
backbone: mobilenetv2
pretrained: true
dropout_rate: 0.5

# Optimization parameters
learning_rate: 0.001
weight_decay: 0.0001

# Training parameters
epochs: 30
batch_size: 32
early_stopping: 10
```

For detailed configuration options, see [Configuration Guide](docs/configuration.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.
