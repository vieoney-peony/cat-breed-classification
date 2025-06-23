# Cat Breed Classification Pipeline

This document outlines the end-to-end pipeline of the Cat Breed Classification system, from data preparation to inference.

## Overview

This project implements a deep learning system to classify cat breeds from images. The pipeline consists of several interconnected components:

1. **Data Processing**: Preparing and organizing cat breed images
2. **Model Architecture**: Using pre-trained CNN backbones with custom classification layers
3. **Training Pipeline**: Including data loading, optimization, and checkpointing
4. **Evaluation**: Comprehensive metrics for model assessment
5. **Inference**: Methods for making predictions on new images

## Project Structure

```
project/
├── config/             # Configuration files
│   └── training_config.yaml
├── data/               # Data directory
│   ├── archive/        # Compressed source data
│   ├── raw/            # Raw images
│   └── processed/      # Preprocessed images organized by breed
├── docs/               # Documentation
├── src/                # Source code
│   ├── checkpoints/    # Model checkpoints
│   ├── configs/        # Configuration files
│   ├── utils/          # Utility functions
│   │   ├── data_utils.py     # Data processing utilities
│   │   └── visualization.py  # Visualization utilities
│   ├── main.py         # Main entry point
│   ├── model.py        # Model definitions
│   ├── trainer.py      # Training logic
│   ├── evaluate.py     # Evaluation metrics
│   └── test.py         # Inference/testing code
└── scripts/            # Utility scripts
    └── run_with_config.py
```

## 1. Data Pipeline

### Source Data

-   The dataset consists of images of 21 different cat breeds
-   Raw images are stored in the `data/raw/` directory
-   Source archives are stored in `data/archive/`

### Data Processing

-   Images are organized in the `data/processed/` directory, categorized by breed
-   Each breed has its own subdirectory containing all relevant images
-   During training, images undergo the following transformations:
    -   Resizing to a standard size (224×224 pixels)
    -   Data augmentation (random flips, rotations, color jitter)
    -   Normalization using ImageNet statistics

### Data Loading

-   The `utils/data_utils.py` module handles data loading and preparation
-   Images are loaded into PyTorch DataLoaders with configurable batch sizes
-   The dataset is split into training, validation, and test sets

## 2. Model Architecture

### Backbone Models

-   The system leverages pre-trained CNN models as feature extractors
-   Supported backbones include:
    -   AlexNet
    -   ResNet (various versions)
    -   MobileNet
    -   ShuffleNet
    -   And other torchvision models

### Classification Head

-   The pre-trained backbone is enhanced with:
    -   Custom fully connected layers
    -   Dropout for regularization (configurable rate)
    -   Final softmax layer for breed probability distribution

### Configuration

-   Model architecture settings are defined in configuration files
-   Parameters include:
    -   Backbone model choice
    -   Number of classes (21 cat breeds)
    -   Dropout rate (default: 0.5)
    -   Whether to use pre-trained weights

## 3. Training Pipeline

### Training Process

-   The `CatBreedTrainer` class in `trainer.py` orchestrates the training process
-   Training follows these steps:
    1. Initialize model with selected backbone
    2. Set up optimizer (default: AdamW)
    3. Configure learning rate scheduler (default: ReduceLROnPlateau)
    4. Train for a specified number of epochs
    5. Save checkpoints periodically
    6. Monitor validation metrics
    7. Apply early stopping when necessary

### Configuration

-   Training parameters are defined in `config/training_config.yaml`
-   Parameters include:
    -   Batch size
    -   Learning rate
    -   Weight decay
    -   Number of epochs
    -   Early stopping patience
    -   Optimizer choice

### Monitoring

-   The system logs training progress with:
    -   Loss curves
    -   Accuracy metrics
    -   Learning rate changes
    -   Time per epoch

## 4. Evaluation Pipeline

### Metrics

-   The `Evaluator` class in `evaluate.py` handles comprehensive model assessment
-   Metrics calculated include:
    -   Accuracy
    -   Precision
    -   Recall
    -   F1 Score
    -   Confusion Matrix

### Visualization

-   Performance visualizations include:
    -   Confusion matrix heatmaps
    -   ROC curves
    -   Learning curves
    -   Classification reports

### Output

-   Evaluation results are saved to:
    -   JSON format for programmatic use
    -   Visual plots for human review
    -   Detailed logs for analysis

## 5. Inference Pipeline

### Prediction Process

-   The `CatBreedPredictor` class in `test.py` handles inference on new images
-   The inference process follows these steps:
    1. Load a trained model from checkpoint
    2. Preprocess the input image
    3. Run the image through the model
    4. Get prediction probabilities
    5. Return the predicted breed with confidence score

### Deployment Options

-   Single image prediction
-   Batch processing of multiple images
-   Video processing for breed identification
-   Webcam integration for real-time classification

## Usage

### Training

```bash
python src/main.py train --config-path config/training_config.yaml
```

### Evaluation

```bash
python src/main.py evaluate --checkpoint-dir path/to/checkpoint
```

### Inference

```bash
python src/main.py predict --image-path path/to/cat/image.jpg
```

### Video Processing

```bash
python src/main.py video --video-path path/to/video.mp4
```

### Webcam

```bash
python src/main.py webcam
```

## Configuration

The system is highly configurable through YAML configuration files. The main configuration parameters include:

```yaml
backbone: CatBreedClassifier
batch_size: 64
checkpoint_dir: path/to/checkpoints
device: cuda
dropout_rate: 0.5
early_stopping: 10
epochs: 3
learning_rate: 0.001
num_classes: 21
optimizer: AdamW
pretrained: true
scheduler: ReduceLROnPlateau
weight_decay: 0.0001
```

These parameters control all aspects of the training and inference pipeline, allowing for easy experimentation with different model architectures and hyperparameters.

## Extension Points

The pipeline is designed to be modular and extensible. Potential extension points include:

1. Adding new backbone architectures
2. Implementing additional data augmentation techniques
3. Integrating advanced training techniques (mixed precision, gradient accumulation)
4. Adding support for transfer learning and fine-tuning
5. Implementing model interpretability features
6. Adding support for model quantization and optimization
