# Configuration Guide

This guide explains how to use YAML configuration files with the Cat Breed Classification system.

## Using Configuration Files

The system supports loading configurations from YAML files using the `--config-path` argument:

```bash
python src/main.py train --config-path config/training_config.yaml
```

You can override any configuration parameter by passing it as a command-line argument:

```bash
python src/main.py train --config-path config/training_config.yaml --lr 0.0005 --epochs 50
```

In the example above, the learning rate and number of epochs from the command line will override the values in the configuration file.

## Configuration Parameters

### Model Parameters

| Parameter    | Description                        | Default     |
| ------------ | ---------------------------------- | ----------- |
| backbone     | Model architecture to use          | mobilenetv2 |
| pretrained   | Whether to use pre-trained weights | true        |
| dropout_rate | Dropout rate for regularization    | 0.5         |

### Optimization Parameters

| Parameter     | Description                         | Default |
| ------------- | ----------------------------------- | ------- |
| learning_rate | Initial learning rate               | 0.001   |
| weight_decay  | Weight decay (L2 penalty)           | 0.0001  |
| optimizer     | Optimizer to use (AdamW, SGD, etc.) | AdamW   |

### Training Parameters

| Parameter      | Description                    | Default |
| -------------- | ------------------------------ | ------- |
| epochs         | Number of training epochs      | 30      |
| batch_size     | Batch size for training        | 32      |
| early_stopping | Patience for early stopping    | 10      |
| img_size       | Input image size               | 224     |
| num_workers    | Number of data loading workers | 4       |

## Example Configuration

```yaml
# Model parameters
backbone: mobilenetv2
pretrained: true
dropout_rate: 0.5

# Optimization parameters
learning_rate: 0.001
weight_decay: 0.0001
optimizer: AdamW

# Training parameters
epochs: 30
batch_size: 32
early_stopping: 10
device: cuda
img_size: 224
num_workers: 4
```

## Saved Configurations

During training, the system automatically saves the configuration to the checkpoint directory as `training_config.yaml`. This allows you to easily reproduce experiments by using this saved configuration file.

To use a saved configuration from a previous training run:

```bash
python src/main.py train --config-path checkpoints/mobilenetv2_20230615_123456/training_config.yaml
```
