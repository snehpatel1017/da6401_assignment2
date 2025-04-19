# Deep Learning Assignment 2: Image Classification with iNaturalist Dataset

This repository contains the implementation of a deep learning project focused on image classification using the iNaturalist dataset. The project is structured in two parts:

- **Part A**: Implementation of a custom CNN model from scratch
- **Part B**: Transfer learning with pre-trained models (EfficientNetV2)

## Repository Structure

```
├── PartA/
│   ├── __pycache__/
│   ├── Model.py                  # CNN model architecture definition
│   ├── Utility_functions.py      # Helper functions for data processing and visualization
│   ├── command_line.py           # Main execution script with CLI support
│   ├── train.py                  # Training, validation and testing functions
│
├── PartB/
│   ├── __pycache__/
│   ├── Utility.py                # Helper functions for Part B
│   ├── executor.py               # Entry point for transfer learning experiments
│   ├── train.py                  # Training functions for transfer learning
│
└── da6401_assignment_2.ipynb     # Jupyter notebook with comprehensive experiments
```

## Setup and Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/snehpatel1017/da6401_assignment2.git
   ```

2. **Install dependencies**:

   ```bash
   pip install torch torchvision tqdm numpy matplotlib sklearn wandb
   ```

3. **Download the iNaturalist dataset**:

   ```bash
   wget https://storage.googleapis.com/inaturalist_12k/inaturalist_12k.zip
   unzip inaturalist_12k.zip -d nature_12k
   ```

## Part A: Custom CNN Implementation

### Usage

The entry point for Part A is `command_line.py`, which provides a command-line interface to train and evaluate the custom CNN model.

```bash
python -m PartA.command_line \
  --train_dir "/path/to/inaturalist_12K/train" \
  --test_dir "/path/to/inaturalist_12K/val"
```

### Supported Arguments

| Argument                | Description                                                    | Default | Required |
| ----------------------- | -------------------------------------------------------------- | ------- | -------- |
| `--train_dir`           | Path to training data directory                                | —       | Yes      |
| `--test_dir`            | Path to test/validation data directory                         | —       | Yes      |
| `--input_size`          | Input image size                                               | 224     | No       |
| `--batch_size`          | Batch size for training                                        | 32      | No       |
| `--epochs`              | Number of training epochs                                      | 10      | No       |
| `--learning_rate`       | Learning rate for optimizer                                    | 0.001   | No       |
| `--num_filters`         | Number of filters in convolutional layers                      | 32      | No       |
| `--filter_size`         | Size of convolutional filters                                  | 3       | No       |
| `--activation`          | Activation function (`relu`, `gelu`, `silu`, `mish`)           | relu    | No       |
| `--dense_neurons`       | Number of neurons in dense layer                               | 128     | No       |
| `--filter_organization` | Filter organization strategy (`same`, `double`, `half`)        | same    | No       |
| `--use_batchnorm`       | Whether to use batch normalization (`True`/`False`)            | False   | No       |
| `--dropout_rate`        | Dropout rate for regularization                                | 0.0     | No       |
| `--data_augmentation`   | Whether to use data augmentation (`True`/`False`)              | False   | No       |


## Part B: Transfer Learning with Pre-trained Models

### Usage

The entry point for Part B is `executor.py`, which runs transfer learning experiments using EfficientNetV2.

```bash
python -m PartB.executor \
  --dataset_path "path/to/inaturalist_12k"
```

### Supported Arguments

| Argument              | Description                                                                                | Default                  | Required |
| --------------------- | ------------------------------------------------------------------------------------------ | ------------------------ | -------- |
| `--dataset_path`      | Base path to the dataset directory                                                         | —                        | Yes      |


## Key Features

- **Custom CNN Architecture**: Flexible CNN model with configurable parameters.
- **Transfer Learning**: Implementation of three different transfer learning strategies.
- **Data Processing**: Image rotation handling and center cropping optimizations.
- **Memory Efficiency**: Gradient accumulation and mixed precision training.
- **Visualization**: Prediction grid generation for model evaluation.
- **Experiment Tracking**: Optional Weights & Biases integration for experiment tracking.

## Results

The experiments demonstrate that:

1. **Part A**: The optimal model configuration uses double filter organization, GELU activation, and moderate dropout (0.2).
2. **Part B**: The `freeze_first_k_layers` strategy achieves similar performance to training all layers while being more computationally efficient.
3. Using 400×400 input images provides the optimal balance between accuracy and training speed.

Wandb Report : https://api.wandb.ai/links/cs24m048-iit-madras/t2udzuv9
