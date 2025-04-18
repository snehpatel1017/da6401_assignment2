import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm  # for progress bar
import wandb
import gc
from .Model import FlexibleCNN  # Import your model class
from .Utility_functions import create_prediction_grid , create_data_loaders # Import your data loader function
import os
os.environ['TQDM_DISABLE'] = '0'  # Force enable tqdm

# Clean up GPU memory
# gc.collect()
# torch.cuda.empty_cache()

# Set up device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the class names for the inaturalist dataset
CLASSES = [
    'Amphibia',
    'Animalia',
    'Arachnida',
    'Aves',
    'Fungi',
    'Insecta',
    'Mammalia',
    'Mollusca',
    'Plantae',
    'Reptilia'
    ]

# Print GPU information if using CUDA
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory Usage:")
    print(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
    print(f"Cached: {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")





# Dictionary of activation functions
activation_functions = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'mish': nn.Mish
}


def train_with_wandb(config=None):
   
    # with wandb.init(config=config):
    #     config = wandb.config

        # Create data loaders
        train_loader, val_loader, test_loader, num_classes = create_data_loaders(
            train_dir=config.train_dir,
            test_dir=config.test_dir,
            input_size=(config.input_size, config.input_size),
            batch_size=config.batch_size,
            data_augmentation=config.data_augmentation
        )

        # Create model
        model = FlexibleCNN(
            num_filters=config.num_filters,
            filter_size=config.filter_size,
            activation_fn=activation_functions[config.activation],
            filter_organization=config.filter_organization,
            use_batchnorm=config.use_batchnorm,
            dropout_rate=config.dropout_rate,
            dense_neurons=config.dense_neurons,
            num_classes=num_classes,
            input_size=(config.input_size, config.input_size)
        )

        # Move model to device
        model = model.to(device)

        # Print model info
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        # Training loop
        for epoch in range(config.epochs):
            # Training phase
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]") as pbar:
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(device), targets.to(device)

                    # Forward pass
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    # Track metrics
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    # Update progress bar
                    pbar.set_postfix({
                        'loss': train_loss / (pbar.n + 1),
                        'acc': 100. * correct / total
                    })
                    del inputs, targets, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            train_loss = train_loss / len(train_loader)
            train_accuracy = 100. * correct / total

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]") as pbar:
                    for inputs, targets in pbar:
                        inputs, targets = inputs.to(device), targets.to(device)

                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        # Track metrics
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                        # Update progress bar
                        pbar.set_postfix({
                            'loss': val_loss / (pbar.n + 1),
                            'acc': 100. * correct / total
                        })
                        # Free memory
                        del inputs, targets, outputs, loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            val_loss = val_loss / len(val_loader)
            val_accuracy = 100. * correct / total

            # Test phase
            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                with tqdm(test_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Test]") as pbar:
                    for inputs, targets in pbar:
                        inputs, targets = inputs.to(device), targets.to(device)

                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                        # Track metrics
                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                        # Update progress bar
                        pbar.set_postfix({
                            'loss': test_loss / (pbar.n + 1),
                            'acc': 100. * correct / total
                        })
                        # Free memory
                        del inputs, targets, outputs, loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            test_loss = test_loss / len(test_loader)
            test_accuracy = 100. * correct / total

            # Log metrics to wandb
            # wandb.log({
            #     'epoch': epoch + 1,
            #     'train_loss': train_loss,
            #     'train_accuracy': train_accuracy,
            #     'val_loss': val_loss,
            #     'val_accuracy': val_accuracy,
            #     'test_loss': test_loss,
            #     'test_accuracy': test_accuracy
            # })
            
            # Generate and log prediction grid
            create_prediction_grid(model, test_loader, CLASSES)

            print(f"Epoch {epoch+1}/{config.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

        return model, val_accuracy


# Define sweep configuration
sweep_config = {
    'method': 'grid',  # Grid search method
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'train_dir': {
            'value': '/kaggle/working/nature_12k/inaturalist_12K/train'
        },
        'test_dir': {
            'value': '/kaggle/working/nature_12k/inaturalist_12K/val'
        },
        'batch_size': {
            'values': [32]  # Small batch sizes to avoid OOM
        },
        'input_size': {
            'values': [600]
        },
        'num_filters': {
            'values': [32]  # Filter counts
        },
        'filter_size': {
            'values': [5]  # Filter sizes
        },
        'activation': {
            'values': ['gelu']  # Activation functions
        },
        'filter_organization': {
            'values': ['double']  # Filter organization strategies
        },
        'data_augmentation': {
            'values': [True]  # Whether to use data augmentation
        },
        'use_batchnorm': {
            'values': [False]  # Whether to use batch normalization
        },
        'dropout_rate': {
            'values': [0.2, 0.0]  # Dropout rates
        },
        'dense_neurons': {
            'values': [128]  # Number of neurons in dense layer
        },
        'learning_rate': {
            'values': [0.0001]  # Learning rates
        },
        'epochs': {
            'value': 10  # Fixed number of epochs for all runs
        }
    }
}
