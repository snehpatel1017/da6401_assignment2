import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import numpy as np
from tqdm.notebook import tqdm  # for progress bar
import wandb
import gc
from sklearn.model_selection import StratifiedShuffleSplit # for stratified sampling

# Clean up GPU memory
gc.collect()
torch.cuda.empty_cache()

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


# Custom transform to handle image orientation
class RotateIfNeeded:
    """Custom transform that rotates portrait images to landscape orientation"""
    def __call__(self, img):
        width, height = img.size
        if height > width:
            return img.transpose(2)  # Rotate 90 degrees if portrait
        return img


def create_prediction_grid(model, test_loader, class_names):
    """
    Creates a visualization grid showing model predictions on test data
    
    Args:
        model: The trained PyTorch model
        test_loader: DataLoader for test dataset
        class_names: List of class names
        
    Returns:
        matplotlib figure object with the prediction grid
    """
    model.eval()
    all_images = []
    all_preds = []
    all_targets = []
    all_probs = []
    
    # Collect images and predictions
    with torch.no_grad():
        for inputs, targets in test_loader:
            if len(all_images) >= 30:  # We need 30 images for 10x3 grid
                break
                
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, preds = torch.max(probs, 1)
            
            # Convert to CPU and save
            batch_images = inputs.cpu().numpy()
            batch_preds = preds.cpu().numpy()
            batch_confidence = confidence.cpu().numpy()
            
            # Add samples to our collections
            for i in range(min(len(batch_images), 30 - len(all_images))):
                all_images.append(batch_images[i])
                all_preds.append(batch_preds[i])
                all_targets.append(targets[i].item())
                all_probs.append(batch_confidence[i])
    
    # Set up the figure with a custom layout
    fig = plt.figure(figsize=(18, 30))
    gs = gridspec.GridSpec(10, 3, figure=fig, wspace=0.2, hspace=0.4)
    
    # Color mapping for correct/incorrect predictions
    correct_color = '#2ecc71'  # Green
    incorrect_color = '#e74c3c'  # Red
    
    # Create custom title for the grid
    fig.suptitle('Model Predictions on Test Data', fontsize=24, y=0.92)
    
    # For each image in our grid
    for i in range(10):
        for j in range(3):
            idx = i * 3 + j
            if idx < len(all_images):
                # Get image and prediction info
                img = all_images[idx].transpose(1, 2, 0)
                # Denormalize the image
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                pred_class = all_preds[idx]
                true_class = all_targets[idx]
                confidence = all_probs[idx]
                is_correct = pred_class == true_class
                color = correct_color if is_correct else incorrect_color
                
                # Create subplot
                ax = fig.add_subplot(gs[i, j])
                
                # Display image with custom border
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add a colored border based on correctness
                for spine in ax.spines.values():
                    spine.set_linewidth(6)
                    spine.set_color(color)
                
                # Create a top banner with the class name
                pred_name = class_names[pred_class]
                true_name = class_names[true_class]
                
                # Add prediction text
                ax.set_title(f"Prediction: {pred_name}", 
                           fontsize=12, color='white', 
                           bbox=dict(facecolor=color, alpha=0.9, pad=5))
                
                # Add ground truth text below image
                ax.annotate(f"Ground Truth: {true_name}", 
                          xy=(0.5, -0.03), xycoords='axes fraction', 
                          fontsize=10, ha='center', va='top',
                          bbox=dict(facecolor='gray', alpha=0.8, pad=3))
                
                # Add confidence score
                ax.annotate(f"Confidence: {confidence:.1%}", 
                          xy=(0.5, 1.02), xycoords='axes fraction', 
                          fontsize=10, ha='center',
                          bbox=dict(facecolor='#3498db', alpha=0.8, pad=1))
                
                # Add an icon to indicate correctness
                if is_correct:
                    ax.annotate('✓', xy=(0.95, 0.95), xycoords='axes fraction', 
                              fontsize=18, ha='right', va='top', color='white',
                              bbox=dict(facecolor=color, alpha=0.8, boxstyle='circle'))
                else:
                    ax.annotate('✗', xy=(0.95, 0.95), xycoords='axes fraction', 
                              fontsize=18, ha='right', va='top', color='white',
                              bbox=dict(facecolor=color, alpha=0.8, boxstyle='circle'))
                
                # Add a small confidence bar
                bar_width = confidence
                bar_height = 0.04
                bar_y = 0.01
                ax.add_patch(patches.Rectangle(
                    (0.1, bar_y), 0.8 * bar_width, bar_height,
                    transform=ax.transAxes, facecolor='#f39c12', alpha=0.8
                ))
                # Add background for full confidence bar
                ax.add_patch(patches.Rectangle(
                    (0.1, bar_y), 0.8, bar_height,
                    transform=ax.transAxes, facecolor='#bdc3c7', alpha=0.3
                ))
    
    # Add a legend
    legend_elements = [
        patches.Patch(facecolor=correct_color, label='Correct Prediction'),
        patches.Patch(facecolor=incorrect_color, label='Incorrect Prediction')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for title and legend
    
    # Save the figure
    plt.savefig('test_predictions.png', dpi=300, bbox_inches='tight')
    
    # Log to wandb if desired
    try:
        wandb.log({"test_predictions_grid": wandb.Image(fig)})
    except:
        print("Couldn't log to wandb, continuing...")
    
    return fig


def create_data_loaders(train_dir, test_dir, input_size=(224, 224), batch_size=32,
                        data_augmentation=False, val_ratio=0.2, num_workers=2):
    """
    Create data loaders for training, validation, and testing.
    Uses stratified sampling to ensure class balance in validation set.
    
    Args:
        train_dir: Directory containing training images organized in class folders
        test_dir: Directory containing test images organized in class folders
        input_size: Tuple of (height, width) for resizing images
        batch_size: Number of samples per batch
        data_augmentation: Whether to apply data augmentation to training set
        val_ratio: Fraction of training data to use for validation
        num_workers: Number of worker threads for data loading
        
    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        test_loader: DataLoader for test set
        num_classes: Number of classes in the dataset
    """
    # Base transform
    base_transform = [
        RotateIfNeeded(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

    # Training transform with optional augmentation
    if data_augmentation:
        train_transform = transforms.Compose([
            base_transform[0],  # RotateIfNeeded
            base_transform[1],  # Resize
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            base_transform[2],  # ToTensor
            base_transform[3]   # Normalize
        ])
    else:
        train_transform = transforms.Compose(base_transform)

    # Test transform (no augmentation)
    test_transform = transforms.Compose(base_transform)

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # Create stratified train/validation split
    targets = np.array([sample[1] for sample in train_dataset.samples])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    # Create samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Training set: {len(train_idx)} images")
    print(f"Validation set: {len(val_idx)} images")
    print(f"Test set: {len(test_dataset)} images")
    print(f"Number of classes: {len(train_dataset.classes)}")

    return train_loader, val_loader, test_loader, len(train_dataset.classes)


class FlexibleCNN(nn.Module):
  
    def __init__(self, in_channels=3, num_filters=32, filter_size=3,
                 activation_fn=nn.ReLU, filter_organization='same',
                 use_batchnorm=False, dropout_rate=0.0,
                 dense_neurons=128, num_classes=10, input_size=(224, 224)):
        """
        Initialize the CNN model
        
        Args:
            in_channels: Number of input channels (3 for RGB images)
            num_filters: Base number of filters in convolutional layers
            filter_size: Size of convolutional filters
            activation_fn: Activation function to use
            filter_organization: Strategy for organizing filter counts across layers
                                ('same', 'double', or 'half')
            use_batchnorm: Whether to use batch normalization
            dropout_rate: Dropout rate for regularization
            dense_neurons: Number of neurons in the dense layer
            num_classes: Number of output classes
            input_size: Size of input images
        """
        super(FlexibleCNN, self).__init__()

        # Determine filter counts based on organization strategy
        if filter_organization == 'same':
            filters = [num_filters] * 5
        elif filter_organization == 'double':
            filters = [num_filters * (2**i) for i in range(5)]
        elif filter_organization == 'half':
            filters = [num_filters // (2**(i)) for i in range(5)]
            filters = [max(16, f) for f in filters]  # Ensure minimum filter count

        # Build layers list
        layers = []

        # First conv block
        layers.append(nn.Conv2d(in_channels, filters[0], kernel_size=filter_size, padding='same'))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(filters[0]))
        layers.append(activation_fn())
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(dropout_rate))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Remaining conv blocks
        for i in range(1, 5):
            layers.append(nn.Conv2d(filters[i-1], filters[i], kernel_size=filter_size, padding='same'))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(filters[i]))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)

        # Calculate output dimensions dynamically
        with torch.no_grad():
            x = torch.randn(1, in_channels, *input_size)
            x = self.features(x)
            self.flattened_size = x.numel() // x.size(0)

        # Classifier
        classifier_layers = [
            nn.Flatten(),
            nn.Linear(self.flattened_size, dense_neurons),
            activation_fn()
        ]

        if dropout_rate > 0:
            classifier_layers.append(nn.Dropout(dropout_rate))

        classifier_layers.append(nn.Linear(dense_neurons, num_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        """Forward pass through the network"""
        x = self.features(x)
        x = self.classifier(x)
        return x


# Dictionary of activation functions
activation_functions = {
    'relu': nn.ReLU,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
    'mish': nn.Mish
}


def train_with_wandb(config=None):
   
    with wandb.init(config=config):
        config = wandb.config

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
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            })
            
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
