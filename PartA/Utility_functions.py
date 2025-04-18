from matplotlib import gridspec, patches, pyplot as plt
import numpy as np
import torch
import wandb
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit

# Custom transform to handle image orientation
class RotateIfNeeded:
    """Custom transform that rotates portrait images to landscape orientation"""
    def __call__(self, img):
        width, height = img.size
        if height > width:
            return img.transpose(2)  # Rotate 90 degrees if portrait
        return img



######## Function to create a grid of predictions from the model ########
# This function creates a grid of images with their predicted and true labels, along with confidence scores.
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
                
            inputs = inputs.to('cuda')
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
    # try:
    #     wandb.log({"test_predictions_grid": wandb.Image(fig)})
    # except:
    #     print("Couldn't log to wandb, continuing...")
    
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

