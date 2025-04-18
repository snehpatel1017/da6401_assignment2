import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.cuda import amp
from tqdm import tqdm
import numpy as np
import wandb
from sklearn.model_selection import StratifiedShuffleSplit

from PartB.Utility import RotateIfNeeded, center_crop_image, count_parameters, freeze_all_layers_except_last, freeze_first_k_layers, no_freezing

# Free up GPU memory
torch.cuda.empty_cache()

# Set PyTorch memory allocation config
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")



# Function that will be called by wandb sweep
def train_model_sweep(config=None):
    # with wandb.init(config=config):
        # Access all hyperparameters through wandb.config
        # config = wandb.config
        
        # Data transformations
        train_transform = transforms.Compose([
            RotateIfNeeded(),
            transforms.Lambda(lambda img: center_crop_image(img, target_size=(config.image_size, config.image_size))),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            RotateIfNeeded(),
            transforms.Lambda(lambda img: center_crop_image(img, target_size=(config.image_size, config.image_size))),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            RotateIfNeeded(),
            transforms.Lambda(lambda img: center_crop_image(img, target_size=(config.image_size, config.image_size))),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load full training dataset
        print(f"Loading datasets from {config.dataset_path}")
        full_train_dataset = datasets.ImageFolder(config.dataset_path + "/train", transform=train_transform)
        
        # Get targets for stratified split
        targets = np.array([label for _, label in full_train_dataset.samples])
        
        # Create stratified split - 80% train, 20% validation
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
        
        # Verify the class distribution
        train_labels = targets[train_idx]
        val_labels = targets[val_idx]
        unique_labels = np.unique(targets)
        
        print("\nClass distribution:")
        for label in unique_labels:
            train_count = np.sum(train_labels == label)
            val_count = np.sum(val_labels == label)
            total_count = np.sum(targets == label)
            print(f"Class {full_train_dataset.classes[label]}: "
                  f"Train {train_count}/{total_count} ({train_count/total_count:.2f}), "
                  f"Val {val_count}/{total_count} ({val_count/total_count:.2f})")
        
        # Create Subset objects for train and validation
        train_subset = Subset(full_train_dataset, train_idx)
        val_subset = Subset(full_train_dataset, val_idx)
        
        # Load the test dataset (using the val folder)
        test_dataset = datasets.ImageFolder(config.dataset_path + "/val", transform=test_transform)
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=config.batch_size, 
            shuffle=True, 
            num_workers=config.num_workers, 
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers, 
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers, 
            pin_memory=True
        )
        
        # Get number of classes
        num_classes = len(full_train_dataset.classes)
        print(f"\nNumber of classes: {num_classes}")
        print(f"Full training dataset: {len(full_train_dataset)} images")
        print(f"Training subset: {len(train_subset)} images ({len(train_subset)/len(full_train_dataset):.1%})")
        print(f"Validation subset: {len(val_subset)} images ({len(val_subset)/len(full_train_dataset):.1%})")
        print(f"Test dataset: {len(test_dataset)} images")

        # Load model
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # Replace classifier
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Apply freezing strategy based on config
        if config.freezing_strategy == "freeze_all_except_last":
            model = freeze_all_layers_except_last(model)
        elif config.freezing_strategy == "freeze_first_k_layers":
            model = freeze_first_k_layers(model, config.freeze_k_layers)
        elif config.freezing_strategy == "no_freezing":
            model = no_freezing(model)
        else:
            raise ValueError(f"Unknown freezing strategy: {config.freezing_strategy}")
        
        # Count parameters
        trainable_params, non_trainable_params = count_parameters(model)
        total_params = trainable_params + non_trainable_params
        percent_trainable = 100 * trainable_params / total_params
        
        print(f"\nTrainable parameters: {trainable_params:,} ({percent_trainable:.2f}%)")
        print(f"Non-trainable parameters: {non_trainable_params:,} ({100-percent_trainable:.2f}%)")
        print(f"Total parameters: {total_params:,}")
        
        # Log parameter counts to wandb
        # wandb.log({
        #     "trainable_params": trainable_params,
        #     "non_trainable_params": non_trainable_params,
        #     "percent_trainable": percent_trainable
        # })
        
        # Move model to device
        model = model.to(device)
        
        # Define loss function and optimizer (only train parameters that require gradients)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=config.learning_rate
        )
        
        # Initialize mixed precision scaler
        scaler = amp.GradScaler()
        
        # Create directory for saving models
        os.makedirs(f"models/{123}", exist_ok=True)
        
        # Training loop - fixed at 10 epochs, no early stopping
        best_val_accuracy = 0.0
        
        for epoch in range(config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            optimizer.zero_grad()
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} Training")
            
            for i, (inputs, labels) in enumerate(progress_bar):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Use mixed precision
                with amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / config.accumulation_steps
                
                # Scale gradients and backpropagate
                scaler.scale(loss).backward()
                
                # Update weights after accumulating gradients
                if (i + 1) % config.accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                # Track statistics
                train_loss += loss.item() * inputs.size(0) * config.accumulation_steps
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * config.accumulation_steps:.4f}",
                    "accuracy": f"{100.0 * train_correct / train_total:.2f}%"
                })
            
            train_loss = train_loss / len(train_subset)
            train_accuracy = 100.0 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} Validation")
                for inputs, labels in progress_bar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "accuracy": f"{100.0 * val_correct / val_total:.2f}%"
                    })
            
            val_loss = val_loss / len(val_subset)
            val_accuracy = 100.0 * val_correct / val_total
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{config.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Log to wandb
            # wandb.log({
            #     "epoch": epoch + 1,
            #     "train_loss": train_loss,
            #     "train_accuracy": train_accuracy,
            #     "val_loss": val_loss,
            #     "val_accuracy": val_accuracy
            # })
            
            # Save the best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(f"models/", "best_model.pth"))
                print(f"New best model saved with validation accuracy: {best_val_accuracy:.2f}%")
        
        # Load the best model for final evaluation on test set
        print("\nLoading best model for final test evaluation...")
        model.load_state_dict(torch.load(os.path.join(f"models/", "best_model.pth")))
        model.eval()
        
        # Perform test evaluation
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        print("Test Evaluation")
        progress_bar = tqdm(test_loader, desc="Testing on val folder (test set)")
        
        with torch.no_grad():
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "accuracy": f"{100.0 * test_correct / test_total:.2f}%"
                })
        
        test_loss = test_loss / len(test_dataset)
        test_accuracy = 100.0 * test_correct / test_total
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
        
        # Log final metrics to wandb
        # wandb.log({
        #     "test_loss": test_loss,
        #     "test_accuracy": test_accuracy,
        #     "best_val_accuracy": best_val_accuracy
        # })

# Define sweep configuration
sweep_config = {
    'method': 'grid',  # Try all combinations since we have a focused set of strategies
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'freezing_strategy': {
            'values': ['freeze_all_except_last', 'freeze_first_k_layers', 'no_freezing']
        },
        'freeze_k_layers': {
            'value': 3  # Only used when freezing_strategy is 'freeze_first_k_layers'
        },
        'learning_rate': {
            'value': 0.0001
        },
        'batch_size': {
            'value': 16
        },
        'epochs': {
            'value': 10  # Fixed at 10 epochs as requested
        },
        'image_size': {
            'value': 400  # Sweet spot as per your findings
        },
        'num_workers': {
            'value': 4
        },
        'accumulation_steps': {
            'value': 4
        },
        'dataset_path': {
            'value': '/kaggle/working/nature_12k/inaturalist_12K'  # Update with your dataset path
        }
    }
}


