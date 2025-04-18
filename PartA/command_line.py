
import argparse
import wandb
from .train import train_with_wandb # Import the training function from your script
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring the model training sweep"""
    parser = argparse.ArgumentParser(description='Configure a sweep for training a CNN model')
    
    # Dataset paths
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Directory containing training images organized in class folders')
    parser.add_argument('--test_dir', type=str, required=True, 
                        help='Directory containing test images organized in class folders')
    
    # Sweep method
    parser.add_argument('--method', type=str, default='bayes', choices=['grid', 'random', 'bayes'],
                        help='Sweep search method (grid, random, or bayes)')
    
    # Model parameters - Multiple values can be provided for sweep
    parser.add_argument('--batch_size', type=int, nargs='+', default=[32],
                        help='Batch size(s) for training (can provide multiple for sweep)')
    
    parser.add_argument('--input_size', type=int, nargs='+', default=[600],
                        help='Size(s) to resize input images to (can provide multiple for sweep)')
    
    parser.add_argument('--num_filters', type=int, nargs='+', default=[32],
                        help='Number of filters in first conv layer (can provide multiple for sweep)')
    
    parser.add_argument('--filter_size', type=int, nargs='+', default=[5],
                        help='Size of convolutional filters (can provide multiple for sweep)')
    
    parser.add_argument('--activation', type=str, nargs='+', default=['gelu'],
                        choices=['relu', 'gelu', 'silu', 'mish'],
                        help='Activation function(s) to use (can provide multiple for sweep)')
    
    parser.add_argument('--filter_organization', type=str, nargs='+', default=['double'],
                        choices=['same', 'double', 'half'],
                        help='Filter organization strategy (can provide multiple for sweep)')
    
    parser.add_argument('--data_augmentation', type=str, nargs='+', default=['True'],
                        choices=['True', 'False'],
                        help='Whether to use data augmentation (can provide multiple for sweep)')
    
    parser.add_argument('--use_batchnorm', type=str, nargs='+', default=['False'],
                        choices=['True', 'False'],
                        help='Whether to use batch normalization (can provide multiple for sweep)')
    
    parser.add_argument('--dropout_rate', type=float, nargs='+', default=[0.2],
                        help='Dropout rate(s) for regularization (can provide multiple for sweep)')
    
    parser.add_argument('--dense_neurons', type=int, nargs='+', default=[128],
                        help='Number of neurons in dense layer (can provide multiple for sweep)')
    
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[0.0001],
                        help='Learning rate(s) for optimizer (can provide multiple for sweep)')
    
    # Fixed parameter
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (fixed for all sweep runs)')
    
    # WandB parameters
    parser.add_argument('--project', type=str, default='da6401_assignment2',
                        help='WandB project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='WandB entity (username or team name)')
    parser.add_argument('--count', type=int, default=1,
                        help='Number of sweep runs to execute')
    
    return parser.parse_args()


def convert_boolean_args(arg_list: List[str]) -> List[bool]:
    """Convert string boolean arguments to actual boolean values"""
    return [arg.lower() == 'true' for arg in arg_list]


def build_sweep_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build a sweep configuration dictionary from parsed command-line arguments
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary containing the sweep configuration
    """
    # Convert string boolean arguments to actual booleans
    use_batchnorm = convert_boolean_args(args.use_batchnorm)
    data_augmentation = convert_boolean_args(args.data_augmentation)
    
    # Build parameter dictionary
    parameters = {
        'train_dir': {'value': args.train_dir},
        'test_dir': {'value': args.test_dir},
        'epochs': {'value': args.epochs}
    }
    
    # Add parameters with potentially multiple values for sweeping
    param_lists = {
        'batch_size': args.batch_size,
        'input_size': args.input_size,
        'num_filters': args.num_filters,
        'filter_size': args.filter_size,
        'activation': args.activation,
        'filter_organization': args.filter_organization,
        'data_augmentation': data_augmentation,
        'use_batchnorm': use_batchnorm,
        'dropout_rate': args.dropout_rate,
        'dense_neurons': args.dense_neurons,
        'learning_rate': args.learning_rate
    }
    
    # Add parameters to sweep config
    for param_name, param_values in param_lists.items():
        if len(param_values) == 1:
            # If only one value, use it as a fixed parameter
            parameters[param_name] = {'value': param_values[0]}
        else:
            # If multiple values, set up for sweeping
            parameters[param_name] = {'values': param_values}
    
    # Create the final sweep configuration
    sweep_config = {
        'method': args.method,
        'metric': {
            'name': 'val_accuracy',
            'goal': 'maximize'
        },
        'parameters': parameters
    }
    
    return sweep_config


def get_sweep_config():
    """Main function to parse arguments and build sweep configuration"""
    args = parse_args()
    sweep_config = build_sweep_config(args)
    
    # Print the sweep configuration
    import json
    print(json.dumps(sweep_config, indent=2))
    
    # Return the configuration values that could be used with train_with_wandb
    return {
        'sweep_config': sweep_config,
        'project': args.project,
        'entity': args.entity,
        'count': args.count
    }

class Custom_config:
    def __init__(self, sweep_config: Dict[str, Any], project: str, entity: str, count: int):
        
        self.project = project
        self.entity = entity
        self.count = count
        self.train_dir = sweep_config['parameters']['train_dir']['value']
        self.test_dir = sweep_config['parameters']['test_dir']['value']
        self.epochs = sweep_config['parameters']['epochs']['value']
        self.batch_size = sweep_config['parameters']['batch_size']['value']
        self.input_size = sweep_config['parameters']['input_size']['value']
        self.num_filters = sweep_config['parameters']['num_filters']['value']
        self.filter_size = sweep_config['parameters']['filter_size']['value']
        self.activation = sweep_config['parameters']['activation']['value']
        self.filter_organization = sweep_config['parameters']['filter_organization']['value']
        self.data_augmentation = sweep_config['parameters']['data_augmentation']['value']
        self.use_batchnorm = sweep_config['parameters']['use_batchnorm']['value']
        self.dropout_rate = sweep_config['parameters']['dropout_rate']['value']
        self.dense_neurons = sweep_config['parameters']['dense_neurons']['value']
        self.learning_rate = sweep_config['parameters']['learning_rate']['value']
       

def convert_sweep_config_to_dict(sweep_config,project,entity,count):
    CONFIG = Custom_config(
        sweep_config=sweep_config,
        project=project,
        entity=entity,
        count=count
    )
    return CONFIG 


if __name__ == "__main__":
    config = get_sweep_config()
   
    CONFIG = convert_sweep_config_to_dict(config["sweep_config"],config["project"],config["entity"],config["count"])
    wandb.login()
    train_with_wandb(CONFIG)

    # Create the sweep
    # sweep_id = wandb.sweep(config["sweep_config"], project="da6401_assignment2")

    # Run the sweep (limit to 20 runs for efficiency)
    # wandb.agent(sweep_id, entity="cs24m048-iit-madras", project="da6401_assignment2", function=train_with_wandb, count=2)
    
   