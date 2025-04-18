
import argparse
import wandb

from typing import Dict, Any, List

from PartB.train import train_model_sweep


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuring the model training sweep"""
    parser = argparse.ArgumentParser(description='Configure a sweep for training a CNN model')
    
    # Dataset paths
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Directory containing datasets')
       
    return parser.parse_args()




class Custom_config:
    def __init__(self, dataset_path:str):
        self.dataset_path = dataset_path
        self.freezing_strategy = 'freeze_first_k_layers'
        self.freeze_k_layers = 3  # Only used when freezing_strategy is 'freeze_first_k_layers'
        self.learning_rate = 0.0001
        self.batch_size = 16
        self.epochs = 10  # Fixed at 10 epochs as requested
        self.image_size = 400
        self.num_workers = 4
        self.accumulation_steps = 4
       


if __name__ == "__main__":
    args = parse_args()
    CONFIG = Custom_config(args.dataset_path)
    # wandb.login("")
    # wandb.init(project="da6401_assignment2", entity="cs24m048-iit-madras")
    train_model_sweep(CONFIG)

    # Create the sweep
    # sweep_id = wandb.sweep(config["sweep_config"], project="da6401_assignment2")

    # Run the sweep (limit to 20 runs for efficiency)
    # wandb.agent(sweep_id, entity="cs24m048-iit-madras", project="da6401_assignment2", function=train_with_wandb, count=2)
    
   