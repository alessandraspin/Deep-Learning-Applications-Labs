import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Configuration for model training and evaluation.")

    # Parametri generali per il dataset
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"], help="Choose mnist or cifar10.")

    # Parametri di training
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for the optimizer.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training and validation.")
    parser.add_argument("--max_epoch", type=int, default=100, help="Maximum number of epochs for training.")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer to use (SGD or Adam).")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay (L2 penalty) for the optimizer.")
    
    # Parametri per il dataloader
    parser.add_argument("--num_workers", type=int, default=4, help="Number of subprocesses to use for data loading.")
    parser.add_argument("--val_size", type=int, default=5000, help="Number of samples to use for the validation set.")

    # Parametri per il modello
    parser.add_argument("--model", type=str, default="ResNet18", help="Name of the model to use.")
    
    # Parametri di logging e salvataggio
    parser.add_argument("--wandb_project", type=str, default="LAB1", help="WandB project name.")
    #parser.add_argument("--save_dir", type=str, default="../models", help="Directory to save the best model checkpoints.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use for training (e.g., 'cuda:0' or 'cpu').")
                        
    return parser