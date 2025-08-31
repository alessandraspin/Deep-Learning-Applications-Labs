import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torchvision.models import resnet18, ResNet18_Weights
from models import DynamicMLP_improved, DynamicMLP
from dataloader import get_cifar10_loaders, get_mnist_loaders
from trainer import trainer
from tester import tester
from utils import plot_single_performance, get_model_path
from config import get_config
import wandb


if __name__ == "__main__":
    parser = get_config()
    config = parser.parse_args()

    device = torch.device(config.device if torch.cuda.is_available() and "cuda" in config.device else "cpu")
    print(f"Using device: {device}")

    # Dataset MNIST
    config.dataset = "mnist"
    train_loader, val_loader, test_loader = get_mnist_loaders(config)
    input_dim = 28 * 28
    num_classes = 10

    # Configurazioni MLP
    mlp_configs = {
        "MLP_2layers": [128, 64],
        "MLP_4layers": [128, 64, 64, 32],
        "MLP_8layers": [128, 128, 64, 64, 32, 32, 16, 16]
    }

    all_models_data = {}

    print("\nStarting comparative analysis of MLPs with varying layers...")

    # ==================================
    # DynamicMLP
    # ==================================
    for model_name, hidden_sizes in mlp_configs.items():
        run_name = f"Dynamic_improved_{model_name}"
        # run_name = f"Dynamic_{model_name}"
        print(f"\n--- Running experiment for {run_name} ---")

        wandb.init(
            project=config.wandb_project,
            name=run_name,
            group="MLP_Comparison",
            config={
                "model_name": run_name,
                "hidden_layers": len(hidden_sizes),
                "hidden_sizes": hidden_sizes,
                "learning_rate": config.lr,
                "batch_size": config.batch_size,
                "max_epoch": config.max_epoch,
                "optimizer": config.optimizer,
            }
        )

        # Istanzia modello
        model = DynamicMLP_improved(input_dim, hidden_sizes, num_classes).to(device)
        # model = DynamicMLP(input_dim, hidden_sizes, num_classes).to(device)

        # Training
        train_losses, val_losses, val_accuracies, _ = trainer(model, train_loader, val_loader, device, config)

        # Testing
        save_path = get_model_path(run_name)
        if os.path.exists(save_path):
            try:
                model.load_state_dict(torch.load(save_path), strict=True)
                test_loss, test_accuracy = tester(model, test_loader, device)
                print(f"Final results for {run_name}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
                if wandb.run is not None:
                    wandb.log({
                        "test_accuracy": test_accuracy,
                        "model_name": run_name
                    })
            except RuntimeError as e:
                print(f"⚠️ State_dict mismatch for {run_name}, skipping load.\n{e}")
                test_loss, test_accuracy = None, None
        else:
            print(f"⚠️ Model file {save_path} not found. Skipping test.")
            test_loss, test_accuracy = None, None

        all_models_data[run_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_accuracies": val_accuracies,
            "test_accuracy": test_accuracy
        }

        wandb.finish() 

    print("\n--- Comparative analysis complete ---")

    # Stampa tutti i risultati di testing
    print("\nOverall Test Accuracies:")
    for name, data in all_models_data.items():
        if data["test_accuracy"] is not None:
            print(f"{name}: {data['test_accuracy']:.4f}")
    
    # Calcola la larghezza massima dei nomi dei modelli
    max_name_len = max(len(name) for name in all_models_data.keys())
    col_width = max(max_name_len, len("Model")) + 2  # +2 per padding

    # Header
    header = f"| {'Model'.ljust(col_width)} | {'Test Accuracy'.ljust(12)} |"
    separator = f"|{'-'*(col_width+2)}|{'-'*14}|"

    print("\nOverall Test Accuracies:")
    print(separator)
    print(header)
    print(separator)

    # Righe della tabella
    for name, data in all_models_data.items():
        if data["test_accuracy"] is not None:
            print(f"| {name.ljust(col_width)} | {data['test_accuracy']:<12.4f} |")

    print(separator)