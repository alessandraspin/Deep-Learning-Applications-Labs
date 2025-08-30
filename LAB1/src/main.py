import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from models import MLP_2layers, MLP_3layers, ResidualMLP, myCNN, myCNN_improved, myMidCNN, myDeepCNN
from dataloader import get_cifar10_loaders, get_mnist_loaders
from trainer import trainer
from tester import tester
from utils import plot_single_performance, get_model_path
from config import get_config 

if __name__ == "__main__":
    parser = get_config()
    config = parser.parse_args()

    device = torch.device(config.device if torch.cuda.is_available() and "cuda" in config.device else "cpu")
    print(f"Using device: {device}")

    if config.model == "MLP_2layers":
        model = MLP_2layers().to(device)
    elif config.model == "MLP_3layers":
        model = MLP_3layers().to(device)
    elif config.model == "ResidualMLP":
        model = ResidualMLP().to(device)
    elif config.model == "myCNN":
        model = myCNN().to(device)
    elif config.model == "myMidCNN":
        model = myMidCNN().to(device)
    elif config.model == "myDeepCNN":
        model = myDeepCNN().to(device)
    elif config.model == "myCNN_improved":
        model = myCNN_improved().to(device)
    elif config.model == "ResNet18":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    elif config.model == "ResNet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    else:
        raise ValueError("Modello non supportato.")

    if config.dataset == "mnist":
        train_loader, val_loader, test_loader = get_mnist_loaders(config)
    elif config.dataset == "cifar10":
        train_loader, val_loader, test_loader = get_cifar10_loaders(config)

    # model_name = model.__class__.__name__
    save_path = os.path.join(os.path.dirname(os.getcwd()), "models", f"best_{config.model}.pth")

    train_losses, val_losses, val_accuracies, best_val_acc = trainer(model, train_loader, val_loader, device, save_path)
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
    else:
        print(f"Attenzione: file del modello {save_path} non trovato. Test interrotto.")
        exit()

    test_loss, test_accuracy = tester(model, test_loader, device)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    plots_dir = os.path.join(os.path.dirname(os.getcwd()), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_save_path = os.path.join(plots_dir, f"{config.model}_{config.dataset}_performance.pdf")
    plot_single_performance(train_losses, val_losses, val_accuracies, config.model, save_path=plot_save_path)