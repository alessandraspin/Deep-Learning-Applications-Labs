import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import os
import logging
from tqdm import tqdm
import wandb


# La funzione trainer accetta ora l'oggetto config come argomento
def trainer(model, dl_train, dl_val, device, save_path, config):
    # Path di salvataggio modelli (sempre in LAB1/models/)
    lab1_root = os.path.dirname(os.getcwd())  # sale di una cartella
    save_dir = os.path.join(lab1_root, "models")
    os.makedirs(save_dir, exist_ok=True)

    # Nome del modello e percorso completo
    model_name = model.__class__.__name__
    #save_path = os.path.join(save_dir, f"best_{model_name}.pth")

    # Ottimizzatore e scheduler
    optimizer_choice = config.optimizer
    lr = config.lr
    weight_decay = config.weight_decay
    max_epoch = config.max_epoch

    if optimizer_choice == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_choice == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = ExponentialLR(optimizer, gamma=0.9)

    wandb.init(
        project=config.wandb_project,
        name=f"{config.model}",
        config={
            "learning_rate": lr,
            "batch_size": config.batch_size,
            "max_epoch": max_epoch,
            "optimizer": optimizer_choice,
            "scheduler": "ExponentialLR",
            "weight_decay": weight_decay,
            "dataset": config.dataset
        },
    )

    logging.info("%d iterations per epoch", len(dl_train))
    logging.info("%d val iterations per epoch", len(dl_val))
    print(f"{len(dl_train)} iterations per epoch")
    print(f"{len(dl_val)} val iterations per epoch")

    total_losses = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0

    print("\n")
    print("STARTING TRAINING of ", model_name)
    print("\n")

    for epoch in tqdm(range(max_epoch), desc=f"Training {model_name}", ncols=70):
        # Training
        model.train()
        total_loss = []

        for batch_idx, (data, target) in enumerate(dl_train):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            total_loss.append(loss.item())

        epoch_loss = np.mean(total_loss)
        total_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_loss = []
        correct = 0

        with torch.no_grad():
            for data, target in dl_val:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss.append(F.cross_entropy(output, target).item())
                preds = output.argmax(dim=1)
                correct += preds.eq(target).sum().item()

        val_epoch_loss = np.mean(val_loss)
        val_losses.append(val_epoch_loss)
        val_acc = correct / len(dl_val.dataset)
        val_accuracies.append(val_acc)

        logging.info(
            f"Epoch {epoch+1}/{max_epoch} - "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_epoch_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        # Log su WandB
        wandb.log({
            "train/loss": epoch_loss,
            "val/loss": val_epoch_loss,
            "val/accuracy": val_acc,
            "train/lr": scheduler.get_last_lr()[0],
        }, step=epoch)

        # Salva best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)

        scheduler.step()

    wandb.finish()

    print("\n")
    print("FINISHED TRAINING of ", model_name)
    print("\n")

    return total_losses, val_losses, val_accuracies, best_val_acc