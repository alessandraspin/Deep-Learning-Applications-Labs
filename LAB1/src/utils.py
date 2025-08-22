import matplotlib.pyplot as plt
import os

def get_model_path(run_name: str) -> str:
    """
    Restituisce il path assoluto per salvare/caricare un modello.
    """
    model_dir = os.path.join(os.path.dirname(os.getcwd()), "models")
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"best_{run_name}.pth")


def plot_single_performance(train_losses, val_losses, val_accuracies, model_name, save_path=None):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'{model_name} - Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:  # Se è stato passato un percorso, salva il PDF
        plt.savefig(save_path, format='pdf')
    plt.show()


def plot_all_performances(model_data, save_path=None):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 6))

    # Subplot 1: Combined Losses
    plt.subplot(1, 2, 1)
    for model_name, data in model_data.items():
        epochs = range(1, len(data['train_losses']) + 1)
        plt.plot(epochs, data['train_losses'], label=f'{model_name} - Training Loss')
        plt.plot(epochs, data['val_losses'], linestyle='--', label=f'{model_name} - Validation Loss')
    
    plt.title('Training and Validation Losses for All Models', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)

    # Subplot 2: Combined Accuracies
    plt.subplot(1, 2, 2)
    for model_name, data in model_data.items():
        epochs = range(1, len(data['val_accuracies']) + 1)
        plt.plot(epochs, data['val_accuracies'], label=f'{model_name} - Validation Accuracy')
    
    plt.title('Validation Accuracies for All Models', fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right', fontsize=10)

    plt.tight_layout(pad=3.0)
    
    if save_path:  # Salva in PDF se specificato
        plt.savefig(save_path, format='pdf')
    plt.show()
