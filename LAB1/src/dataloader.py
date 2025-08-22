import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import Subset, DataLoader

def get_mnist_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    ds_train_full = MNIST(root='./data', train=True, download=True, transform=transform)
    ds_test = MNIST(root='./data', train=False, download=True, transform=transform)

    indices = np.random.permutation(len(ds_train_full))
    ds_val = Subset(ds_train_full, indices[:config.val_size])
    ds_train = Subset(ds_train_full, indices[config.val_size:])

    train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    print("MNIST DataLoaders pronti.")
    print(f"  - Training samples: {len(ds_train)}, Validation samples: {len(ds_val)}, Test samples: {len(ds_test)}")
    
    return train_loader, val_loader, test_loader


def get_cifar10_loaders(config):
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    ds_train_full = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    ds_val_full = CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    ds_test = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    indices = np.random.permutation(len(ds_train_full))
    ds_val = Subset(ds_val_full, indices[:config.val_size])
    ds_train = Subset(ds_train_full, indices[config.val_size:])

    train_loader = DataLoader(ds_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(ds_val, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(ds_test, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    print("CIFAR-10 DataLoaders pronti.")
    print(f"  - Training samples: {len(ds_train)}, Validation samples: {len(ds_val)}, Test samples: {len(ds_test)}")

    return train_loader, val_loader, test_loader