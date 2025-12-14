"""
Data Loading Utilities.

Provides functions for loading CIFAR-10 and creating fixed batches
for consistent loss evaluation across different parameter configurations.

Authors: Apurva Mishra and Ayush Tibrewal
Course: CPSC 6440 - Geometric and Topological Methods in Machine Learning
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional


def load_cifar10(
    batch_size: int = 128,
    subset_size: Optional[int] = None,
    num_workers: int = 2,
    data_dir: str = "./data"
) -> Tuple[DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset with standard normalization.
    
    Args:
        batch_size: Batch size for dataloaders.
        subset_size: If provided, use only this many training samples.
        num_workers: Number of data loading workers.
        data_dir: Directory to store/load data.
    
    Returns:
        Tuple of (train_loader, test_loader).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )
    
    if subset_size is not None and subset_size < len(trainset):
        indices = np.random.choice(len(trainset), subset_size, replace=False)
        trainset = Subset(trainset, indices)
    
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return trainloader, testloader


def get_fixed_batch(
    trainloader: DataLoader,
    batch_size: int = 1024,
    device: torch.device = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a fixed batch of data for consistent loss evaluation.
    
    Using a fixed batch reduces variance in loss computation when
    comparing different parameter configurations.
    
    Args:
        trainloader: DataLoader to sample from.
        batch_size: Number of samples in the fixed batch.
        device: Device to place tensors on.
    
    Returns:
        Tuple of (images, labels) tensors.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_images = []
    all_labels = []
    
    for images, labels in trainloader:
        all_images.append(images)
        all_labels.append(labels)
        if sum(img.shape[0] for img in all_images) >= batch_size:
            break
    
    images = torch.cat(all_images, dim=0)[:batch_size]
    labels = torch.cat(all_labels, dim=0)[:batch_size]
    
    return images.to(device), labels.to(device)


def compute_loss(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Compute cross-entropy loss for given parameters.
    
    Args:
        model: Neural network model.
        images: Input images.
        labels: Ground truth labels.
    
    Returns:
        Loss value as float.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
    return loss.item()


def compute_loss_and_accuracy(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = None
) -> Tuple[float, float]:
    """
    Compute average loss and accuracy over a dataloader.
    
    Args:
        model: Neural network model.
        dataloader: DataLoader to evaluate on.
        device: Device for computation.
    
    Returns:
        Tuple of (average_loss, accuracy).
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels, reduction='sum')
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / total, correct / total


def compute_loss_and_gradient(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    """
    Compute loss and gradient with respect to parameters.
    
    Args:
        model: Neural network model (must have get_flat_grads method).
        images: Input images.
        labels: Ground truth labels.
    
    Returns:
        Tuple of (loss_value, flat_gradient_tensor).
    """
    model.train()
    model.zero_grad()
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    
    grads = model.get_flat_grads()
    return loss.item(), grads.detach().clone()


if __name__ == "__main__":
    # test data loading
    print("Testing data loading...")
    
    trainloader, testloader = load_cifar10(batch_size=128, subset_size=5000)
    print(f"Train batches: {len(trainloader)}")
    print(f"Test batches: {len(testloader)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = get_fixed_batch(trainloader, batch_size=1024, device=device)
    print(f"Fixed batch - Images: {images.shape}, Labels: {labels.shape}")
    
    # test with a network
    from networks import SmallCNN
    model = SmallCNN().to(device)
    
    loss = compute_loss(model, images, labels)
    print(f"Random init loss: {loss:.4f} (expected ~{np.log(10):.4f})")
    
    test_loss, test_acc = compute_loss_and_accuracy(model, testloader, device)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
