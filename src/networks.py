"""
Target Network Architectures for Loss Landscape Analysis.

These are the neural networks whose loss landscapes we analyze.
Each network provides methods to flatten/unflatten parameters for
the loss predictor to consume.

Authors: Apurva Mishra and Ayush Tibrewal
Course: CPSC 6440 - Geometric and Topological Methods in Machine Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class SmallCNN(nn.Module):
    """
    Small CNN for CIFAR-10 with approximately 34K parameters.
    
    Architecture:
        - Conv1: 3 -> 8 channels, 3x3 kernel (224 params)
        - Conv2: 8 -> 16 channels, 3x3 kernel (1,168 params)
        - FC1: 1024 -> 32 (32,800 params)
        - FC2: 32 -> 10 (330 params)
        - Total: ~34,522 parameters
    
    This size provides a good balance between having a realistic
    loss landscape and being tractable for our analysis.
    """
    
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_flat_params(self) -> torch.Tensor:
        """Flatten all parameters into a single vector."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])
    
    def set_flat_params(self, flat_params: torch.Tensor) -> None:
        """Set parameters from a flat vector."""
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(flat_params[idx:idx + size].view(p.shape))
            idx += size
    
    def get_flat_grads(self) -> torch.Tensor:
        """Flatten all gradients into a single vector."""
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)
    
    def count_parameters(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())


class TinyCNN(nn.Module):
    """
    Tiny CNN for CIFAR-10 with approximately 8.7K parameters.
    
    Uses aggressive pooling to reduce parameter count while
    maintaining some convolutional structure.
    """
    
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.parameters()])
    
    def set_flat_params(self, flat_params: torch.Tensor) -> None:
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(flat_params[idx:idx + size].view(p.shape))
            idx += size
    
    def get_flat_grads(self) -> torch.Tensor:
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MicroCNN(nn.Module):
    """
    Micro CNN for CIFAR-10 with approximately 2.7K parameters.
    
    The smallest network option, useful for quick experiments
    and validating the methodology.
    """
    
    def __init__(self):
        super(MicroCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in self.parameters()])
    
    def set_flat_params(self, flat_params: torch.Tensor) -> None:
        idx = 0
        for p in self.parameters():
            size = p.numel()
            p.data.copy_(flat_params[idx:idx + size].view(p.shape))
            idx += size
    
    def get_flat_grads(self) -> torch.Tensor:
        grads = []
        for p in self.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
        return torch.cat(grads)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def get_target_network(name: str) -> nn.Module:
    """Factory function to get target network by name."""
    networks = {
        "SmallCNN": SmallCNN,
        "TinyCNN": TinyCNN,
        "MicroCNN": MicroCNN,
    }
    if name not in networks:
        raise ValueError(f"Unknown network: {name}. Available: {list(networks.keys())}")
    return networks[name]()


def get_param_shapes(model: nn.Module) -> List[torch.Size]:
    """Get shapes of all parameters for reconstruction."""
    return [p.shape for p in model.parameters()]


def print_network_summary(model: nn.Module) -> None:
    """Print a summary of the network architecture."""
    print(f"\n{'='*50}")
    print(f"Network: {model.__class__.__name__}")
    print(f"{'='*50}")
    total = 0
    for name, param in model.named_parameters():
        print(f"  {name}: {list(param.shape)} = {param.numel():,}")
        total += param.numel()
    print(f"{'='*50}")
    print(f"  Total parameters: {total:,}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # test all networks
    for name in ["SmallCNN", "TinyCNN", "MicroCNN"]:
        model = get_target_network(name)
        print_network_summary(model)
        
        # test forward pass
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        
        # test param flattening
        flat = model.get_flat_params()
        print(f"  Flat params shape: {flat.shape}")
        
        # test param setting
        model.set_flat_params(flat + 0.01)
        print(f"  Param setting: OK\n")
