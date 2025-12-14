"""
Loss Predictor Network (Stage 1).

Neural network that learns to predict the loss of a target network
given its flattened parameter vector. This enables efficient sampling
of the loss landscape without running forward passes.

Authors: Apurva Mishra and Ayush Tibrewal
Course: CPSC 6440 - Geometric and Topological Methods in Machine Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm and GELU activation."""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super(ResidualBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # skip connection with projection if dimensions differ
        if in_dim != out_dim:
            self.skip = nn.Linear(in_dim, out_dim)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        x = self.norm1(x)
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        
        return x + residual


class LossPredictor(nn.Module):
    """
    Neural network that predicts loss from flattened parameters.
    
    Architecture:
        - Input LayerNorm for stability
        - Initial projection to hidden dimension
        - Stack of residual blocks with decreasing dimensions
        - Output head predicting scalar loss
    
    The network uses GELU activations and LayerNorm throughout
    for stable training on high-dimensional inputs.
    """
    
    def __init__(
        self,
        param_dim: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize the loss predictor.
        
        Args:
            param_dim: Dimensionality of input parameter vectors.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability.
        """
        super(LossPredictor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [
                min(2048, param_dim * 2),
                1024, 512, 256, 128
            ]
        
        self.param_dim = param_dim
        self.hidden_dims = hidden_dims
        
        # input normalization and projection
        self.input_norm = nn.LayerNorm(param_dim)
        self.input_proj = nn.Linear(param_dim, hidden_dims[0])
        
        # stack of residual blocks
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            block = ResidualBlock(hidden_dims[i], hidden_dims[i + 1], dropout)
            self.blocks.append(block)
        
        # output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Flattened parameter vectors [batch_size, param_dim].
        
        Returns:
            Predicted loss values [batch_size].
        """
        x = self.input_norm(x)
        x = F.gelu(self.input_proj(x))
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_head(x).squeeze(-1)


def train_loss_predictor(
    model: LossPredictor,
    train_data: Dict[str, torch.Tensor],
    val_split: float = 0.1,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 10,
    patience: int = 30,
    device: torch.device = None,
    verbose: bool = True
) -> Dict:
    """
    Train the loss predictor on generated data.
    
    Args:
        model: LossPredictor model.
        train_data: Dictionary with 'params' and 'losses' tensors.
        val_split: Fraction of data for validation.
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        weight_decay: Weight decay for AdamW.
        warmup_epochs: Number of warmup epochs.
        patience: Early stopping patience.
        device: Computation device.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with training history and normalization parameters.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    params = train_data['params']
    losses = train_data['losses']
    
    # normalize inputs
    param_mean = params.mean(dim=0)
    param_std = params.std(dim=0).clamp(min=1e-8)
    params_normalized = (params - param_mean) / param_std
    
    # normalize outputs and clip extreme values
    losses_clipped = losses.clamp(min=0.1, max=100)
    loss_mean = losses_clipped.mean()
    loss_std = losses_clipped.std().clamp(min=1e-8)
    losses_normalized = (losses_clipped - loss_mean) / loss_std
    
    # train/val split
    n_samples = len(params)
    n_val = int(n_samples * val_split)
    indices = torch.randperm(n_samples)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    train_params = params_normalized[train_idx]
    train_losses = losses_normalized[train_idx]
    val_params = params_normalized[val_idx].to(device)
    val_losses = losses_normalized[val_idx].to(device)
    val_losses_original = losses_clipped[val_idx].to(device)
    
    # create dataloader
    train_dataset = TensorDataset(train_params, train_losses)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    
    # setup training
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # learning rate schedule: warmup + cosine annealing
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_corr': [],
        'lr': []
    }
    
    best_val_mae = float('inf')
    best_state = None
    patience_counter = 0
    
    if verbose:
        print(f"\nTraining loss predictor...")
        print(f"  Training samples: {len(train_idx):,}")
        print(f"  Validation samples: {len(val_idx):,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs: {epochs}")
    
    for epoch in range(epochs):
        # training phase
        model.train()
        train_losses_epoch = []
        
        for batch_params, batch_losses in train_loader:
            batch_params = batch_params.to(device)
            batch_losses = batch_losses.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_params)
            loss = F.mse_loss(pred, batch_losses)
            
            # small L1 regularization on predictions
            loss = loss + 0.001 * pred.abs().mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses_epoch.append(loss.item())
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # validation phase
        model.eval()
        with torch.no_grad():
            val_pred_norm = model(val_params)
            val_loss = F.mse_loss(val_pred_norm, val_losses).item()
            
            # convert to original scale
            val_pred = val_pred_norm * loss_std + loss_mean
            val_mae = F.l1_loss(val_pred, val_losses_original).item()
            
            # correlation
            val_pred_np = val_pred.cpu().numpy()
            val_true_np = val_losses_original.cpu().numpy()
            
            # handle potential NaN in correlation
            valid = np.isfinite(val_pred_np) & np.isfinite(val_true_np)
            if valid.sum() > 10:
                val_corr = np.corrcoef(val_pred_np[valid], val_true_np[valid])[0, 1]
            else:
                val_corr = 0.0
        
        # record history
        history['train_loss'].append(np.mean(train_losses_epoch))
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_corr'].append(val_corr if not np.isnan(val_corr) else 0.0)
        history['lr'].append(current_lr)
        
        # early stopping check
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train: {history['train_loss'][-1]:.4f}, "
                  f"Val: {val_loss:.4f}, "
                  f"MAE: {val_mae:.4f}, "
                  f"Corr: {val_corr:.4f}")
        
        # early stopping
        if patience_counter >= patience and epoch > warmup_epochs + 20:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # store normalization parameters
    history['param_mean'] = param_mean
    history['param_std'] = param_std
    history['loss_mean'] = loss_mean.item()
    history['loss_std'] = loss_std.item()
    history['best_state'] = best_state
    
    if verbose:
        print(f"\nTraining complete!")
        print(f"  Best validation MAE: {best_val_mae:.4f}")
        print(f"  Best validation correlation: {max(history['val_corr']):.4f}")
    
    return history


def save_checkpoint(
    model: LossPredictor,
    history: Dict,
    path: str
) -> None:
    """Save model checkpoint with training history."""
    checkpoint = {
        'model_state': model.state_dict(),
        'param_dim': model.param_dim,
        'hidden_dims': model.hidden_dims,
        'history': history
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device: torch.device = None
) -> Tuple[LossPredictor, Dict]:
    """Load model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    model = LossPredictor(
        param_dim=checkpoint['param_dim'],
        hidden_dims=checkpoint['hidden_dims']
    )
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    
    return model, checkpoint['history']


if __name__ == "__main__":
    # test the loss predictor
    print("Testing LossPredictor...")
    
    # create model
    param_dim = 34522  # SmallCNN
    model = LossPredictor(param_dim)
    
    print(f"Parameter dimension: {param_dim:,}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # test forward pass
    x = torch.randn(32, param_dim)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # test with dummy data
    dummy_data = {
        'params': torch.randn(1000, param_dim),
        'losses': torch.rand(1000) * 5 + 2
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    history = train_loss_predictor(
        model, dummy_data,
        epochs=5, batch_size=64,
        device=device, verbose=True
    )
    
    print("\nTraining history keys:", list(history.keys()))
