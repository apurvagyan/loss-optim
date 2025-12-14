"""
Initialization Quality Predictor (Stage 2).

Predicts the quality of a neural network initialization based on
topological features of its local loss landscape. Quality is measured
by test loss/accuracy after training.

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


class QualityPredictor(nn.Module):
    """
    Predicts initialization quality from topological features.
    
    Architecture:
        - Stack of fully connected layers with BatchNorm
        - ReLU activations and dropout for regularization
        - Single scalar output (predicted test loss)
    """
    
    def __init__(
        self,
        n_features: int,
        hidden_dims: List[int] = None,
        dropout: float = 0.2
    ):
        """
        Initialize the quality predictor.
        
        Args:
            n_features: Number of input topological features.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout probability.
        """
        super(QualityPredictor, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        
        layers = []
        prev_dim = n_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Topological features [batch_size, n_features].
        
        Returns:
            Predicted quality [batch_size].
        """
        x = self.features(x)
        return self.output(x).squeeze(-1)


def train_quality_predictor(
    model: QualityPredictor,
    features: np.ndarray,
    labels: np.ndarray,
    val_split: float = 0.2,
    epochs: int = 150,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = None,
    verbose: bool = True
) -> Dict:
    """
    Train the quality predictor.
    
    Args:
        model: QualityPredictor model.
        features: Topological features [n_samples, n_features].
        labels: Quality labels (test loss) [n_samples].
        val_split: Fraction of data for validation.
        epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Computation device.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with training results and scaler.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # handle NaN/Inf values
    valid_mask = np.isfinite(features).all(axis=1) & np.isfinite(labels)
    features = features[valid_mask]
    labels = labels[valid_mask]
    
    if verbose:
        print(f"Valid samples after filtering: {len(labels)}")
    
    # normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # normalize labels
    label_mean = labels.mean()
    label_std = labels.std() + 1e-8
    labels_scaled = (labels - label_mean) / label_std
    
    # train/val split
    n_val = int(len(features) * val_split)
    indices = np.random.permutation(len(features))
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    train_features = torch.tensor(features_scaled[train_idx], dtype=torch.float32).to(device)
    train_labels = torch.tensor(labels_scaled[train_idx], dtype=torch.float32).to(device)
    val_features = torch.tensor(features_scaled[val_idx], dtype=torch.float32).to(device)
    val_labels = torch.tensor(labels_scaled[val_idx], dtype=torch.float32).to(device)
    val_labels_original = torch.tensor(labels[val_idx], dtype=torch.float32)
    
    # dataloader
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # training setup
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_corr': []
    }
    
    best_val_mae = float('inf')
    best_state = None
    
    if verbose:
        print(f"\nTraining quality predictor...")
        print(f"  Train samples: {len(train_idx)}")
        print(f"  Val samples: {len(val_idx)}")
    
    for epoch in range(epochs):
        # training
        model.train()
        train_losses = []
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            pred = model(batch_features)
            loss = F.mse_loss(pred, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # validation
        model.eval()
        with torch.no_grad():
            val_pred_scaled = model(val_features)
            val_loss = F.mse_loss(val_pred_scaled, val_labels).item()
            
            # convert to original scale
            val_pred = val_pred_scaled.cpu() * label_std + label_mean
            val_mae = F.l1_loss(val_pred, val_labels_original).item()
            
            # correlation
            val_pred_np = val_pred.numpy()
            val_true_np = val_labels_original.numpy()
            valid = np.isfinite(val_pred_np) & np.isfinite(val_true_np)
            if valid.sum() > 5:
                val_corr = np.corrcoef(val_pred_np[valid], val_true_np[valid])[0, 1]
            else:
                val_corr = 0.0
        
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_corr'].append(val_corr if not np.isnan(val_corr) else 0.0)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train: {history['train_loss'][-1]:.4f}, "
                  f"Val MAE: {val_mae:.4f}, "
                  f"Corr: {val_corr:.4f}")
    
    # restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # final evaluation
    model.eval()
    with torch.no_grad():
        final_pred_scaled = model(val_features)
        final_pred = final_pred_scaled.cpu() * label_std + label_mean
    
    final_pred_np = final_pred.numpy()
    final_true_np = val_labels_original.numpy()
    
    final_mae = mean_absolute_error(final_true_np, final_pred_np)
    final_corr = np.corrcoef(final_pred_np, final_true_np)[0, 1]
    final_r2 = r2_score(final_true_np, final_pred_np)
    
    if verbose:
        print(f"\nFinal Results:")
        print(f"  MAE: {final_mae:.4f}")
        print(f"  Correlation: {final_corr:.4f}")
        print(f"  R²: {final_r2:.4f}")
    
    return {
        'model': model,
        'scaler': scaler,
        'label_mean': label_mean,
        'label_std': label_std,
        'history': history,
        'val_predictions': final_pred_np,
        'val_labels': final_true_np,
        'metrics': {
            'mae': final_mae,
            'correlation': final_corr,
            'r2': final_r2
        }
    }


def evaluate_initialization(
    model_class: type,
    init_params: torch.Tensor,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    epochs: int = 15,
    lr: float = 0.01,
    momentum: float = 0.9,
    device: torch.device = None,
    verbose: bool = False
) -> Dict:
    """
    Train a model from given initialization and measure quality.
    
    This provides ground truth labels for the quality predictor.
    
    Args:
        model_class: Class of the target network.
        init_params: Initial parameters.
        trainloader: Training data loader.
        testloader: Test data loader.
        epochs: Training epochs.
        lr: Learning rate.
        momentum: SGD momentum.
        device: Computation device.
        verbose: Whether to print progress.
    
    Returns:
        Dictionary with training history and quality metrics.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model_class().to(device)
    model.set_flat_params(init_params.to(device))
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in range(epochs):
        # training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        history['train_loss'].append(running_loss / total)
        history['train_acc'].append(correct / total)
        
        # evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        history['test_loss'].append(test_loss / total)
        history['test_acc'].append(correct / total)
        
        scheduler.step()
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {history['train_loss'][-1]:.4f}, "
                  f"Test Acc: {history['test_acc'][-1]:.4f}")
    
    # compute quality metrics
    history['final_test_loss'] = history['test_loss'][-1]
    history['final_test_acc'] = history['test_acc'][-1]
    history['best_test_loss'] = min(history['test_loss'])
    history['best_test_acc'] = max(history['test_acc'])
    history['auc_test_loss'] = np.trapz(history['test_loss'])
    
    return history


if __name__ == "__main__":
    # test quality predictor
    print("Testing QualityPredictor...")
    
    # synthetic data
    n_samples = 200
    n_features = 27
    
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.rand(n_samples).astype(np.float32) * 2 + 1.5
    
    model = QualityPredictor(n_features)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = train_quality_predictor(
        model, features, labels,
        epochs=50, device=device, verbose=True
    )
    
    print(f"\nHistory keys: {list(result['history'].keys())}")
    print(f"Metrics: {result['metrics']}")
