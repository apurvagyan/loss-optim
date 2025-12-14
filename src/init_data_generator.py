"""
Initialization Data Generator.

Generates large datasets of (parameters, loss) pairs for training
the loss predictor network. Optimized for efficient generation of
hundreds of thousands of samples.

Authors: Apurva Mishra and Ayush Tibrewal
Course: CPSC 6440 - Geometric and Topological Methods in Machine Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
import os


class InitializationDataGenerator:
    """
    Generates training data for the loss predictor network.
    
    Produces (parameter_vector, loss) pairs by:
    1. Sampling initializations from various methods (Xavier, Kaiming, etc.)
    2. Adding perturbations to explore the local landscape
    3. Creating interpolations between good initializations
    
    Attributes:
        model_class: The target network class.
        images: Fixed batch of images for loss computation.
        labels: Corresponding labels.
        param_dim: Dimensionality of the parameter space.
        device: Computation device.
    """
    
    def __init__(
        self,
        model_class: type,
        images: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device = None
    ):
        """
        Initialize the data generator.
        
        Args:
            model_class: Class of the target network.
            images: Fixed batch of images for loss computation.
            labels: Corresponding labels.
            device: Device for computation.
        """
        self.model_class = model_class
        self.images = images
        self.labels = labels
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # create reference model and extract architecture info
        self.ref_model = model_class().to(self.device)
        self.param_dim = self.ref_model.count_parameters()
        self.param_shapes = [p.shape for p in self.ref_model.parameters()]
        self.param_sizes = [p.numel() for p in self.ref_model.parameters()]
        
    def _generate_init_params(self, method: str, scale: float = 1.0) -> torch.Tensor:
        """
        Generate initialization parameters as a flat tensor.
        
        Args:
            method: Initialization method ('xavier', 'kaiming', 'orthogonal', 'normal', 'uniform').
            scale: Scaling factor for the initialization.
        
        Returns:
            Flat parameter tensor.
        """
        params = []
        
        for shape, size in zip(self.param_shapes, self.param_sizes):
            if len(shape) >= 2:
                # weight matrix
                fan_in = shape[1] if len(shape) == 2 else np.prod(shape[1:])
                fan_out = shape[0]
                
                if method == 'xavier':
                    std = scale * np.sqrt(2.0 / (fan_in + fan_out))
                    p = torch.randn(size) * std
                elif method == 'kaiming':
                    std = scale * np.sqrt(2.0 / fan_in)
                    p = torch.randn(size) * std
                elif method == 'orthogonal':
                    std = scale * np.sqrt(1.0 / fan_in)
                    p = torch.randn(size) * std
                elif method == 'normal':
                    p = torch.randn(size) * scale * 0.1
                elif method == 'uniform':
                    bound = scale * np.sqrt(3.0 / fan_in)
                    p = torch.empty(size).uniform_(-bound, bound)
                else:
                    p = torch.randn(size) * 0.1
            else:
                # bias - initialize to zeros
                p = torch.zeros(size)
            
            params.append(p)
        
        return torch.cat(params)
    
    def _compute_loss_batch(self, flat_params_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for a batch of parameter configurations.
        
        Args:
            flat_params_batch: Batch of flat parameter vectors [B, param_dim].
        
        Returns:
            Tensor of loss values [B].
        """
        batch_size = flat_params_batch.shape[0]
        losses = []
        
        for i in range(batch_size):
            self.ref_model.set_flat_params(flat_params_batch[i].to(self.device))
            self.ref_model.eval()
            with torch.no_grad():
                outputs = self.ref_model(self.images)
                loss = F.cross_entropy(outputs, self.labels)
            losses.append(loss.item())
        
        return torch.tensor(losses, dtype=torch.float32)
    
    def generate(
        self,
        n_samples: int = 100000,
        perturbation_std: float = 0.2,
        save_interval: int = 20000,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate dataset of (parameters, loss) pairs.
        
        Args:
            n_samples: Total number of samples to generate.
            perturbation_std: Standard deviation of perturbations.
            save_interval: Save checkpoint every N samples.
            save_path: Path prefix for saving checkpoints.
            verbose: Whether to print progress.
        
        Returns:
            Dictionary with 'params', 'losses', 'init_types', 'param_dim'.
        """
        params_list = []
        losses_list = []
        init_types = []
        
        methods = ['xavier', 'kaiming', 'orthogonal', 'normal', 'uniform']
        method_weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        
        # allocate samples: 90% base initializations, 10% interpolations
        interpolation_fraction = 0.10
        n_base_samples = int(n_samples * (1 - interpolation_fraction))
        n_interp_samples = n_samples - n_base_samples
        
        if verbose:
            print(f"\nGenerating {n_samples:,} initialization samples...")
            print(f"  Parameter dimension: {self.param_dim:,}")
            print(f"  Base samples: {n_base_samples:,}")
            print(f"  Interpolation samples: {n_interp_samples:,}")
            start_time = time.time()
        
        # generate base samples
        if verbose:
            print("\nGenerating base initializations...")
        
        batch_size = 100
        iterator = tqdm(range(0, n_base_samples, batch_size), disable=not verbose)
        
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, n_base_samples)
            batch_params = []
            batch_types = []
            
            for _ in range(batch_end - batch_start):
                # randomly choose initialization method
                method = np.random.choice(methods, p=method_weights)
                scale = np.random.uniform(0.5, 1.5)
                
                # generate base initialization
                flat_params = self._generate_init_params(method, scale)
                
                # add perturbation with 90% probability
                if np.random.rand() > 0.1:
                    perturb_scale = perturbation_std * (1 + 0.2 * np.random.randn())
                    perturbation = torch.randn_like(flat_params) * perturb_scale
                    flat_params = flat_params + perturbation
                
                batch_params.append(flat_params)
                batch_types.append(method)
            
            # compute losses for this batch
            batch_params_tensor = torch.stack(batch_params)
            batch_losses = self._compute_loss_batch(batch_params_tensor)
            
            params_list.extend(batch_params)
            losses_list.extend(batch_losses.tolist())
            init_types.extend(batch_types)
            
            # save checkpoint periodically
            if save_path and len(params_list) % save_interval < batch_size:
                elapsed = time.time() - start_time
                rate = len(params_list) / elapsed
                if verbose:
                    print(f"\n  Checkpoint: {len(params_list):,} samples, {rate:.1f} samples/sec")
                
                checkpoint = {
                    'params': torch.stack(params_list),
                    'losses': torch.tensor(losses_list, dtype=torch.float32),
                    'init_types': init_types,
                    'n_samples': len(params_list),
                    'param_dim': self.param_dim
                }
                torch.save(checkpoint, f"{save_path}_checkpoint.pt")
        
        # generate interpolation samples from good initializations
        if verbose:
            print("\nGenerating interpolation samples...")
        
        # find good initializations (lower loss)
        losses_tensor = torch.tensor(losses_list, dtype=torch.float32)
        threshold = losses_tensor.quantile(0.3)
        good_mask = losses_tensor < threshold
        good_indices = torch.where(good_mask)[0].tolist()
        
        if len(good_indices) < 10:
            good_indices = list(range(min(100, len(params_list))))
        
        # cap at 500 for memory efficiency
        good_params = [params_list[i] for i in good_indices[:500]]
        
        iterator = tqdm(range(0, n_interp_samples, batch_size), disable=not verbose)
        
        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, n_interp_samples)
            batch_params = []
            
            for _ in range(batch_end - batch_start):
                # random convex combination of 2-4 good initializations
                n_combine = np.random.choice([2, 3, 4])
                indices = np.random.choice(len(good_params), n_combine, replace=False)
                weights = np.random.dirichlet(np.ones(n_combine))
                
                flat_params = sum(w * good_params[idx] for w, idx in zip(weights, indices))
                
                # add smaller perturbation
                perturbation = torch.randn_like(flat_params) * perturbation_std * 0.5
                flat_params = flat_params + perturbation
                
                batch_params.append(flat_params)
            
            # compute losses
            batch_params_tensor = torch.stack(batch_params)
            batch_losses = self._compute_loss_batch(batch_params_tensor)
            
            params_list.extend(batch_params)
            losses_list.extend(batch_losses.tolist())
            init_types.extend(['interpolation'] * len(batch_params))
        
        # create final result
        params_tensor = torch.stack(params_list)
        losses_tensor = torch.tensor(losses_list, dtype=torch.float32)
        
        result = {
            'params': params_tensor,
            'losses': losses_tensor,
            'init_types': init_types,
            'param_dim': self.param_dim
        }
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"\nDataset generated in {elapsed/60:.1f} minutes")
            print(f"  Total samples: {len(losses_list):,}")
            print(f"  Generation rate: {len(losses_list)/elapsed:.1f} samples/sec")
            print(f"  Loss range: [{losses_tensor.min():.4f}, {losses_tensor.max():.4f}]")
            print(f"  Loss mean: {losses_tensor.mean():.4f} ± {losses_tensor.std():.4f}")
            
            print("\n  By initialization type:")
            for method in methods + ['interpolation']:
                mask = torch.tensor([t == method for t in init_types])
                if mask.sum() > 0:
                    method_losses = losses_tensor[mask]
                    print(f"    {method}: n={mask.sum():,}, "
                          f"loss={method_losses.mean():.4f} ± {method_losses.std():.4f}")
        
        # save final dataset
        if save_path:
            torch.save(result, f"{save_path}.pt")
            if verbose:
                print(f"\nDataset saved to {save_path}.pt")
        
        return result


def load_init_data(path: str) -> Dict[str, torch.Tensor]:
    """
    Load previously generated initialization data.
    
    Args:
        path: Path to the saved data file.
    
    Returns:
        Dictionary with params, losses, init_types, param_dim.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    # test the generator
    from networks import SmallCNN
    from data_utils import load_cifar10, get_fixed_batch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # load data
    trainloader, _ = load_cifar10(batch_size=128, subset_size=5000)
    images, labels = get_fixed_batch(trainloader, batch_size=1024, device=device)
    
    # create generator
    generator = InitializationDataGenerator(SmallCNN, images, labels, device)
    
    # generate small test dataset
    data = generator.generate(
        n_samples=1000,
        perturbation_std=0.2,
        save_path="../data/test_init_data",
        verbose=True
    )
    
    print(f"\nGenerated data shapes:")
    print(f"  Params: {data['params'].shape}")
    print(f"  Losses: {data['losses'].shape}")
