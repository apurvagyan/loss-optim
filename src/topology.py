"""
Topological Feature Extraction (Stage 2).

Samples local loss landscapes and computes persistent homology features
including persistence diagrams, Betti curves, and derived statistics.

Authors: Apurva Mishra and Ayush Tibrewal
Course: CPSC 6440 - Geometric and Topological Methods in Machine Learning
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from ripser import ripser
from sklearn.decomposition import PCA


class LossLandscapeSampler:
    """
    Samples the local loss landscape around an initialization using
    the trained loss predictor network.
    
    This allows us to densely sample the landscape without running
    expensive forward passes through the target network.
    """
    
    def __init__(
        self,
        loss_predictor: torch.nn.Module,
        param_mean: torch.Tensor,
        param_std: torch.Tensor,
        loss_mean: float,
        loss_std: float,
        device: torch.device = None
    ):
        """
        Initialize the sampler.
        
        Args:
            loss_predictor: Trained loss predictor network.
            param_mean: Mean of parameters (for normalization).
            param_std: Std of parameters (for normalization).
            loss_mean: Mean of losses (for denormalization).
            loss_std: Std of losses (for denormalization).
            device: Computation device.
        """
        self.predictor = loss_predictor
        self.param_mean = param_mean
        self.param_std = param_std
        self.loss_mean = loss_mean
        self.loss_std = loss_std
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor.eval()
    
    def predict_loss(self, params: torch.Tensor) -> torch.Tensor:
        """
        Predict loss for given parameters.
        
        Args:
            params: Parameter vectors [batch_size, param_dim] or [param_dim].
        
        Returns:
            Predicted loss values.
        """
        with torch.no_grad():
            if params.dim() == 1:
                params = params.unsqueeze(0)
            
            # normalize
            params_norm = (params - self.param_mean.to(params.device)) / self.param_std.to(params.device)
            
            # predict
            pred_norm = self.predictor(params_norm.to(self.device))
            pred = pred_norm * self.loss_std + self.loss_mean
            
        return pred.cpu()
    
    def sample_local_landscape(
        self,
        center_params: torch.Tensor,
        n_samples: int = 2000,
        radius: float = 0.3,
        method: str = "gaussian"
    ) -> Dict:
        """
        Sample the loss landscape around a center point.
        
        Args:
            center_params: The initialization to analyze [param_dim].
            n_samples: Number of samples in the local region.
            radius: Scale of the local region.
            method: Sampling method ('gaussian' or 'uniform_sphere').
        
        Returns:
            Dictionary with sampled data and metadata.
        """
        param_dim = center_params.shape[0]
        center_params = center_params.clone()
        
        if method == "gaussian":
            # sample from gaussian centered at initialization
            perturbations = torch.randn(n_samples, param_dim) * radius
            
        elif method == "uniform_sphere":
            # sample uniformly from hypersphere
            directions = torch.randn(n_samples, param_dim)
            directions = directions / directions.norm(dim=1, keepdim=True)
            # uniform radii in ball
            radii = torch.rand(n_samples, 1) ** (1.0 / min(param_dim, 100)) * radius
            perturbations = directions * radii
            
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # generate sampled parameters
        sampled_params = center_params.unsqueeze(0) + perturbations
        
        # predict losses in batches
        batch_size = 512
        all_losses = []
        for i in range(0, n_samples, batch_size):
            batch = sampled_params[i:i + batch_size]
            losses = self.predict_loss(batch)
            all_losses.append(losses)
        
        losses = torch.cat(all_losses)
        center_loss = self.predict_loss(center_params).item()
        
        return {
            'center_params': center_params,
            'sampled_params': sampled_params,
            'perturbations': perturbations,
            'losses': losses,
            'center_loss': center_loss,
            'radius': radius,
            'n_samples': n_samples,
            'method': method
        }


class TopologicalFeatureExtractor:
    """
    Extracts topological features from sampled loss landscapes using
    persistent homology.
    
    Computes:
        - Vietoris-Rips persistence diagrams
        - Sublevel set persistence (Betti curves)
        - Loss-weighted persistence
        - Various derived statistics
    """
    
    def __init__(
        self,
        max_homology_dim: int = 1,
        max_points_for_rips: int = 500,
        projection_dim: int = 50,
        n_sublevel_thresholds: int = 50
    ):
        """
        Initialize the extractor.
        
        Args:
            max_homology_dim: Maximum homology dimension to compute.
            max_points_for_rips: Maximum points for Rips complex.
            projection_dim: Dimension to project to before persistence.
            n_sublevel_thresholds: Number of thresholds for Betti curves.
        """
        self.max_dim = max_homology_dim
        self.max_points = max_points_for_rips
        self.proj_dim = projection_dim
        self.n_thresholds = n_sublevel_thresholds
    
    def _subsample(self, points: np.ndarray, n: int) -> np.ndarray:
        """Subsample points if too many."""
        if len(points) <= n:
            return points
        indices = np.random.choice(len(points), n, replace=False)
        return points[indices]
    
    def _project(self, points: np.ndarray, dim: int) -> np.ndarray:
        """Project to lower dimension using random projection."""
        if points.shape[1] <= dim:
            return points
        
        # random projection (faster than PCA for high dim)
        proj = np.random.randn(points.shape[1], dim)
        proj = proj / np.linalg.norm(proj, axis=0, keepdims=True)
        return points @ proj
    
    def compute_rips_persistence(self, points: np.ndarray) -> Dict:
        """
        Compute Vietoris-Rips persistence on point cloud.
        
        Args:
            points: Point cloud [n_points, dim].
        
        Returns:
            Dictionary with persistence diagrams.
        """
        points = self._subsample(points, self.max_points)
        points = self._project(points, self.proj_dim)
        
        result = ripser(points, maxdim=self.max_dim, n_perm=min(200, len(points)))
        
        return {
            'diagrams': result['dgms'],
            'n_points': len(points)
        }
    
    def compute_sublevel_persistence(
        self,
        points: np.ndarray,
        losses: np.ndarray
    ) -> Dict:
        """
        Compute Betti curves of sublevel sets L_α = {θ : loss(θ) ≤ α}.
        
        This directly captures the topology of the loss landscape by
        tracking how connected components and cycles appear/disappear
        as we increase the loss threshold.
        
        Args:
            points: Point cloud (perturbations) [n_points, dim].
            losses: Loss values at each point [n_points].
        
        Returns:
            Dictionary with Betti curves and thresholds.
        """
        points = self._project(points, self.proj_dim)
        
        loss_min, loss_max = losses.min(), losses.max()
        thresholds = np.linspace(loss_min, loss_max, self.n_thresholds)
        
        betti_0_curve = []
        betti_1_curve = []
        
        for alpha in thresholds:
            mask = losses <= alpha
            n_points = mask.sum()
            
            if n_points < 4:
                betti_0_curve.append(max(1, n_points))
                betti_1_curve.append(0)
                continue
            
            sublevel_points = points[mask]
            sublevel_points = self._subsample(sublevel_points, 300)
            
            try:
                result = ripser(sublevel_points, maxdim=1, thresh=2.0)
                
                # count persistent features (filter noise)
                h0 = result['dgms'][0]
                h0_lifetimes = h0[:, 1] - h0[:, 0]
                h0_persistent = np.sum(h0_lifetimes > 0.1)
                betti_0_curve.append(max(1, h0_persistent))
                
                if len(result['dgms']) > 1 and len(result['dgms'][1]) > 0:
                    h1 = result['dgms'][1]
                    h1_lifetimes = h1[:, 1] - h1[:, 0]
                    h1_persistent = np.sum(h1_lifetimes > 0.1)
                    betti_1_curve.append(h1_persistent)
                else:
                    betti_1_curve.append(0)
                    
            except Exception:
                betti_0_curve.append(n_points)
                betti_1_curve.append(0)
        
        return {
            'thresholds': thresholds,
            'betti_0_curve': np.array(betti_0_curve),
            'betti_1_curve': np.array(betti_1_curve),
            'loss_range': (loss_min, loss_max)
        }
    
    def compute_weighted_persistence(
        self,
        points: np.ndarray,
        losses: np.ndarray
    ) -> Dict:
        """
        Compute persistence on loss-weighted point cloud.
        
        Points with similar loss are considered "closer" even if
        geometrically far, capturing loss landscape structure.
        
        Args:
            points: Point cloud [n_points, dim].
            losses: Loss values [n_points].
        
        Returns:
            Dictionary with persistence diagrams.
        """
        points = self._subsample(points, self.max_points)
        points = self._project(points, self.proj_dim)
        
        # subsample losses correspondingly
        if len(losses) > len(points):
            losses = losses[:len(points)]
        
        # normalize losses to [0, 1]
        loss_norm = (losses - losses.min()) / (losses.max() - losses.min() + 1e-8)
        
        # augment with loss as extra coordinate
        loss_weight = 3.0  # emphasize loss dimension
        augmented = np.column_stack([points, loss_norm.reshape(-1, 1) * loss_weight])
        
        result = ripser(augmented, maxdim=self.max_dim, n_perm=min(200, len(augmented)))
        
        return {
            'diagrams': result['dgms'],
            'n_points': len(points),
            'loss_weight': loss_weight
        }
    
    def extract_all_features(self, landscape_data: Dict) -> Dict:
        """
        Compute all topological features from landscape data.
        
        Args:
            landscape_data: Output from LossLandscapeSampler.sample_local_landscape().
        
        Returns:
            Dictionary with all topological computations.
        """
        perturbations = landscape_data['perturbations'].numpy()
        losses = landscape_data['losses'].numpy()
        
        # standard rips persistence
        rips_result = self.compute_rips_persistence(perturbations)
        
        # sublevel set persistence (Betti curves)
        sublevel_result = self.compute_sublevel_persistence(perturbations, losses)
        
        # loss-weighted persistence
        weighted_result = self.compute_weighted_persistence(perturbations, losses)
        
        return {
            'rips': rips_result,
            'sublevel': sublevel_result,
            'weighted': weighted_result,
            'center_loss': landscape_data['center_loss'],
            'loss_stats': {
                'mean': float(losses.mean()),
                'std': float(losses.std()),
                'min': float(losses.min()),
                'max': float(losses.max()),
                'median': float(np.median(losses))
            }
        }


def extract_numerical_features(topo_data: Dict) -> np.ndarray:
    """
    Extract numerical feature vector from topological data.
    
    Features are designed to capture:
        - Local landscape complexity (number of components, cycles)
        - Sharpness/smoothness (persistence statistics)
        - Basin structure (Betti curve properties)
        - Persistence entropy (topological complexity measure)
    
    Args:
        topo_data: Output from TopologicalFeatureExtractor.extract_all_features().
    
    Returns:
        Feature vector as numpy array.
    """
    features = []
    
    # === Rips persistence features ===
    rips_diagrams = topo_data['rips']['diagrams']
    
    # H0 features (connected components)
    h0 = rips_diagrams[0]
    if len(h0) > 0:
        lifetimes = h0[:, 1] - h0[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        
        features.extend([
            len(h0),                                               # n_components
            lifetimes.mean() if len(lifetimes) > 0 else 0,        # mean_lifetime_h0
            lifetimes.std() if len(lifetimes) > 1 else 0,         # std_lifetime_h0
            lifetimes.max() if len(lifetimes) > 0 else 0,         # max_lifetime_h0
            np.percentile(lifetimes, 75) if len(lifetimes) > 0 else 0  # p75_lifetime_h0
        ])
    else:
        features.extend([0, 0, 0, 0, 0])
    
    # H1 features (1-cycles/loops)
    if len(rips_diagrams) > 1 and len(rips_diagrams[1]) > 0:
        h1 = rips_diagrams[1]
        lifetimes = h1[:, 1] - h1[:, 0]
        
        features.extend([
            len(h1),                                    # n_cycles
            lifetimes.mean(),                           # mean_lifetime_h1
            lifetimes.std() if len(h1) > 1 else 0,     # std_lifetime_h1
            lifetimes.max()                            # max_lifetime_h1
        ])
    else:
        features.extend([0, 0, 0, 0])
    
    # === Sublevel set features (Betti curves) ===
    sublevel = topo_data['sublevel']
    b0 = sublevel['betti_0_curve']
    b1 = sublevel['betti_1_curve']
    
    # Betti-0 curve statistics
    features.extend([
        b0.mean(),                                # mean_betti0
        b0.std(),                                 # std_betti0
        b0.max(),                                 # max_betti0
        np.argmax(b0) / len(b0),                 # relative_pos_max_betti0
        (np.diff(b0) > 0).sum() / len(b0)        # fraction_increasing
    ])
    
    # Betti-1 curve statistics
    features.extend([
        b1.mean(),                                             # mean_betti1
        b1.std(),                                              # std_betti1
        b1.max(),                                              # max_betti1
        np.argmax(b1) / len(b1) if b1.max() > 0 else 0.5      # relative_pos_max_betti1
    ])
    
    # === Loss-weighted persistence features ===
    weighted_diagrams = topo_data['weighted']['diagrams']
    
    # Persistence entropy for each dimension
    for diagram in weighted_diagrams[:2]:
        if len(diagram) > 0:
            lifetimes = diagram[:, 1] - diagram[:, 0]
            lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
            
            if len(lifetimes) > 0:
                # persistence entropy
                probs = lifetimes / lifetimes.sum()
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                features.append(entropy)
                
                # total persistence
                features.append(lifetimes.sum())
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
    
    # === Loss statistics features ===
    loss_stats = topo_data['loss_stats']
    center_loss = topo_data['center_loss']
    
    features.extend([
        center_loss,                                           # center_loss
        loss_stats['mean'],                                    # mean_local_loss
        loss_stats['std'],                                     # std_local_loss
        loss_stats['mean'] - center_loss,                     # mean_vs_center
        loss_stats['min'] - center_loss,                      # basin_depth
        loss_stats['std'] / (loss_stats['mean'] + 1e-8)       # coeff_of_variation
    ])
    
    return np.array(features, dtype=np.float32)


def get_feature_names() -> List[str]:
    """Return names of all extracted features."""
    return [
        # rips h0
        'n_components', 'mean_lifetime_h0', 'std_lifetime_h0', 
        'max_lifetime_h0', 'p75_lifetime_h0',
        # rips h1
        'n_cycles', 'mean_lifetime_h1', 'std_lifetime_h1', 'max_lifetime_h1',
        # sublevel betti0
        'mean_betti0', 'std_betti0', 'max_betti0', 
        'rel_pos_max_betti0', 'frac_increasing',
        # sublevel betti1
        'mean_betti1', 'std_betti1', 'max_betti1', 'rel_pos_max_betti1',
        # weighted persistence
        'entropy_h0_weighted', 'total_pers_h0_weighted',
        'entropy_h1_weighted', 'total_pers_h1_weighted',
        # loss stats
        'center_loss', 'mean_local_loss', 'std_local_loss',
        'mean_vs_center', 'basin_depth', 'coeff_of_variation'
    ]


if __name__ == "__main__":
    # test with synthetic data
    print("Testing TopologicalFeatureExtractor...")
    
    # create fake landscape data
    n_samples = 1000
    param_dim = 100
    
    fake_landscape = {
        'center_params': torch.randn(param_dim),
        'sampled_params': torch.randn(n_samples, param_dim),
        'perturbations': torch.randn(n_samples, param_dim) * 0.3,
        'losses': torch.rand(n_samples) * 5 + 2,
        'center_loss': 2.5,
        'radius': 0.3,
        'n_samples': n_samples,
        'method': 'gaussian'
    }
    
    # extract features
    extractor = TopologicalFeatureExtractor()
    topo_data = extractor.extract_all_features(fake_landscape)
    
    print(f"\nTopological data keys: {list(topo_data.keys())}")
    print(f"Rips diagrams: H0 has {len(topo_data['rips']['diagrams'][0])} points")
    print(f"Betti-0 curve shape: {topo_data['sublevel']['betti_0_curve'].shape}")
    
    features = extract_numerical_features(topo_data)
    print(f"\nExtracted {len(features)} numerical features")
    print(f"Feature names: {get_feature_names()}")
