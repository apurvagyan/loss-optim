"""
Baseline Methods for Comparison.

Implements baseline approaches for predicting initialization quality
to compare against our topological method.

Authors: Apurva Mishra and Ayush Tibrewal
Course: CPSC 6440 - Geometric and Topological Methods in Machine Learning
"""

import numpy as np
from typing import Dict, List
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


def run_baseline_comparisons(
    topo_features: np.ndarray,
    quality_labels: np.ndarray,
    landscape_stats: List[Dict],
    test_split: float = 0.2,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Compare topological approach with baselines.
    
    Baselines:
        1. Random (mean prediction)
        2. Center loss only
        3. Local statistics only (no topology)
        4. Linear model on topological features
        5. Random Forest on topological features
        6. Gradient Boosting on topological features
    
    Args:
        topo_features: Topological features [n_samples, n_features].
        quality_labels: Quality labels (test loss) [n_samples].
        landscape_stats: List of dicts with center_loss, mean_loss, std_loss.
        test_split: Fraction for test set.
        verbose: Whether to print results.
    
    Returns:
        Dictionary mapping method names to metric dicts.
    """
    # extract center losses and local statistics
    center_losses = np.array([s['center_loss'] for s in landscape_stats])
    mean_losses = np.array([s['mean_loss'] for s in landscape_stats])
    std_losses = np.array([s['std_loss'] for s in landscape_stats])
    
    # filter invalid values
    valid_mask = (
        np.isfinite(topo_features).all(axis=1) &
        np.isfinite(quality_labels) &
        np.isfinite(center_losses)
    )
    
    topo_features = topo_features[valid_mask]
    quality_labels = quality_labels[valid_mask]
    center_losses = center_losses[valid_mask]
    mean_losses = mean_losses[valid_mask]
    std_losses = std_losses[valid_mask]
    
    # train/test split
    n_test = int(len(topo_features) * test_split)
    indices = np.random.permutation(len(topo_features))
    train_idx, test_idx = indices[n_test:], indices[:n_test]
    
    results = {}
    
    if verbose:
        print("\n" + "=" * 60)
        print("BASELINE COMPARISONS")
        print("=" * 60)
    
    # baseline 1: random (mean prediction)
    mean_pred = np.full(len(test_idx), quality_labels[train_idx].mean())
    results['random_mean'] = {
        'mae': mean_absolute_error(quality_labels[test_idx], mean_pred),
        'correlation': 0.0,
        'r2': 0.0
    }
    
    # baseline 2: center loss only
    lr_center = LinearRegression()
    lr_center.fit(center_losses[train_idx].reshape(-1, 1), quality_labels[train_idx])
    pred_center = lr_center.predict(center_losses[test_idx].reshape(-1, 1))
    corr_center = np.corrcoef(pred_center, quality_labels[test_idx])[0, 1]
    results['center_loss'] = {
        'mae': mean_absolute_error(quality_labels[test_idx], pred_center),
        'correlation': corr_center if not np.isnan(corr_center) else 0.0,
        'r2': r2_score(quality_labels[test_idx], pred_center)
    }
    
    # baseline 3: local statistics only
    local_stats = np.column_stack([center_losses, mean_losses, std_losses])
    lr_stats = LinearRegression()
    lr_stats.fit(local_stats[train_idx], quality_labels[train_idx])
    pred_stats = lr_stats.predict(local_stats[test_idx])
    corr_stats = np.corrcoef(pred_stats, quality_labels[test_idx])[0, 1]
    results['local_stats'] = {
        'mae': mean_absolute_error(quality_labels[test_idx], pred_stats),
        'correlation': corr_stats if not np.isnan(corr_stats) else 0.0,
        'r2': r2_score(quality_labels[test_idx], pred_stats)
    }
    
    # normalize features for remaining methods
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(topo_features)
    
    # baseline 4: linear (ridge) on topological features
    ridge = Ridge(alpha=1.0)
    ridge.fit(features_scaled[train_idx], quality_labels[train_idx])
    pred_ridge = ridge.predict(features_scaled[test_idx])
    corr_ridge = np.corrcoef(pred_ridge, quality_labels[test_idx])[0, 1]
    results['linear_topo'] = {
        'mae': mean_absolute_error(quality_labels[test_idx], pred_ridge),
        'correlation': corr_ridge if not np.isnan(corr_ridge) else 0.0,
        'r2': r2_score(quality_labels[test_idx], pred_ridge)
    }
    
    # baseline 5: random forest on topological features
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(features_scaled[train_idx], quality_labels[train_idx])
    pred_rf = rf.predict(features_scaled[test_idx])
    corr_rf = np.corrcoef(pred_rf, quality_labels[test_idx])[0, 1]
    results['random_forest'] = {
        'mae': mean_absolute_error(quality_labels[test_idx], pred_rf),
        'correlation': corr_rf if not np.isnan(corr_rf) else 0.0,
        'r2': r2_score(quality_labels[test_idx], pred_rf)
    }
    
    # baseline 6: gradient boosting on topological features
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(features_scaled[train_idx], quality_labels[train_idx])
    pred_gb = gb.predict(features_scaled[test_idx])
    corr_gb = np.corrcoef(pred_gb, quality_labels[test_idx])[0, 1]
    results['gradient_boost'] = {
        'mae': mean_absolute_error(quality_labels[test_idx], pred_gb),
        'correlation': corr_gb if not np.isnan(corr_gb) else 0.0,
        'r2': r2_score(quality_labels[test_idx], pred_gb)
    }
    
    if verbose:
        print(f"\n{'Method':<20} | {'MAE':<8} | {'Corr':<8} | {'R²':<8}")
        print("-" * 55)
        for method, metrics in results.items():
            print(f"{method:<20} | {metrics['mae']:<8.4f} | "
                  f"{metrics['correlation']:<8.4f} | {metrics['r2']:<8.4f}")
    
    return results


def get_feature_importances(
    topo_features: np.ndarray,
    quality_labels: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Compute feature importances using Random Forest.
    
    Args:
        topo_features: Topological features.
        quality_labels: Quality labels.
        feature_names: Names of features.
    
    Returns:
        Dictionary mapping feature names to importances.
    """
    # filter invalid values
    valid_mask = np.isfinite(topo_features).all(axis=1) & np.isfinite(quality_labels)
    features = topo_features[valid_mask]
    labels = quality_labels[valid_mask]
    
    # fit random forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(features, labels)
    
    # get importances
    importances = dict(zip(feature_names, rf.feature_importances_))
    
    # sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    
    return importances


if __name__ == "__main__":
    # test baselines with synthetic data
    print("Testing baseline comparisons...")
    
    n_samples = 200
    n_features = 27
    
    topo_features = np.random.randn(n_samples, n_features).astype(np.float32)
    quality_labels = np.random.rand(n_samples).astype(np.float32) * 2 + 1.5
    
    # add some correlation
    quality_labels += 0.3 * topo_features[:, 0] + 0.2 * topo_features[:, 5]
    
    landscape_stats = [
        {'center_loss': np.random.rand() * 3 + 2,
         'mean_loss': np.random.rand() * 3 + 2,
         'std_loss': np.random.rand() * 0.5}
        for _ in range(n_samples)
    ]
    
    results = run_baseline_comparisons(
        topo_features, quality_labels, landscape_stats, verbose=True
    )
