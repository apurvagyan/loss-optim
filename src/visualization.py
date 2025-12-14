"""
Visualization Utilities.

Provides plotting functions for training curves, persistence diagrams,
loss landscapes, and quality prediction results.

Authors: Apurva Mishra and Ayush Tibrewal
Course: CPSC 6440 - Geometric and Topological Methods in Machine Learning
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from sklearn.decomposition import PCA
from persim import plot_diagrams
import os


# set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_loss_predictor_training(
    history: Dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot training curves for the loss predictor."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.8, linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Validation', alpha=0.8, linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss (normalized)')
    axes[0, 0].set_title('Loss Predictor Training')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['val_mae'], color='green', alpha=0.8, linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (original scale)')
    best_mae = min(history['val_mae'])
    axes[0, 1].set_title(f'Validation MAE (best: {best_mae:.4f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['val_corr'], color='purple', alpha=0.8, linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Correlation')
    best_corr = max(history['val_corr'])
    axes[1, 0].set_title(f'Validation Correlation (best: {best_corr:.4f})')
    axes[1, 0].axhline(0.9, color='red', linestyle='--', alpha=0.5, label='0.9 target')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    axes[1, 1].plot(history['lr'], color='orange', alpha=0.8, linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_loss_predictor_evaluation(
    true_losses: np.ndarray,
    pred_losses: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot loss predictor evaluation results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    valid = np.isfinite(true_losses) & np.isfinite(pred_losses)
    true_valid = true_losses[valid]
    pred_valid = pred_losses[valid]
    
    corr = np.corrcoef(true_valid, pred_valid)[0, 1]
    mae = np.abs(true_valid - pred_valid).mean()
    
    axes[0].scatter(true_valid, pred_valid, alpha=0.3, s=15)
    lims = [
        max(0, min(true_valid.min(), pred_valid.min()) - 0.5),
        min(50, max(true_valid.max(), pred_valid.max()) + 0.5)
    ]
    axes[0].plot(lims, lims, 'r--', linewidth=2, label='Perfect prediction')
    axes[0].set_xlabel('True Loss')
    axes[0].set_ylabel('Predicted Loss')
    axes[0].set_title(f'Predicted vs True Loss (r={corr:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    
    residuals = pred_valid - true_valid
    axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residuals (MAE={mae:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_persistence_diagrams(
    topo_data: Dict,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot persistence diagrams and Betti curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    plot_diagrams(topo_data['rips']['diagrams'], ax=axes[0], show=False)
    axes[0].set_title('Rips Persistence Diagram')
    
    plot_diagrams(topo_data['weighted']['diagrams'], ax=axes[1], show=False)
    axes[1].set_title('Loss-Weighted Persistence')
    
    sublevel = topo_data['sublevel']
    axes[2].plot(sublevel['thresholds'], sublevel['betti_0_curve'],
                 label='β₀ (components)', linewidth=2)
    axes[2].plot(sublevel['thresholds'], sublevel['betti_1_curve'],
                 label='β₁ (cycles)', linewidth=2)
    axes[2].axvline(topo_data['center_loss'], color='red', linestyle='--',
                    alpha=0.7, label='Center loss')
    axes[2].set_xlabel('Loss Threshold (α)')
    axes[2].set_ylabel('Betti Number')
    axes[2].set_title('Sublevel Set Betti Curves')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_loss_landscape_2d(
    landscape_data: Dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Visualize sampled loss landscape using PCA projection."""
    perturbations = landscape_data['perturbations'].numpy()
    losses = landscape_data['losses'].numpy()
    center_loss = landscape_data['center_loss']
    
    pca = PCA(n_components=2)
    projected = pca.fit_transform(perturbations)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    scatter = axes[0].scatter(projected[:, 0], projected[:, 1],
                               c=losses, cmap='viridis', alpha=0.5, s=15)
    axes[0].scatter([0], [0], c='red', s=150, marker='*',
                    edgecolors='black', linewidths=1, label='Center', zorder=5)
    plt.colorbar(scatter, ax=axes[0], label='Predicted Loss')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    axes[0].set_title(f'Loss Landscape (Center: {center_loss:.3f})')
    axes[0].legend()
    
    axes[1].hist(losses, bins=50, edgecolor='black', alpha=0.7, density=True)
    axes[1].axvline(center_loss, color='red', linestyle='--', linewidth=2,
                    label=f'Center: {center_loss:.3f}')
    axes[1].axvline(losses.mean(), color='blue', linestyle='--', linewidth=2,
                    label=f'Mean: {losses.mean():.3f}')
    axes[1].set_xlabel('Predicted Loss')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Loss Distribution')
    axes[1].legend()
    
    distances = np.linalg.norm(perturbations, axis=1)
    axes[2].scatter(distances, losses, alpha=0.3, s=10)
    axes[2].axhline(center_loss, color='red', linestyle='--', alpha=0.7)
    
    z = np.polyfit(distances, losses, 2)
    p = np.poly1d(z)
    x_fit = np.linspace(0, distances.max(), 100)
    axes[2].plot(x_fit, p(x_fit), 'g-', linewidth=2, label='Quadratic fit')
    
    axes[2].set_xlabel('Distance from Center')
    axes[2].set_ylabel('Predicted Loss')
    axes[2].set_title('Loss vs Distance')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_quality_prediction_results(
    predictions: np.ndarray,
    actuals: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot quality prediction results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    corr = np.corrcoef(predictions, actuals)[0, 1]
    mae = np.abs(predictions - actuals).mean()
    
    axes[0].scatter(actuals, predictions, alpha=0.6, s=40)
    lims = [min(actuals.min(), predictions.min()) - 0.1,
            max(actuals.max(), predictions.max()) + 0.1]
    axes[0].plot(lims, lims, 'r--', linewidth=2, label='Perfect')
    axes[0].set_xlabel('Actual Test Loss')
    axes[0].set_ylabel('Predicted Test Loss')
    axes[0].set_title(f'Quality Prediction (r={corr:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    residuals = predictions - actuals
    axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residuals (MAE={mae:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def plot_baseline_comparison(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot comparison with baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(results.keys())
    correlations = [results[m].get('correlation', results[m].get('corr', 0)) for m in methods]
    maes = [results[m]['mae'] for m in methods]
    
    display_names = [m.replace('_', '\n') for m in methods]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(methods)))
    
    bars1 = axes[0].bar(range(len(methods)), correlations, color=colors, edgecolor='black')
    axes[0].set_xticks(range(len(methods)))
    axes[0].set_xticklabels(display_names, fontsize=9)
    axes[0].set_ylabel('Correlation')
    axes[0].set_title('Method Comparison: Correlation')
    axes[0].set_ylim([0, 1])
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, correlations):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    bars2 = axes[1].bar(range(len(methods)), maes, color=colors, edgecolor='black')
    axes[1].set_xticks(range(len(methods)))
    axes[1].set_xticklabels(display_names, fontsize=9)
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Method Comparison: MAE (lower is better)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars2, maes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    
    return fig


def create_paper_figures(
    loss_pred_history: Dict,
    quality_result: Dict,
    baseline_results: Dict,
    example_topo_data: Dict,
    example_landscape: Dict,
    save_dir: str = "figures"
) -> None:
    """Create all figures for the paper."""
    os.makedirs(save_dir, exist_ok=True)
    
    plot_loss_predictor_training(
        loss_pred_history,
        save_path=os.path.join(save_dir, "fig1_loss_predictor_training.pdf"),
        show=False
    )
    
    plot_persistence_diagrams(
        example_topo_data,
        title="Example Initialization",
        save_path=os.path.join(save_dir, "fig2_persistence_diagrams.pdf"),
        show=False
    )
    
    plot_loss_landscape_2d(
        example_landscape,
        save_path=os.path.join(save_dir, "fig3_loss_landscape.pdf"),
        show=False
    )
    
    plot_quality_prediction_results(
        quality_result['val_predictions'],
        quality_result['val_labels'],
        save_path=os.path.join(save_dir, "fig4_quality_prediction.pdf"),
        show=False
    )
    
    plot_baseline_comparison(
        baseline_results,
        save_path=os.path.join(save_dir, "fig5_baseline_comparison.pdf"),
        show=False
    )
    
    print(f"All figures saved to {save_dir}/")
