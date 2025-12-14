#!/usr/bin/env python3
"""
Step 4: Train Quality Predictor and Run Baselines

This script:
1. Trains the quality predictor on topological features
2. Runs baseline comparisons
3. Generates final results and figures

Usage:
    python scripts/04_train_quality_predictor.py --epochs 150

Input:
    data/quality_data.pt

Output:
    checkpoints/quality_predictor.pt
    figures/quality_prediction.png
    figures/baseline_comparison.png
    outputs/results.json

Authors: Apurva Mishra and Ayush Tibrewal
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quality_predictor import QualityPredictor, train_quality_predictor
from baselines import run_baseline_comparisons, get_feature_importances
from visualization import plot_quality_prediction_results, plot_baseline_comparison


def main():
    parser = argparse.ArgumentParser(description='Train quality predictor')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--figure_dir', type=str, default='figures',
                        help='Figure directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n" + "=" * 60)
    print("STEP 4: TRAIN QUALITY PREDICTOR AND RUN BASELINES")
    print("=" * 60)
    
    # load quality data
    quality_data_path = os.path.join(args.data_dir, 'quality_data.pt')
    if not os.path.exists(quality_data_path):
        print(f"ERROR: {quality_data_path} not found!")
        print("Run step 3 first: python scripts/03_generate_quality_data.py")
        return
    
    print(f"\nLoading quality data from {quality_data_path}...")
    quality_data = torch.load(quality_data_path, weights_only=False)
    
    topo_features = quality_data['topo_features']
    quality_labels = quality_data['quality_labels']
    landscape_stats = quality_data['landscape_stats']
    feature_names = quality_data['feature_names']
    
    print(f"  Samples: {len(quality_labels)}")
    print(f"  Features: {topo_features.shape[1]}")
    print(f"  Quality range: [{quality_labels.min():.4f}, {quality_labels.max():.4f}]")
    
    # train quality predictor
    print("\n" + "-" * 40)
    print("Training Quality Predictor")
    print("-" * 40)
    
    n_features = topo_features.shape[1]
    quality_predictor = QualityPredictor(
        n_features=n_features,
        hidden_dims=[128, 64, 32],
        dropout=0.2
    )
    
    n_params = sum(p.numel() for p in quality_predictor.parameters())
    print(f"  Model parameters: {n_params:,}")
    
    start_time = time.time()
    
    quality_result = train_quality_predictor(
        quality_predictor,
        topo_features,
        quality_labels,
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        verbose=True
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    # save quality predictor checkpoint
    checkpoint = {
        'model_state': quality_result['model'].state_dict(),
        'scaler': quality_result['scaler'],
        'label_mean': quality_result['label_mean'],
        'label_std': quality_result['label_std'],
        'history': quality_result['history'],
        'metrics': quality_result['metrics']
    }
    checkpoint_path = os.path.join(args.checkpoint_dir, 'quality_predictor.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # plot quality prediction results
    print("\nGenerating quality prediction plots...")
    plot_quality_prediction_results(
        quality_result['val_predictions'],
        quality_result['val_labels'],
        save_path=os.path.join(args.figure_dir, 'quality_prediction.png'),
        show=False
    )
    
    # run baseline comparisons
    print("\n" + "-" * 40)
    print("Running Baseline Comparisons")
    print("-" * 40)
    
    baseline_results = run_baseline_comparisons(
        topo_features,
        quality_labels,
        landscape_stats,
        test_split=args.val_split,
        verbose=True
    )
    
    # add our neural network method
    baseline_results['neural_net_topo'] = quality_result['metrics']
    
    # plot comparison
    print("\nGenerating comparison plots...")
    plot_baseline_comparison(
        baseline_results,
        save_path=os.path.join(args.figure_dir, 'baseline_comparison.png'),
        show=False
    )
    
    # compute feature importances
    print("\n" + "-" * 40)
    print("Feature Importances (Random Forest)")
    print("-" * 40)
    
    importances = get_feature_importances(topo_features, quality_labels, feature_names)
    print("\nTop 10 most important features:")
    for i, (name, imp) in enumerate(list(importances.items())[:10]):
        print(f"  {i+1}. {name}: {imp:.4f}")
    
    # save final results
    results = {
        'quality_predictor_metrics': quality_result['metrics'],
        'baseline_results': {k: {kk: float(vv) for kk, vv in v.items()} 
                            for k, v in baseline_results.items()},
        'feature_importances': {k: float(v) for k, v in importances.items()},
        'config': {
            'n_samples': len(quality_labels),
            'n_features': int(topo_features.shape[1]),
            'val_split': args.val_split,
            'epochs': args.epochs
        }
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # verification
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print("\nQuality Predictor (Neural Net on Topological Features):")
    print(f"  MAE: {quality_result['metrics']['mae']:.4f}")
    print(f"  Correlation: {quality_result['metrics']['correlation']:.4f}")
    print(f"  R²: {quality_result['metrics']['r2']:.4f}")
    
    print("\nBaseline Comparison:")
    print(f"  {'Method':<20} | {'MAE':<8} | {'Corr':<8} | {'R²':<8}")
    print("  " + "-" * 50)
    for method, metrics in baseline_results.items():
        corr = metrics.get('correlation', metrics.get('corr', 0))
        print(f"  {method:<20} | {metrics['mae']:<8.4f} | {corr:<8.4f} | {metrics['r2']:<8.4f}")
    
    # check if our method is best
    our_corr = quality_result['metrics']['correlation']
    best_baseline_corr = max(
        v.get('correlation', v.get('corr', 0)) 
        for k, v in baseline_results.items() 
        if k != 'neural_net_topo'
    )
    
    if our_corr > best_baseline_corr:
        print(f"\n✓ Our method outperforms best baseline by {our_corr - best_baseline_corr:.4f}")
    else:
        print(f"\n⚠ Best baseline outperforms our method by {best_baseline_corr - our_corr:.4f}")
    
    print(f"\nResults saved to: {results_path}")
    print(f"Figures saved to: {args.figure_dir}/")
    
    print("\n" + "=" * 60)
    print("STEP 4 COMPLETE")
    print("All steps finished! Ready for paper writing.")
    print("=" * 60)


if __name__ == '__main__':
    main()
