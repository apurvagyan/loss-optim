#!/usr/bin/env python3
"""
Step 3: Generate Quality Dataset with Topological Features

This script:
1. Samples local loss landscapes using the trained loss predictor
2. Computes persistent homology features
3. Trains target networks to get ground truth quality labels

Usage:
    python scripts/03_generate_quality_data.py --n_samples 200

Input:
    checkpoints/loss_predictor.pt

Output:
    data/quality_data.pt
    figures/landscape_*.png
    figures/persistence_*.png

Authors: Apurva Mishra and Ayush Tibrewal
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from networks import get_target_network
from data_utils import load_cifar10, get_fixed_batch
from loss_predictor import load_checkpoint
from topology import (
    LossLandscapeSampler, TopologicalFeatureExtractor,
    extract_numerical_features, get_feature_names
)
from quality_predictor import evaluate_initialization
from visualization import plot_loss_landscape_2d, plot_persistence_diagrams


def main():
    parser = argparse.ArgumentParser(description='Generate quality dataset')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of initializations to evaluate')
    parser.add_argument('--n_landscape_samples', type=int, default=2000,
                        help='Samples per local landscape')
    parser.add_argument('--landscape_radius', type=float, default=0.3,
                        help='Radius for landscape sampling')
    parser.add_argument('--training_epochs', type=int, default=15,
                        help='Epochs to train target for quality measurement')
    parser.add_argument('--architecture', type=str, default='SmallCNN',
                        choices=['SmallCNN', 'TinyCNN', 'MicroCNN'],
                        help='Target network architecture')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--figure_dir', type=str, default='figures',
                        help='Figure directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_interval', type=int, default=50,
                        help='Save checkpoint every N samples')
    args = parser.parse_args()
    
    # setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "=" * 60)
    print("STEP 3: GENERATE QUALITY DATASET")
    print("=" * 60)
    
    # load loss predictor
    checkpoint_path = os.path.join(args.checkpoint_dir, 'loss_predictor.pt')
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: {checkpoint_path} not found!")
        print("Run step 2 first: python scripts/02_train_loss_predictor.py")
        return
    
    print(f"\nLoading loss predictor from {checkpoint_path}...")
    loss_predictor, history = load_checkpoint(checkpoint_path, device)
    print("  Loss predictor loaded successfully")
    print(f"  Best validation MAE: {min(history['val_mae']):.4f}")
    print(f"  Best validation correlation: {max(history['val_corr']):.4f}")
    
    # check for existing quality data
    output_path = os.path.join(args.data_dir, 'quality_data.pt')
    checkpoint_quality_path = os.path.join(args.data_dir, 'quality_data_checkpoint.pt')
    
    start_idx = 0
    all_topo_features = []
    all_quality_labels = []
    all_landscape_stats = []
    
    if os.path.exists(checkpoint_quality_path):
        print(f"\nFound checkpoint at {checkpoint_quality_path}")
        response = input("Resume from checkpoint? (y/n): ")
        if response.lower() == 'y':
            checkpoint = torch.load(checkpoint_quality_path, weights_only=False)
            all_topo_features = checkpoint['topo_features']
            all_quality_labels = checkpoint['quality_labels']
            all_landscape_stats = checkpoint['landscape_stats']
            start_idx = len(all_quality_labels)
            print(f"  Resuming from sample {start_idx}")
    
    if os.path.exists(output_path) and start_idx == 0:
        print(f"\nWARNING: {output_path} already exists!")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
    
    # load data for training
    print("\nLoading CIFAR-10...")
    trainloader, testloader = load_cifar10(
        batch_size=128,
        subset_size=5000,
        data_dir=args.data_dir
    )
    
    # create sampler and extractor
    print("\nInitializing landscape sampler and topological extractor...")
    sampler = LossLandscapeSampler(
        loss_predictor,
        history['param_mean'],
        history['param_std'],
        history['loss_mean'],
        history['loss_std'],
        device
    )
    
    topo_extractor = TopologicalFeatureExtractor(
        max_homology_dim=1,
        max_points_for_rips=500,
        projection_dim=50,
        n_sublevel_thresholds=50
    )
    
    # get target model class
    target_model = get_target_network(args.architecture)
    target_model_class = type(target_model)
    
    # generate quality dataset
    print(f"\nGenerating quality data for {args.n_samples} initializations...")
    print(f"  Landscape samples per init: {args.n_landscape_samples}")
    print(f"  Training epochs per init: {args.training_epochs}")
    
    start_time = time.time()
    
    for i in tqdm(range(start_idx, args.n_samples)):
        # generate random initialization
        model = target_model_class().to(device)
        
        # random initialization method
        method = np.random.choice(['xavier', 'kaiming', 'orthogonal'])
        scale = np.random.uniform(0.6, 1.4)
        
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                if method == 'xavier':
                    torch.nn.init.xavier_uniform_(param, gain=scale)
                elif method == 'kaiming':
                    torch.nn.init.kaiming_uniform_(param, a=np.sqrt(5))
                    param.data *= scale
                else:
                    torch.nn.init.orthogonal_(param, gain=scale)
            elif 'bias' in name:
                torch.nn.init.zeros_(param)
        
        init_params = model.get_flat_params().cpu()
        
        # sample local loss landscape
        landscape_data = sampler.sample_local_landscape(
            init_params,
            n_samples=args.n_landscape_samples,
            radius=args.landscape_radius,
            method='gaussian'
        )
        
        # compute topological features
        topo_data = topo_extractor.extract_all_features(landscape_data)
        topo_features = extract_numerical_features(topo_data)
        
        # get ground truth quality by actually training
        quality = evaluate_initialization(
            target_model_class, init_params,
            trainloader, testloader,
            epochs=args.training_epochs,
            lr=0.01,
            momentum=0.9,
            device=device,
            verbose=False
        )
        
        all_topo_features.append(topo_features)
        all_quality_labels.append(quality['final_test_loss'])
        all_landscape_stats.append({
            'center_loss': landscape_data['center_loss'],
            'mean_loss': float(landscape_data['losses'].mean()),
            'std_loss': float(landscape_data['losses'].std()),
            'init_method': method,
            'init_scale': scale,
            'final_test_acc': quality['final_test_acc']
        })
        
        # save example visualizations for first few
        if i < 5:
            plot_loss_landscape_2d(
                landscape_data,
                save_path=os.path.join(args.figure_dir, f'landscape_{i}.png'),
                show=False
            )
            plot_persistence_diagrams(
                topo_data,
                title=f'Initialization {i} ({method}, scale={scale:.2f})',
                save_path=os.path.join(args.figure_dir, f'persistence_{i}.png'),
                show=False
            )
        
        # periodic checkpoint
        if (i + 1) % args.save_interval == 0:
            checkpoint = {
                'topo_features': all_topo_features,
                'quality_labels': all_quality_labels,
                'landscape_stats': all_landscape_stats,
                'feature_names': get_feature_names()
            }
            torch.save(checkpoint, checkpoint_quality_path)
            elapsed = time.time() - start_time
            rate = (i + 1 - start_idx) / elapsed * 60
            print(f"\n  Checkpoint saved: {i+1}/{args.n_samples} samples, {rate:.1f} samples/min")
    
    elapsed = time.time() - start_time
    
    # save final dataset
    quality_data = {
        'topo_features': np.stack(all_topo_features),
        'quality_labels': np.array(all_quality_labels),
        'landscape_stats': all_landscape_stats,
        'feature_names': get_feature_names(),
        'config': {
            'n_samples': args.n_samples,
            'n_landscape_samples': args.n_landscape_samples,
            'landscape_radius': args.landscape_radius,
            'training_epochs': args.training_epochs,
            'architecture': args.architecture
        }
    }
    
    torch.save(quality_data, output_path)
    
    # clean up checkpoint
    if os.path.exists(checkpoint_quality_path):
        os.remove(checkpoint_quality_path)
    
    # verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    print(f"\nData saved to: {output_path}")
    print(f"Generation time: {elapsed/60:.1f} minutes")
    print(f"Topological features shape: {quality_data['topo_features'].shape}")
    print(f"Quality labels shape: {quality_data['quality_labels'].shape}")
    print(f"Feature names: {len(quality_data['feature_names'])} features")
    
    # quality label statistics
    labels = quality_data['quality_labels']
    print(f"\nQuality labels (final test loss):")
    print(f"  Min: {labels.min():.4f}")
    print(f"  Max: {labels.max():.4f}")
    print(f"  Mean: {labels.mean():.4f} ± {labels.std():.4f}")
    
    # test accuracy statistics
    test_accs = [s['final_test_acc'] for s in all_landscape_stats]
    print(f"\nTest accuracy:")
    print(f"  Min: {min(test_accs):.4f}")
    print(f"  Max: {max(test_accs):.4f}")
    print(f"  Mean: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    
    # verify file
    print("\nVerifying saved file...")
    loaded = torch.load(output_path, weights_only=False)
    assert loaded['topo_features'].shape == quality_data['topo_features'].shape
    assert loaded['quality_labels'].shape == quality_data['quality_labels'].shape
    print("✓ File verified successfully")
    
    print("\n" + "=" * 60)
    print("STEP 3 COMPLETE")
    print(f"Figures saved to: {args.figure_dir}/")
    print("Next: python scripts/04_train_quality_predictor.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
