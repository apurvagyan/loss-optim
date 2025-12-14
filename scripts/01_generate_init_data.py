#!/usr/bin/env python3
"""
Step 1: Generate Initialization Data

This script generates the (parameters, loss) pairs needed to train
the loss predictor network. This is the most time-consuming step.

Usage:
    python scripts/01_generate_init_data.py --n_samples 100000

Output:
    data/init_data.pt - Contains params, losses, init_types

Authors: Apurva Mishra and Ayush Tibrewal
"""

import os
import sys
import argparse
import torch
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from networks import get_target_network, print_network_summary
from data_utils import load_cifar10, get_fixed_batch
from init_data_generator import InitializationDataGenerator


def main():
    parser = argparse.ArgumentParser(description='Generate initialization data')
    parser.add_argument('--n_samples', type=int, default=100000,
                        help='Number of samples to generate')
    parser.add_argument('--perturbation_std', type=float, default=0.2,
                        help='Standard deviation of perturbations')
    parser.add_argument('--architecture', type=str, default='SmallCNN',
                        choices=['SmallCNN', 'TinyCNN', 'MicroCNN'],
                        help='Target network architecture')
    parser.add_argument('--train_subset', type=int, default=5000,
                        help='Number of training samples to use')
    parser.add_argument('--fixed_batch_size', type=int, default=1024,
                        help='Batch size for loss computation')
    parser.add_argument('--save_interval', type=int, default=20000,
                        help='Save checkpoint every N samples')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to save data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # check if data already exists
    output_path = os.path.join(args.data_dir, 'init_data.pt')
    if os.path.exists(output_path):
        print(f"\nWARNING: {output_path} already exists!")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            return
    
    # load cifar-10
    print("\n" + "=" * 60)
    print("STEP 1: GENERATE INITIALIZATION DATA")
    print("=" * 60)
    
    print("\nLoading CIFAR-10...")
    trainloader, testloader = load_cifar10(
        batch_size=128,
        subset_size=args.train_subset,
        data_dir=args.data_dir
    )
    
    fixed_images, fixed_labels = get_fixed_batch(
        trainloader, 
        batch_size=args.fixed_batch_size, 
        device=device
    )
    print(f"Fixed batch shape: {fixed_images.shape}")
    
    # get target network info
    target_model = get_target_network(args.architecture)
    print_network_summary(target_model)
    
    # create generator
    generator = InitializationDataGenerator(
        model_class=type(target_model),
        images=fixed_images,
        labels=fixed_labels,
        device=device
    )
    
    # generate data
    print(f"\nGenerating {args.n_samples:,} samples...")
    print(f"This will take approximately {args.n_samples / 3000:.0f} minutes")
    
    start_time = time.time()
    
    init_data = generator.generate(
        n_samples=args.n_samples,
        perturbation_std=args.perturbation_std,
        save_interval=args.save_interval,
        save_path=os.path.join(args.data_dir, 'init_data'),
        verbose=True
    )
    
    elapsed = time.time() - start_time
    
    # verify data
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    print(f"\nData saved to: {output_path}")
    print(f"Generation time: {elapsed/60:.1f} minutes")
    print(f"Params shape: {init_data['params'].shape}")
    print(f"Losses shape: {init_data['losses'].shape}")
    print(f"Loss range: [{init_data['losses'].min():.4f}, {init_data['losses'].max():.4f}]")
    print(f"Loss mean: {init_data['losses'].mean():.4f} ± {init_data['losses'].std():.4f}")
    
    # verify file can be loaded
    print("\nVerifying saved file...")
    loaded = torch.load(output_path, weights_only=False)
    assert loaded['params'].shape == init_data['params'].shape
    assert loaded['losses'].shape == init_data['losses'].shape
    print("✓ File verified successfully")
    
    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("Next: python scripts/02_train_loss_predictor.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
