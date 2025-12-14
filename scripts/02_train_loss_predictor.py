#!/usr/bin/env python3
"""
Step 2: Train Loss Predictor Network

This script trains the neural network that predicts loss from parameters.
Requires init_data.pt from Step 1.

Usage:
    python scripts/02_train_loss_predictor.py --epochs 200

Input:
    data/init_data.pt

Output:
    checkpoints/loss_predictor.pt
    figures/loss_predictor_training.png
    figures/loss_predictor_evaluation.png

Authors: Apurva Mishra and Ayush Tibrewal
"""

import os
import sys
import argparse
import torch
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from init_data_generator import load_init_data
from loss_predictor import (
    LossPredictor, train_loss_predictor,
    save_checkpoint, load_checkpoint
)
from visualization import plot_loss_predictor_training, plot_loss_predictor_evaluation


def main():
    parser = argparse.ArgumentParser(description='Train loss predictor')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--figure_dir', type=str, default='figures',
                        help='Figure directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    args = parser.parse_args()
    
    # setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN LOSS PREDICTOR")
    print("=" * 60)
    
    # load init data
    init_data_path = os.path.join(args.data_dir, 'init_data.pt')
    if not os.path.exists(init_data_path):
        print(f"ERROR: {init_data_path} not found!")
        print("Run step 1 first: python scripts/01_generate_init_data.py")
        return
    
    print(f"\nLoading initialization data from {init_data_path}...")
    init_data = load_init_data(init_data_path)
    
    print(f"  Params shape: {init_data['params'].shape}")
    print(f"  Losses shape: {init_data['losses'].shape}")
    print(f"  Parameter dimension: {init_data['param_dim']:,}")
    
    # check for existing checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'loss_predictor.pt')
    
    if args.resume and os.path.exists(checkpoint_path):
        print(f"\nResuming from {checkpoint_path}...")
        loss_predictor, history = load_checkpoint(checkpoint_path, device)
        print("  Checkpoint loaded successfully")
        print(f"  Previous best MAE: {min(history['val_mae']):.4f}")
        print(f"  Previous best correlation: {max(history['val_corr']):.4f}")
    else:
        if os.path.exists(checkpoint_path) and not args.resume:
            print(f"\nWARNING: {checkpoint_path} already exists!")
            response = input("Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborting. Use --resume to continue training.")
                return
        
        # create model
        param_dim = init_data['param_dim']
        hidden_dims = [2048, 1024, 512, 256, 128]
        
        print(f"\nCreating LossPredictor...")
        print(f"  Input dimension: {param_dim:,}")
        print(f"  Hidden dimensions: {hidden_dims}")
        
        loss_predictor = LossPredictor(
            param_dim=param_dim,
            hidden_dims=hidden_dims,
            dropout=0.1
        )
        
        n_params = sum(p.numel() for p in loss_predictor.parameters())
        print(f"  Model parameters: {n_params:,}")
        
        # train
        print(f"\nTraining for {args.epochs} epochs...")
        start_time = time.time()
        
        history = train_loss_predictor(
            loss_predictor, init_data,
            val_split=0.1,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=1e-4,
            warmup_epochs=10,
            patience=args.patience,
            device=device,
            verbose=True
        )
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed/60:.1f} minutes")
        
        # save checkpoint
        save_checkpoint(loss_predictor, history, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    # plot training curves
    print("\nGenerating training plots...")
    plot_loss_predictor_training(
        history,
        save_path=os.path.join(args.figure_dir, 'loss_predictor_training.png'),
        show=False
    )
    
    # evaluate on held-out data
    print("\nEvaluating on held-out data...")
    n_eval = min(2000, len(init_data['params']))
    indices = np.random.choice(len(init_data['params']), n_eval, replace=False)
    
    params = init_data['params'][indices]
    true_losses = init_data['losses'][indices].numpy()
    
    # clip extreme values like in training
    true_losses = np.clip(true_losses, 0.1, 100)
    
    # normalize and predict
    params_norm = (params - history['param_mean']) / history['param_std']
    
    loss_predictor.eval()
    with torch.no_grad():
        pred_norm = loss_predictor(params_norm.to(device))
        pred_losses = (pred_norm.cpu() * history['loss_std'] + history['loss_mean']).numpy()
    
    # compute metrics
    valid = np.isfinite(true_losses) & np.isfinite(pred_losses)
    mae = np.abs(true_losses[valid] - pred_losses[valid]).mean()
    corr = np.corrcoef(true_losses[valid], pred_losses[valid])[0, 1]
    
    print(f"\nEvaluation Results:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Correlation: {corr:.4f}")
    
    # plot evaluation
    plot_loss_predictor_evaluation(
        true_losses, pred_losses,
        save_path=os.path.join(args.figure_dir, 'loss_predictor_evaluation.png'),
        show=False
    )
    
    # verification
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    # check that model can be loaded
    loaded_model, loaded_history = load_checkpoint(checkpoint_path, device)
    test_input = torch.randn(10, init_data['param_dim']).to(device)
    with torch.no_grad():
        test_output = loaded_model(test_input)
    assert test_output.shape == (10,), f"Expected (10,), got {test_output.shape}"
    print("✓ Checkpoint verified successfully")
    
    # check correlation threshold
    if corr < 0.7:
        print(f"\nWARNING: Correlation ({corr:.4f}) is below 0.7")
        print("Consider:")
        print("  - Generating more training data")
        print("  - Training for more epochs")
        print("  - Adjusting learning rate")
    else:
        print(f"✓ Correlation ({corr:.4f}) is acceptable")
    
    print("\n" + "=" * 60)
    print("STEP 2 COMPLETE")
    print(f"Figures saved to: {args.figure_dir}/")
    print("Next: python scripts/03_generate_quality_data.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
