#!/usr/bin/env python3
"""
Debug script to diagnose why loss is not decreasing.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os

# Add the ObjectSeg directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from datamodule.goose_dataset import GooseDataset
from models.segformer.segformer import LitSegFormer, SegFormer
from datamodule.utils import read_label_mapping

def check_data_distribution():
    """Check if the data distribution is reasonable."""
    print("🔍 Checking data distribution...")
    
    dataset = GooseDataset(
        root_dir="/home/autokarthik/autoware.off-road/goose-dataset", 
        split="train",
        label_mapping_name="goose_label_mapping.csv"
    )
    
    # Check a few samples
    for i in range(3):
        img, mask = dataset[i]
        print(f"Sample {i}:")
        print(f"  Image shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Mask shape: {mask.shape}, unique values: {torch.unique(mask).tolist()}")
        print(f"  Mask range: [{mask.min()}, {mask.max()}]")
        print()

def check_model_outputs():
    """Check if model outputs are reasonable."""
    print("🔍 Checking model outputs...")
    
    # Create model
    model = SegFormer(variant='B2', num_classes=64)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(img)
        print(f"Model output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"Output mean: {output.mean():.3f}")
        print(f"Output std: {output.std():.3f}")
        
        # Check if outputs are reasonable (not all zeros or NaNs)
        if torch.isnan(output).any():
            print("❌ Model output contains NaN values!")
        elif torch.isinf(output).any():
            print("❌ Model output contains infinite values!")
        elif output.abs().max() < 1e-6:
            print("❌ Model output is too close to zero!")
        else:
            print("✅ Model outputs look reasonable")
    print()

def check_loss_computation():
    """Check if loss computation is working correctly."""
    print("🔍 Checking loss computation...")
    
    # Create model and loss
    model = SegFormer(variant='B2', num_classes=64)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    # Create dummy data
    batch_size = 2
    img = torch.randn(batch_size, 3, 224, 224)
    mask = torch.randint(0, 64, (batch_size, 224, 224))
    
    # Forward pass
    output = model(img)
    loss = criterion(output, mask)
    
    print(f"Input shape: {img.shape}")
    print(f"Target shape: {mask.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Loss value: {loss.item():.4f}")
    
    # Check gradients
    loss.backward()
    total_grad_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            param_count += 1
            if grad_norm > 10:
                print(f"⚠️  Large gradient in {name}: {grad_norm:.4f}")
    
    total_grad_norm = total_grad_norm ** 0.5
    print(f"Total gradient norm: {total_grad_norm:.4f}")
    print(f"Parameters with gradients: {param_count}")
    
    if total_grad_norm < 1e-6:
        print("❌ Gradients are too small - this could cause training issues!")
    elif total_grad_norm > 100:
        print("❌ Gradients are too large - this could cause training instability!")
    else:
        print("✅ Gradients look reasonable")
    print()

def check_learning_rate():
    """Check if learning rate is reasonable."""
    print("🔍 Checking learning rate configuration...")
    
    # Current config
    lr = 2e-4
    weight_decay = 0.01
    
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")
    
    if lr < 1e-6:
        print("❌ Learning rate is too small!")
    elif lr > 1e-1:
        print("❌ Learning rate is too large!")
    else:
        print("✅ Learning rate looks reasonable")
    
    if weight_decay > 0.1:
        print("❌ Weight decay is too high!")
    else:
        print("✅ Weight decay looks reasonable")
    print()

def check_class_balance():
    """Check if there's severe class imbalance."""
    print("🔍 Checking class balance...")
    
    dataset = GooseDataset(
        root_dir="/home/autokarthik/autoware.off-road/goose-dataset", 
        split="train",
        label_mapping_name="goose_label_mapping.csv"
    )
    
    # Count class occurrences
    class_counts = {}
    num_samples = min(100, len(dataset))  # Check first 100 samples
    
    for i in range(num_samples):
        _, mask = dataset[i]
        unique, counts = torch.unique(mask, return_counts=True)
        for label, count in zip(unique.tolist(), counts.tolist()):
            class_counts[label] = class_counts.get(label, 0) + count
    
    print(f"Class distribution (first {num_samples} samples):")
    for label in sorted(class_counts.keys()):
        print(f"  Class {label}: {class_counts[label]}")
    
    # Check for severe imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if ratio > 100:
            print(f"❌ Severe class imbalance detected! Ratio: {ratio:.1f}")
            print("   Consider using weighted loss or focal loss")
        else:
            print(f"✅ Class balance looks reasonable (ratio: {ratio:.1f})")
    print()

def main():
    print("🚀 Starting training diagnostics...\n")
    
    check_data_distribution()
    check_model_outputs()
    check_loss_computation()
    check_learning_rate()
    check_class_balance()
    
    print("📋 Summary of potential issues:")
    print("1. If gradients are too small: Increase learning rate or check model initialization")
    print("2. If gradients are too large: Decrease learning rate or add gradient clipping")
    print("3. If class imbalance is severe: Use weighted loss or focal loss")
    print("4. If model outputs are NaN: Check for numerical instability")
    print("5. If data range is wrong: Check normalization")

if __name__ == "__main__":
    main() 