#!/usr/bin/env python3
"""
Quick test version of Phase 4 DQN training.

This script runs a shorter training session (~30 minutes on CPU)
for quick testing and validation. Use this before committing to
the full 8-12 hour training run.

Usage:
    python run_phase4_quick_test.py
"""

import sys
sys.path.insert(0, 'src')

from phase4_train_dqn import train_dqn, create_output_dirs, save_training_results
import torch

def main():
    print("\n" + "="*60)
    print("PHASE 4: QUICK TEST (100k steps, ~30 min on CPU)")
    print("="*60)
    print("\nThis is a shortened version for testing.")
    print("For full training (1M steps), run: python src/phase4_train_dqn.py")
    print("\n" + "="*60 + "\n")
    
    create_output_dirs()
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU (will take ~30 minutes)")
    
    # Train with reduced parameters
    results = train_dqn(
        num_steps=100_000,          # 10x less than full training
        batch_size=32,
        buffer_size=50_000,         # Smaller buffer
        learning_starts=10_000,     # Start learning earlier
        target_update_freq=5_000,   # More frequent updates
        save_freq=25_000,           # More frequent saves
        eval_freq=10_000,           # More frequent evals
        eval_episodes=5,            # Fewer eval episodes
        device=device
    )
    
    # Save results
    save_training_results(results)
    
    print("\n" + "="*60)
    print("✅ QUICK TEST COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("  1. Open notebooks/phase4_dqn_training.ipynb to analyze results")
    print("  2. If satisfied, run full training: python src/phase4_train_dqn.py")
    print("  3. Full training takes 8-12 hours on CPU (2-4 hours on GPU)")
    print("\n")

if __name__ == "__main__":
    main()
