#!/usr/bin/env python3
"""
Quick test version of Phase 5 Improved DQN training.

Tests the Double DQN + Dueling Network + Prioritized Replay improvements
with a shorter training run (~30 minutes on CPU).

Usage:
    python run_phase5_quick_test.py
"""

import sys
sys.path.insert(0, 'src')

from phase5_train_improved_dqn import train_improved_dqn, create_output_dirs, save_training_results
import torch

def main():
    print("\n" + "="*60)
    print("PHASE 5: QUICK TEST (100k steps, ~30 min on CPU)")
    print("="*60)
    print("\nTesting DQN improvements:")
    print("  ✓ Double DQN")
    print("  ✓ Dueling Networks")
    print("  ✓ Prioritized Experience Replay")
    print("\nFor full training (1M steps), run: python src/phase5_train_improved_dqn.py")
    print("\n" + "="*60 + "\n")
    
    create_output_dirs()
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU (will take ~30 minutes)")
    
    # Train with reduced parameters
    results = train_improved_dqn(
        num_steps=100_000,
        batch_size=32,
        buffer_size=50_000,
        learning_starts=10_000,
        target_update_freq=5_000,
        save_freq=25_000,
        eval_freq=10_000,
        eval_episodes=5,
        device=device
    )
    
    # Save results
    save_training_results(results)
    
    print("\n" + "="*60)
    print("✅ QUICK TEST COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("  1. Compare results with Phase 4 (should see improvement!)")
    print("  2. If satisfied, run full training: python src/phase5_train_improved_dqn.py")
    print("  3. Full training takes 8-12 hours on CPU (2-4 hours on GPU)")
    print("\n")

if __name__ == "__main__":
    main()
