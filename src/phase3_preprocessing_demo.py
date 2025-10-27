"""
Phase 3: Preprocessing & Feature Engineering

This script demonstrates the preprocessing pipeline and compares
raw vs preprocessed observations:
1. Shows before/after preprocessing
2. Compares observation shapes and data types
3. Tests preprocessed environment with random agent
4. Validates preprocessing effects on performance
5. Saves visualization of preprocessing steps
"""

import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm

# Import our preprocessing utilities
from preprocessing import make_atari_env, make_atari_env_rgb

# Register ALE environments
gym.register_envs(ale_py)

# Set style
sns.set_style('whitegrid')


def create_output_dirs():
    """Create necessary output directories."""
    dirs = ['outputs/phase3', 'data/phase3']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Created output directories")


def compare_raw_vs_preprocessed():
    """Compare raw and preprocessed observations."""
    print("\n" + "="*60)
    print("COMPARING RAW VS PREPROCESSED OBSERVATIONS")
    print("="*60 + "\n")
    
    # Create raw environment
    env_raw = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
    obs_raw, _ = env_raw.reset()
    
    # Create preprocessed environment
    env_preprocessed = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=True)
    obs_preprocessed, _ = env_preprocessed.reset()
    
    print("📊 OBSERVATION COMPARISON:")
    print("-" * 60)
    print(f"Raw Observation:")
    print(f"  Shape: {obs_raw.shape}")
    print(f"  Dtype: {obs_raw.dtype}")
    print(f"  Memory: {obs_raw.nbytes / 1024:.2f} KB")
    print(f"  Range: [{obs_raw.min()}, {obs_raw.max()}]")
    
    print(f"\nPreprocessed Observation:")
    print(f"  Shape: {obs_preprocessed.shape}")
    print(f"  Dtype: {obs_preprocessed.dtype}")
    print(f"  Memory: {obs_preprocessed.nbytes / 1024:.2f} KB")
    print(f"  Range: [{obs_preprocessed.min():.2f}, {obs_preprocessed.max():.2f}]")
    
    print(f"\n💾 MEMORY REDUCTION:")
    memory_reduction = (1 - obs_preprocessed.nbytes / obs_raw.nbytes) * 100
    print(f"  Reduction: {memory_reduction:.1f}%")
    
    # Calculate dimensions
    raw_dims = np.prod(obs_raw.shape)
    proc_dims = np.prod(obs_preprocessed.shape)
    dim_reduction = (1 - proc_dims / raw_dims) * 100
    print(f"  Dimensionality reduction: {dim_reduction:.1f}%")
    
    env_raw.close()
    env_preprocessed.close()
    
    return obs_raw, obs_preprocessed


def visualize_preprocessing_steps():
    """Visualize each step of the preprocessing pipeline."""
    print("\n" + "="*60)
    print("VISUALIZING PREPROCESSING STEPS")
    print("="*60 + "\n")
    
    # 1. Raw frame
    env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
    obs_raw, _ = env.reset()
    
    # Take a few steps to get interesting frame
    for _ in range(20):
        obs_raw, _, _, _, _ = env.step(env.action_space.sample())
    
    # 2. Grayscale + Resize
    import cv2
    obs_gray = cv2.cvtColor(obs_raw, cv2.COLOR_RGB2GRAY)
    obs_resized = cv2.resize(obs_gray, (84, 84), interpolation=cv2.INTER_AREA)
    
    # 3. Normalized
    obs_normalized = obs_resized.astype(np.float32) / 255.0
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Raw RGB
    axes[0, 0].imshow(obs_raw)
    axes[0, 0].set_title(f'1. Raw RGB Frame\n{obs_raw.shape}', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Grayscale
    axes[0, 1].imshow(obs_gray, cmap='gray')
    axes[0, 1].set_title(f'2. Grayscale\n{obs_gray.shape}', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Resized
    axes[0, 2].imshow(obs_resized, cmap='gray')
    axes[0, 2].set_title(f'3. Resized to 84x84\n{obs_resized.shape}', fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Normalized
    axes[1, 0].imshow(obs_normalized, cmap='gray')
    axes[1, 0].set_title(f'4. Normalized [0,1]\n{obs_normalized.shape}', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Frame stack visualization
    env_stacked = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4)
    obs_stacked, _ = env_stacked.reset()
    for _ in range(20):
        obs_stacked, _, _, _, _ = env_stacked.step(env_stacked.action_space.sample())
    
    # Show first and last frame of stack
    axes[1, 1].imshow(obs_stacked[:, :, 0], cmap='gray')
    axes[1, 1].set_title(f'5. Frame Stack (1st)\n{obs_stacked.shape}', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(obs_stacked[:, :, 3], cmap='gray')
    axes[1, 2].set_title(f'6. Frame Stack (4th)\n{obs_stacked.shape}', fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle('Preprocessing Pipeline Steps', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/phase3/preprocessing_steps.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved preprocessing steps visualization")
    
    env.close()
    env_stacked.close()


def test_preprocessed_environment(num_episodes=20):
    """Test the preprocessed environment with random agent."""
    print("\n" + "="*60)
    print("TESTING PREPROCESSED ENVIRONMENT")
    print("="*60 + "\n")
    
    env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=True)
    
    print(f"✓ Created preprocessed environment")
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    print(f"\n✓ Running {num_episodes} episodes with random agent...\n")
    
    episode_rewards = []
    episode_lengths = []
    clipped_rewards = []
    
    for episode in tqdm(range(num_episodes), desc="Testing episodes"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_clipped = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_clipped += abs(reward)  # Count clipped rewards
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        clipped_rewards.append(episode_clipped)
    
    env.close()
    
    # Calculate statistics
    stats = {
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'mean_clipped_rewards': float(np.mean(clipped_rewards)),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'timestamp': datetime.now().isoformat()
    }
    
    print("\n📊 PREPROCESSED ENVIRONMENT STATISTICS:")
    print("-" * 60)
    print(f"Mean Reward (clipped): {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Mean Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"Mean Clipped Rewards Count: {stats['mean_clipped_rewards']:.1f}")
    
    print("\n💡 NOTE: Rewards are clipped to {-1, 0, 1}")
    print("   This is standard practice for DQN and helps with training stability")
    
    return stats


def create_comparison_visualization(stats):
    """Create visualization comparing preprocessed vs original baseline."""
    print("\n" + "="*60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load Phase 2 baseline for comparison
    try:
        with open('data/phase2/baseline_statistics.json', 'r') as f:
            baseline_stats = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Episode rewards comparison
        episodes = range(1, len(stats['episode_rewards']) + 1)
        axes[0].plot(episodes, stats['episode_rewards'], 'o-', alpha=0.6, 
                     label='Preprocessed (clipped)', color='steelblue')
        axes[0].axhline(y=stats['mean_reward'], color='blue', linestyle='--', 
                        label=f"Mean (clipped): {stats['mean_reward']:.2f}", linewidth=2)
        axes[0].set_xlabel('Episode', fontsize=12)
        axes[0].set_ylabel('Total Reward', fontsize=12)
        axes[0].set_title('Preprocessed Environment Performance\n(with reward clipping)', 
                         fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Episode lengths comparison
        axes[1].bar(episodes, stats['episode_lengths'], alpha=0.7, 
                    color='coral', edgecolor='black')
        axes[1].axhline(y=stats['mean_length'], color='red', linestyle='--', 
                        label=f"Mean: {stats['mean_length']:.1f}", linewidth=2)
        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel('Episode Length (steps)', fontsize=12)
        axes[1].set_title('Episode Lengths (preprocessed)', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/phase3/preprocessed_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved preprocessed performance visualization")
        
    except FileNotFoundError:
        print("⚠ Phase 2 baseline not found, skipping comparison")


def save_preprocessing_info(stats):
    """Save preprocessing information and statistics."""
    print("\n" + "="*60)
    print("SAVING PREPROCESSING INFORMATION")
    print("="*60 + "\n")
    
    preprocessing_info = {
        'preprocessing_steps': {
            '1_noop_reset': 'Random 1-30 no-op actions on reset for stochasticity',
            '2_frame_skip': 'Skip 4 frames, max-pool over last 2 to handle flickering',
            '3_fire_reset': 'Auto-fire on reset for games requiring it',
            '4_grayscale': 'Convert RGB to grayscale (210x160x3 → 210x160x1)',
            '5_resize': 'Resize to 84x84 as per DQN Nature paper',
            '6_reward_clip': 'Clip rewards to {-1, 0, 1} for stability',
            '7_frame_stack': 'Stack 4 frames to capture motion (84x84x1 → 84x84x4)',
            '8_normalize': 'Scale pixel values from [0,255] to [0,1]'
        },
        'original_observation': {
            'shape': [210, 160, 3],
            'dtype': 'uint8',
            'size_bytes': 210 * 160 * 3,
            'size_kb': 210 * 160 * 3 / 1024
        },
        'preprocessed_observation': {
            'shape': [84, 84, 4],
            'dtype': 'float32',
            'size_bytes': 84 * 84 * 4 * 4,  # float32 = 4 bytes
            'size_kb': 84 * 84 * 4 * 4 / 1024
        },
        'performance_stats': stats,
        'benefits': {
            'reduced_dimensions': '100,800 → 28,224 pixels (72% reduction)',
            'grayscale': 'Removes color complexity, focuses on structure',
            'frame_stacking': 'Captures temporal information (motion)',
            'reward_clipping': 'Stabilizes learning, reduces variance',
            'normalization': 'Helps neural network training'
        }
    }
    
    with open('data/phase3/preprocessing_info.json', 'w') as f:
        json.dump(preprocessing_info, f, indent=2)
    
    print("✓ Saved preprocessing information to data/phase3/preprocessing_info.json")
    
    # Create summary
    summary = f"""
# Phase 3: Preprocessing & Feature Engineering Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Preprocessing Pipeline

1. **NoOp Reset**: Random 1-30 no-op actions on reset
2. **Frame Skip**: Skip 4 frames, max-pool over last 2
3. **Fire Reset**: Auto-fire on reset (if needed)
4. **Grayscale**: RGB → Grayscale conversion
5. **Resize**: 210×160 → 84×84 (DQN Nature standard)
6. **Reward Clipping**: Rewards → {{-1, 0, 1}}
7. **Frame Stacking**: Stack 4 consecutive frames
8. **Normalization**: [0, 255] → [0, 1]

## Observation Transformation

**Original:**
- Shape: 210 × 160 × 3 (RGB)
- Type: uint8
- Size: {preprocessing_info['original_observation']['size_kb']:.2f} KB

**Preprocessed:**
- Shape: 84 × 84 × 4 (Stacked Grayscale)
- Type: float32
- Size: {preprocessing_info['preprocessed_observation']['size_kb']:.2f} KB

## Key Benefits

- ✅ 72% reduction in pixel dimensions
- ✅ Temporal information captured (frame stacking)
- ✅ Training stability (reward clipping)
- ✅ Better neural network convergence (normalization)
- ✅ Standard preprocessing for DQN-family algorithms

## Performance with Random Agent

- Mean Reward (clipped): {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}
- Mean Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}

**Note**: Rewards are clipped, so direct comparison with Phase 2 baseline 
requires unclipped reward tracking (available in environment info).
"""
    
    with open('data/phase3/summary.md', 'w') as f:
        f.write(summary)
    
    print("✓ Saved summary to data/phase3/summary.md")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("PHASE 3: PREPROCESSING & FEATURE ENGINEERING")
    print("="*60)
    
    # Create directories
    create_output_dirs()
    
    # Compare raw vs preprocessed
    obs_raw, obs_preprocessed = compare_raw_vs_preprocessed()
    
    # Visualize preprocessing steps
    visualize_preprocessing_steps()
    
    # Test preprocessed environment
    stats = test_preprocessed_environment(num_episodes=20)
    
    # Create visualizations
    create_comparison_visualization(stats)
    
    # Save information
    save_preprocessing_info(stats)
    
    print("\n" + "="*60)
    print("✅ PHASE 3 COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  Code:")
    print("    - src/preprocessing.py (reusable wrapper classes)")
    print("  Data:")
    print("    - data/phase3/preprocessing_info.json")
    print("    - data/phase3/summary.md")
    print("  Visualizations:")
    print("    - outputs/phase3/preprocessing_steps.png")
    print("    - outputs/phase3/preprocessed_performance.png")
    print("\nKey Takeaways:")
    print("  • Preprocessing reduces dimensions by 72%")
    print("  • Frame stacking captures temporal dynamics")
    print("  • Reward clipping stabilizes training")
    print("  • Environment ready for DQN training!")
    print("\nNext Steps:")
    print("  Ready to move to Phase 4: DQN Agent - Part 1 (Basic Implementation)")
    print("\n")


if __name__ == "__main__":
    main()
