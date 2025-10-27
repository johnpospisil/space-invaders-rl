"""
Phase 1: Environment Setup & Exploration

This script sets up the Space Invaders environment and runs basic exploration:
1. Loads the SpaceInvaders-v4 environment
2. Runs a random agent for a few episodes
3. Displays environment statistics
4. Saves sample frames for visualization
"""

import gymnasium as gym
import ale_py  # Import ALE to register Atari environments
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Register ALE environments
gym.register_envs(ale_py)


def create_output_dirs():
    """Create necessary output directories."""
    dirs = ['outputs/phase1', 'data/phase1', 'outputs/phase1/frames']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Created output directories")


def explore_environment():
    """Load and explore the Space Invaders environment."""
    print("\n" + "="*60)
    print("PHASE 1: ENVIRONMENT SETUP & EXPLORATION")
    print("="*60 + "\n")
    
    # Create environment
    env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
    print(f"✓ Loaded environment: {env.spec.id}")
    
    # Get environment information
    obs, info = env.reset()
    
    print("\n📊 ENVIRONMENT STATISTICS:")
    print("-" * 60)
    print(f"Observation Space: {env.observation_space}")
    print(f"Observation Shape: {obs.shape}")
    print(f"Observation Type: {obs.dtype}")
    print(f"Observation Range: [{obs.min()}, {obs.max()}]")
    print(f"\nAction Space: {env.action_space}")
    print(f"Number of Actions: {env.action_space.n}")
    
    # Get action meanings
    try:
        action_meanings = env.unwrapped.get_action_meanings()
        print(f"\nAction Meanings:")
        for i, action in enumerate(action_meanings):
            print(f"  {i}: {action}")
    except:
        print("\nAction meanings not available")
    
    # Save environment info to JSON
    env_info = {
        'environment': env.spec.id,
        'observation_shape': list(obs.shape),
        'observation_dtype': str(obs.dtype),
        'action_space_size': int(env.action_space.n),
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        env_info['action_meanings'] = action_meanings
    except:
        pass
    
    with open('data/phase1/environment_info.json', 'w') as f:
        json.dump(env_info, f, indent=2)
    
    print("\n✓ Saved environment info to data/phase1/environment_info.json")
    
    return env, obs


def run_random_agent(env, num_episodes=5):
    """Run a random agent for a few episodes."""
    print("\n" + "="*60)
    print("RUNNING RANDOM AGENT")
    print("="*60 + "\n")
    
    episode_rewards = []
    episode_lengths = []
    frames_to_save = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        # Save first frame of first episode
        if episode == 0:
            frames_to_save.append(('initial', obs.copy()))
        
        while not (done or truncated):
            # Take random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Save some frames from first episode
            if episode == 0 and episode_length % 50 == 0:
                frames_to_save.append((f'step_{episode_length}', obs.copy()))
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward = {episode_reward:.1f}, Length = {episode_length}")
    
    env.close()
    
    # Calculate statistics
    stats = {
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'min_length': int(np.min(episode_lengths)),
        'max_length': int(np.max(episode_lengths)),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"\n📈 RANDOM AGENT STATISTICS:")
    print("-" * 60)
    print(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"Reward Range: [{stats['min_reward']:.1f}, {stats['max_reward']:.1f}]")
    print(f"Mean Episode Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print(f"Length Range: [{stats['min_length']}, {stats['max_length']}]")
    
    # Save statistics
    with open('data/phase1/random_agent_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n✓ Saved statistics to data/phase1/random_agent_stats.json")
    
    return stats, frames_to_save


def visualize_frames(frames_to_save):
    """Save sample frames as images."""
    print("\n" + "="*60)
    print("SAVING SAMPLE FRAMES")
    print("="*60 + "\n")
    
    # Create a grid of frames
    num_frames = len(frames_to_save)
    if num_frames == 0:
        print("No frames to save")
        return
    
    # Save individual frames
    for label, frame in frames_to_save:
        plt.figure(figsize=(6, 8))
        plt.imshow(frame)
        plt.title(f'Space Invaders - {label}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'outputs/phase1/frames/frame_{label}.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved frame: {label}")
    
    # Create a comparison grid
    if num_frames >= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, (label, frame) in enumerate(frames_to_save[:4]):
            axes[idx].imshow(frame)
            axes[idx].set_title(label, fontsize=12)
            axes[idx].axis('off')
        
        plt.suptitle('Space Invaders - Sample Frames', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/phase1/frame_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Saved frame comparison grid")
    
    print(f"\n✓ All frames saved to outputs/phase1/frames/")


def create_summary_report(env_info, stats):
    """Create a summary report for Phase 1."""
    print("\n" + "="*60)
    print("CREATING SUMMARY REPORT")
    print("="*60 + "\n")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Episode Rewards
    episodes = range(1, len(stats['episode_rewards']) + 1)
    axes[0].bar(episodes, stats['episode_rewards'], color='steelblue', alpha=0.7)
    axes[0].axhline(y=stats['mean_reward'], color='red', linestyle='--', 
                    label=f'Mean: {stats["mean_reward"]:.1f}', linewidth=2)
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Total Reward', fontsize=12)
    axes[0].set_title('Random Agent - Episode Rewards', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Episode Lengths
    axes[1].bar(episodes, stats['episode_lengths'], color='coral', alpha=0.7)
    axes[1].axhline(y=stats['mean_length'], color='red', linestyle='--', 
                    label=f'Mean: {stats["mean_length"]:.1f}', linewidth=2)
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Episode Length (steps)', fontsize=12)
    axes[1].set_title('Random Agent - Episode Lengths', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/phase1/summary_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved summary report to outputs/phase1/summary_report.png")


def main():
    """Main execution function."""
    # Create directories
    create_output_dirs()
    
    # Explore environment
    env, initial_obs = explore_environment()
    
    # Run random agent
    stats, frames = run_random_agent(env, num_episodes=5)
    
    # Visualize frames
    visualize_frames(frames)
    
    # Create summary report
    create_summary_report(None, stats)
    
    print("\n" + "="*60)
    print("✅ PHASE 1 COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Review the saved frames in outputs/phase1/frames/")
    print("2. Check the statistics in data/phase1/")
    print("3. Examine the summary report: outputs/phase1/summary_report.png")
    print("4. Ready to move to Phase 2: Data Collection & Baseline Performance")
    print("\n")


if __name__ == "__main__":
    main()
