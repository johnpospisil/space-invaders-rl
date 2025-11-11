"""
Phase 4: DQN Agent - Part 1 (Basic Implementation)

This script implements and trains a basic DQN agent:
1. Creates DQN agent with CNN architecture
2. Implements experience replay
3. Trains agent for 1M steps
4. Saves checkpoints and training metrics
5. Evaluates agent performance
"""

import gymnasium as gym
import ale_py
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocessing import make_atari_env
from dqn_agent import DQNAgent
from replay_buffer import ReplayBuffer

# Register ALE environments
gym.register_envs(ale_py)


def create_output_dirs():
    """Create necessary output directories."""
    dirs = ['outputs/phase4', 'data/phase4', 'models/phase4']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Created output directories")


def train_dqn(
    num_steps=1_000_000,
    batch_size=32,
    buffer_size=100_000,
    learning_starts=50_000,
    target_update_freq=10_000,
    save_freq=100_000,
    eval_freq=50_000,
    eval_episodes=10,
    device='cpu'
):
    """
    Train DQN agent.
    
    Args:
        num_steps: Total training steps
        batch_size: Batch size for training
        buffer_size: Size of replay buffer
        learning_starts: Steps before training starts
        target_update_freq: Steps between target network updates
        save_freq: Steps between saving checkpoints
        eval_freq: Steps between evaluations
        eval_episodes: Number of episodes for evaluation
        device: Device to train on
    """
    print("\n" + "="*60)
    print("PHASE 4: DQN AGENT TRAINING")
    print("="*60 + "\n")
    
    # Create environment
    env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=True)
    eval_env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=False)
    
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    print(f"Environment: ALE/SpaceInvaders-v5")
    print(f"Observation shape: {observation_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Device: {device}\n")
    
    # Create agent
    agent = DQNAgent(
        observation_shape=observation_shape,
        n_actions=n_actions,
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9999,  # Slower decay for 1M steps
        device=device
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=buffer_size,
        observation_shape=observation_shape,
        device=device
    )
    
    print(f"Agent created:")
    print(f"  Learning rate: 1e-4")
    print(f"  Gamma: 0.99")
    print(f"  Epsilon: {agent.epsilon:.2f} → {agent.epsilon_end}")
    print(f"  Replay buffer size: {buffer_size:,}\n")
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    training_losses = []
    eval_rewards = []
    eval_steps = []
    
    # Training loop
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    print(f"Starting training for {num_steps:,} steps...")
    print(f"Learning starts after {learning_starts:,} steps\n")
    
    pbar = tqdm(range(num_steps), desc="Training")
    
    for step in pbar:
        # Select and perform action
        action = agent.select_action(obs, training=True)
        next_obs, reward, done, truncated, _ = env.step(action)
        
        # Store transition
        replay_buffer.push(obs, action, reward, next_obs, done or truncated)
        
        episode_reward += reward
        episode_length += 1
        
        obs = next_obs
        
        # Train agent
        if step >= learning_starts and replay_buffer.is_ready(batch_size):
            batch = replay_buffer.sample(batch_size)
            loss = agent.train_step(batch)
            training_losses.append(loss)
            
            # Decay epsilon
            agent.decay_epsilon()
        
        # Update target network
        if step > 0 and step % target_update_freq == 0:
            agent.update_target_network()
        
        # Episode end
        if done or truncated:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
        
        # Evaluation
        if step > 0 and step % eval_freq == 0:
            eval_reward = evaluate_agent(agent, eval_env, eval_episodes)
            eval_rewards.append(eval_reward)
            eval_steps.append(step)
            
            # Update progress bar
            pbar.set_postfix({
                'eval_reward': f'{eval_reward:.1f}',
                'epsilon': f'{agent.epsilon:.3f}',
                'buffer': f'{len(replay_buffer):,}'
            })
        
        # Save checkpoint
        if step > 0 and step % save_freq == 0:
            checkpoint_path = f'models/phase4/dqn_checkpoint_{step}.pt'
            agent.save(checkpoint_path)
    
    # Final save
    agent.save('models/phase4/dqn_final.pt')
    
    # Close environments
    env.close()
    eval_env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'training_losses': training_losses,
        'eval_rewards': eval_rewards,
        'eval_steps': eval_steps
    }


def evaluate_agent(agent, env, num_episodes=10):
    """
    Evaluate agent without exploration.
    
    Args:
        agent: DQN agent
        env: Environment (without reward clipping for true performance)
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Mean episode reward
    """
    episode_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(obs, training=False)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
    
    return np.mean(episode_rewards)


def save_training_results(results):
    """Save training results and create visualizations."""
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60 + "\n")
    
    # Save metrics
    metrics = {
        'num_episodes': len(results['episode_rewards']),
        'final_eval_reward': float(results['eval_rewards'][-1]) if results['eval_rewards'] else 0,
        'max_eval_reward': float(max(results['eval_rewards'])) if results['eval_rewards'] else 0,
        'eval_rewards': [float(r) for r in results['eval_rewards']],
        'eval_steps': [int(s) for s in results['eval_steps']],
        'final_episode_reward_mean': float(np.mean(results['episode_rewards'][-100:])),
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/phase4/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("✓ Saved training metrics")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Evaluation rewards
    if results['eval_rewards']:
        axes[0, 0].plot(results['eval_steps'], results['eval_rewards'], 'o-', linewidth=2)
        axes[0, 0].set_xlabel('Training Steps', fontsize=12)
        axes[0, 0].set_ylabel('Mean Eval Reward', fontsize=12)
        axes[0, 0].set_title('Evaluation Performance Over Training', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training episode rewards (moving average)
    if results['episode_rewards']:
        window = 100
        episodes = range(len(results['episode_rewards']))
        rewards_ma = np.convolve(results['episode_rewards'], 
                                 np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], rewards_ma, linewidth=2)
        axes[0, 1].set_xlabel('Episode', fontsize=12)
        axes[0, 1].set_ylabel('Reward (100-ep MA)', fontsize=12)
        axes[0, 1].set_title('Training Episode Rewards', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training loss
    if results['training_losses']:
        window = 1000
        losses = results['training_losses']
        loss_ma = np.convolve(losses, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(loss_ma, linewidth=1, alpha=0.7)
        axes[1, 0].set_xlabel('Training Step', fontsize=12)
        axes[1, 0].set_ylabel('Loss (1000-step MA)', fontsize=12)
        axes[1, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Episode lengths
    if results['episode_lengths']:
        window = 100
        episodes = range(len(results['episode_lengths']))
        lengths_ma = np.convolve(results['episode_lengths'], 
                                 np.ones(window)/window, mode='valid')
        axes[1, 1].plot(episodes[window-1:], lengths_ma, linewidth=2, color='coral')
        axes[1, 1].set_xlabel('Episode', fontsize=12)
        axes[1, 1].set_ylabel('Length (100-ep MA)', fontsize=12)
        axes[1, 1].set_title('Episode Lengths', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('DQN Training Results - Phase 4', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/phase4/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved training visualizations")
    
    print(f"\n📊 TRAINING SUMMARY:")
    print("-" * 60)
    print(f"Total Episodes: {metrics['num_episodes']}")
    print(f"Final Eval Reward: {metrics['final_eval_reward']:.2f}")
    print(f"Max Eval Reward: {metrics['max_eval_reward']:.2f}")
    print(f"Final 100-ep Mean: {metrics['final_episode_reward_mean']:.2f}")


def main():
    """Main execution function."""
    create_output_dirs()
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU (training will be slower)")
    
    # Train agent
    results = train_dqn(
        num_steps=1_000_000,
        batch_size=32,
        buffer_size=100_000,
        learning_starts=50_000,
        target_update_freq=10_000,
        save_freq=100_000,
        eval_freq=50_000,
        eval_episodes=10,
        device=device
    )
    
    # Save results
    save_training_results(results)
    
    print("\n" + "="*60)
    print("✅ PHASE 4 COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  Models:")
    print("    - models/phase4/dqn_final.pt")
    print("    - models/phase4/dqn_checkpoint_*.pt")
    print("  Data:")
    print("    - data/phase4/training_metrics.json")
    print("  Visualizations:")
    print("    - outputs/phase4/training_results.png")
    print("\nNext Steps:")
    print("  Ready to move to Phase 5: DQN Improvements")
    print("  (Double DQN, Dueling Network, Prioritized Replay)")
    print("\n")


if __name__ == "__main__":
    main()
