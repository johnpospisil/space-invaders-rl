"""
Phase 6: Alternative Algorithms - PPO

This script implements and trains a PPO (Proximal Policy Optimization) agent
to compare against the DQN agents from Phases 4 and 5.

PPO is a policy gradient method that:
- Directly optimizes the policy (vs value-based DQN)
- Uses clipped surrogate objective for stable updates
- Collects rollouts and updates in batches
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
from ppo_agent import PPOAgent

# Register ALE environments
gym.register_envs(ale_py)


def create_output_dirs():
    """Create necessary output directories."""
    dirs = ['outputs/phase6', 'data/phase6', 'models/phase6']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Created output directories")


def train_ppo(
    num_steps=1_000_000,
    n_steps=2048,
    eval_freq=50_000,
    eval_episodes=10,
    save_freq=100_000,
    device='cpu'
):
    """
    Train PPO agent.
    
    Args:
        num_steps: Total training steps
        n_steps: Steps per rollout (PPO collects full trajectories)
        eval_freq: Steps between evaluations
        eval_episodes: Number of episodes for evaluation
        save_freq: Steps between saving checkpoints
        device: Device to train on
    """
    print("\n" + "="*60)
    print("PHASE 6: PPO AGENT TRAINING")
    print("="*60 + "\n")
    
    # Create environments
    env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=True)
    eval_env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=False)
    
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    print(f"Environment: ALE/SpaceInvaders-v5")
    print(f"Observation shape: {observation_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Device: {device}\n")
    
    # Create PPO agent
    agent = PPOAgent(
        observation_shape=observation_shape,
        n_actions=n_actions,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=256,
        n_steps=n_steps,
        device=device
    )
    
    print(f"PPO Agent Configuration:")
    print(f"  Algorithm: Proximal Policy Optimization")
    print(f"  Policy: Actor-Critic with shared CNN")
    print(f"  Learning rate: 3e-4")
    print(f"  Gamma: 0.99")
    print(f"  GAE Lambda: 0.95")
    print(f"  Clip epsilon: 0.2")
    print(f"  Rollout steps: {n_steps}")
    print(f"  Mini-batch size: 256")
    print(f"  Epochs per update: 4\n")
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    policy_losses = []
    value_losses = []
    eval_rewards = []
    eval_steps = []
    
    # Training loop
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    total_steps = 0
    
    print(f"Starting training for {num_steps:,} steps...")
    print(f"Collecting {n_steps} steps per rollout\n")
    
    pbar = tqdm(total=num_steps, desc="Training")
    
    while total_steps < num_steps:
        # Collect rollout
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Store in buffer
            agent.buffer.add(obs, action, log_prob, reward, value, done or truncated)
            
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            obs = next_obs
            
            # Episode end
            if done or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                obs, _ = env.reset()
                episode_reward = 0
                episode_length = 0
            
            # Evaluation
            if total_steps % eval_freq == 0:
                eval_reward = evaluate_agent(agent, eval_env, eval_episodes)
                eval_rewards.append(eval_reward)
                eval_steps.append(total_steps)
                
                pbar.set_postfix({
                    'eval_reward': f'{eval_reward:.1f}',
                    'episodes': len(episode_rewards)
                })
            
            # Save checkpoint
            if total_steps % save_freq == 0:
                checkpoint_path = f'models/phase6/ppo_checkpoint_{total_steps}.pt'
                agent.save(checkpoint_path)
            
            pbar.update(1)
            
            if total_steps >= num_steps:
                break
        
        # PPO update after collecting rollout
        if len(agent.buffer) > 0:
            metrics = agent.update()
            policy_losses.append(metrics['policy_loss'])
            value_losses.append(metrics['value_loss'])
    
    pbar.close()
    
    # Final save
    agent.save('models/phase6/ppo_final.pt')
    
    # Close environments
    env.close()
    eval_env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'eval_rewards': eval_rewards,
        'eval_steps': eval_steps
    }


def evaluate_agent(agent, env, num_episodes=10):
    """
    Evaluate agent without exploration.
    
    Args:
        agent: PPO agent
        env: Environment
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
    
    with open('data/phase6/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("✓ Saved training metrics")
    
    # Load previous phases for comparison
    phase4_metrics = None
    phase5_metrics = None
    
    try:
        with open('data/phase4/training_metrics.json', 'r') as f:
            phase4_metrics = json.load(f)
    except:
        pass
    
    try:
        with open('data/phase5/training_metrics.json', 'r') as f:
            phase5_metrics = json.load(f)
    except:
        pass
    
    # Create comparison visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Algorithm comparison
    axes[0, 0].plot(results['eval_steps'], results['eval_rewards'],
                   'o-', linewidth=2, label='PPO (Phase 6)', color='green')
    if phase4_metrics:
        axes[0, 0].plot(phase4_metrics['eval_steps'], phase4_metrics['eval_rewards'],
                       'o-', linewidth=2, label='DQN (Phase 4)', color='orange', alpha=0.7)
    if phase5_metrics:
        axes[0, 0].plot(phase5_metrics['eval_steps'], phase5_metrics['eval_rewards'],
                       'o-', linewidth=2, label='Improved DQN (Phase 5)', color='blue', alpha=0.7)
    axes[0, 0].axhline(146.95, color='red', linestyle='--', linewidth=2,
                      label='Random Baseline', alpha=0.5)
    axes[0, 0].set_xlabel('Training Steps', fontsize=12)
    axes[0, 0].set_ylabel('Mean Eval Reward', fontsize=12)
    axes[0, 0].set_title('Algorithm Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training rewards
    if results['episode_rewards']:
        window = 100
        episodes = range(len(results['episode_rewards']))
        rewards_ma = np.convolve(results['episode_rewards'],
                                 np.ones(window)/window, mode='valid')
        axes[0, 1].plot(episodes[window-1:], rewards_ma, linewidth=2, color='green')
        axes[0, 1].set_xlabel('Episode', fontsize=12)
        axes[0, 1].set_ylabel('Reward (100-ep MA)', fontsize=12)
        axes[0, 1].set_title('PPO Training Progress', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Policy loss
    if results['policy_losses']:
        axes[1, 0].plot(results['policy_losses'], linewidth=1, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('Update', fontsize=12)
        axes[1, 0].set_ylabel('Policy Loss', fontsize=12)
        axes[1, 0].set_title('PPO Policy Loss', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final comparison bar chart
    agents = ['Random\nBaseline']
    max_rewards = [146.95]
    colors = ['red']
    
    if phase4_metrics:
        agents.append('DQN\n(Phase 4)')
        max_rewards.append(phase4_metrics['max_eval_reward'])
        colors.append('orange')
    
    if phase5_metrics:
        agents.append('Improved DQN\n(Phase 5)')
        max_rewards.append(phase5_metrics['max_eval_reward'])
        colors.append('blue')
    
    agents.append('PPO\n(Phase 6)')
    max_rewards.append(metrics['max_eval_reward'])
    colors.append('green')
    
    bars = axes[1, 1].bar(agents, max_rewards, color=colors, alpha=0.8)
    axes[1, 1].set_ylabel('Max Eval Reward', fontsize=12)
    axes[1, 1].set_title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, reward in zip(bars, max_rewards):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{reward:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.suptitle('PPO vs DQN - Phase 6 Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/phase6/training_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved training visualizations")
    
    print(f"\n📊 TRAINING SUMMARY:")
    print("-" * 60)
    print(f"Algorithm: PPO (Policy Gradient)")
    print(f"Total Episodes: {metrics['num_episodes']}")
    print(f"Final Eval Reward: {metrics['final_eval_reward']:.2f}")
    print(f"Max Eval Reward: {metrics['max_eval_reward']:.2f}")
    
    if phase4_metrics:
        improvement_vs_phase4 = ((metrics['max_eval_reward'] - phase4_metrics['max_eval_reward']) /
                                phase4_metrics['max_eval_reward']) * 100
        print(f"\nComparison vs Phase 4 (Basic DQN):")
        print(f"  Phase 4: {phase4_metrics['max_eval_reward']:.2f}")
        print(f"  Phase 6: {metrics['max_eval_reward']:.2f}")
        print(f"  Difference: {improvement_vs_phase4:+.1f}%")
    
    if phase5_metrics:
        improvement_vs_phase5 = ((metrics['max_eval_reward'] - phase5_metrics['max_eval_reward']) /
                                phase5_metrics['max_eval_reward']) * 100
        print(f"\nComparison vs Phase 5 (Improved DQN):")
        print(f"  Phase 5: {phase5_metrics['max_eval_reward']:.2f}")
        print(f"  Phase 6: {metrics['max_eval_reward']:.2f}")
        print(f"  Difference: {improvement_vs_phase5:+.1f}%")


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
    results = train_ppo(
        num_steps=1_000_000,
        n_steps=2048,
        eval_freq=50_000,
        eval_episodes=10,
        save_freq=100_000,
        device=device
    )
    
    # Save results
    save_training_results(results)
    
    print("\n" + "="*60)
    print("✅ PHASE 6 COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  Models:")
    print("    - models/phase6/ppo_final.pt")
    print("    - models/phase6/ppo_checkpoint_*.pt")
    print("  Data:")
    print("    - data/phase6/training_metrics.json")
    print("  Visualizations:")
    print("    - outputs/phase6/training_results.png")
    print("\nKey Insights:")
    print("  - PPO (policy gradient) vs DQN (value-based)")
    print("  - Direct policy optimization vs Q-learning")
    print("  - On-policy (PPO) vs off-policy (DQN)")
    print("\nNext Steps:")
    print("  Ready to move to Phase 7: Hyperparameter Optimization")
    print("  Or skip to Phase 8: Advanced Analysis & Visualization")
    print("\n")


if __name__ == "__main__":
    main()
