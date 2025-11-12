"""
Quick test version of Phase 6 - runs for 100k steps (~30 minutes).
Use this for rapid validation before full training.
"""

import sys
sys.path.insert(0, 'src')

import gymnasium as gym
import ale_py
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from preprocessing import make_atari_env
from ppo_agent import PPOAgent

# Register ALE environments
gym.register_envs(ale_py)


def quick_train_ppo():
    """Quick PPO training for validation."""
    print("\n" + "="*60)
    print("PHASE 6 QUICK TEST: PPO AGENT")
    print("="*60 + "\n")
    
    # Create directories
    Path('models/phase6').mkdir(parents=True, exist_ok=True)
    Path('data/phase6').mkdir(parents=True, exist_ok=True)
    
    # Create environment
    env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=True)
    eval_env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=False)
    
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Quick Test Configuration:")
    print(f"  Training steps: 100,000 (vs 1M full)")
    print(f"  Rollout steps: 2,048")
    print(f"  Eval frequency: 25,000")
    print(f"  Device: {device}")
    print(f"  Expected time: ~30 minutes\n")
    
    # Create agent
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
        n_steps=2048,
        device=device
    )
    
    # Training variables
    num_steps = 100_000
    n_steps = 2048
    eval_freq = 25_000
    
    episode_rewards = []
    eval_rewards = []
    eval_steps = []
    
    obs, _ = env.reset()
    episode_reward = 0
    total_steps = 0
    
    print("Starting quick test training...\n")
    pbar = tqdm(total=num_steps, desc="Training")
    
    while total_steps < num_steps:
        # Collect rollout
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            agent.buffer.add(obs, action, log_prob, reward, value, done or truncated)
            
            episode_reward += reward
            total_steps += 1
            obs = next_obs
            
            if done or truncated:
                episode_rewards.append(episode_reward)
                obs, _ = env.reset()
                episode_reward = 0
            
            # Evaluation
            if total_steps % eval_freq == 0:
                eval_reward = evaluate_agent(agent, eval_env, num_episodes=5)
                eval_rewards.append(eval_reward)
                eval_steps.append(total_steps)
                pbar.set_postfix({'eval_reward': f'{eval_reward:.1f}'})
            
            pbar.update(1)
            
            if total_steps >= num_steps:
                break
        
        # PPO update
        if len(agent.buffer) > 0:
            agent.update()
    
    pbar.close()
    
    # Final evaluation
    final_eval = evaluate_agent(agent, eval_env, num_episodes=10)
    print(f"\n✓ Final evaluation: {final_eval:.2f}")
    
    # Save model
    agent.save('models/phase6/ppo_quick_test.pt')
    print("✓ Model saved")
    
    # Save results
    results = {
        'quick_test': True,
        'num_steps': num_steps,
        'num_episodes': len(episode_rewards),
        'final_eval_reward': float(final_eval),
        'max_eval_reward': float(max(eval_rewards)) if eval_rewards else 0,
        'eval_rewards': [float(r) for r in eval_rewards],
        'eval_steps': [int(s) for s in eval_steps],
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/phase6/quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✓ Results saved")
    
    env.close()
    eval_env.close()
    
    print("\n" + "="*60)
    print("QUICK TEST SUMMARY")
    print("="*60)
    print(f"Episodes completed: {len(episode_rewards)}")
    print(f"Final eval reward: {final_eval:.2f}")
    print(f"Max eval reward: {results['max_eval_reward']:.2f}")
    print(f"Baseline (random): 146.95")
    
    if results['max_eval_reward'] > 146.95:
        improvement = ((results['max_eval_reward'] - 146.95) / 146.95) * 100
        print(f"✓ Improvement over baseline: +{improvement:.1f}%")
    else:
        print("⚠ Below baseline (needs more training)")
    
    print("\nNext: Run full training with phase6_train_ppo.py")
    print("="*60 + "\n")


def evaluate_agent(agent, env, num_episodes=5):
    """Evaluate agent."""
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            action = agent.select_action(obs, training=False)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    return np.mean(rewards)


if __name__ == "__main__":
    quick_train_ppo()
