"""
Phase 7: Hyperparameter Optimization

This script uses Optuna to systematically tune hyperparameters for PPO
to improve upon the current best performance (561.50 max reward).

Optuna uses Bayesian optimization (TPE - Tree-structured Parzen Estimator)
to efficiently search the hyperparameter space.
"""

import gymnasium as gym
import ale_py
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from preprocessing import make_atari_env
from ppo_agent import PPOAgent

# Register ALE environments
gym.register_envs(ale_py)


def create_output_dirs():
    """Create necessary output directories."""
    dirs = ['outputs/phase7', 'data/phase7', 'models/phase7']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Created output directories")


def objective(trial):
    """
    Optuna objective function to maximize evaluation reward.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Mean evaluation reward across training
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
    clip_epsilon = trial.suggest_float('clip_epsilon', 0.1, 0.3)
    value_coef = trial.suggest_float('value_coef', 0.25, 1.0)
    entropy_coef = trial.suggest_float('entropy_coef', 0.001, 0.05, log=True)
    max_grad_norm = trial.suggest_float('max_grad_norm', 0.3, 1.0)
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    n_epochs = trial.suggest_int('n_epochs', 3, 10)
    
    # Create environment
    env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=True)
    eval_env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=False)
    
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create PPO agent with suggested hyperparameters
    agent = PPOAgent(
        observation_shape=observation_shape,
        n_actions=n_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_epsilon=clip_epsilon,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        max_grad_norm=max_grad_norm,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_steps=n_steps,
        device=device
    )
    
    # Training configuration (reduced for optimization speed)
    num_steps = 200_000  # Shorter training for faster optimization
    eval_freq = 50_000
    eval_episodes = 5
    
    # Training metrics
    eval_rewards = []
    
    # Training loop
    obs, _ = env.reset()
    total_steps = 0
    
    while total_steps < num_steps:
        # Collect rollout
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            agent.buffer.add(obs, action, log_prob, reward, value, done or truncated)
            
            total_steps += 1
            obs = next_obs
            
            if done or truncated:
                obs, _ = env.reset()
            
            # Evaluation
            if total_steps % eval_freq == 0:
                eval_reward = evaluate_agent(agent, eval_env, eval_episodes)
                eval_rewards.append(eval_reward)
                
                # Report intermediate value for pruning
                trial.report(eval_reward, total_steps)
                
                # Prune unpromising trials
                if trial.should_prune():
                    env.close()
                    eval_env.close()
                    raise optuna.TrialPruned()
            
            if total_steps >= num_steps:
                break
        
        # PPO update
        if len(agent.buffer) > 0:
            agent.update()
    
    # Final evaluation
    final_eval = evaluate_agent(agent, eval_env, num_episodes=10)
    eval_rewards.append(final_eval)
    
    env.close()
    eval_env.close()
    
    # Return mean of all evaluations as the objective value
    mean_reward = np.mean(eval_rewards)
    
    return mean_reward


def evaluate_agent(agent, env, num_episodes=5):
    """Evaluate agent without exploration."""
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


def run_optimization(n_trials=20, study_name='ppo_optimization'):
    """
    Run hyperparameter optimization study.
    
    Args:
        n_trials: Number of trials to run
        study_name: Name of the study
    """
    print("\n" + "="*60)
    print("PHASE 7: HYPERPARAMETER OPTIMIZATION")
    print("="*60 + "\n")
    
    print(f"Optimization Configuration:")
    print(f"  Algorithm: PPO (current best: 561.50)")
    print(f"  Number of trials: {n_trials}")
    print(f"  Training steps per trial: 200,000")
    print(f"  Optimization method: TPE (Bayesian)")
    print(f"  Pruning: Median pruner (stops bad trials early)")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
    
    print(f"Hyperparameters to tune:")
    print(f"  - learning_rate: [1e-5, 1e-3]")
    print(f"  - gamma: [0.95, 0.999]")
    print(f"  - gae_lambda: [0.9, 0.99]")
    print(f"  - clip_epsilon: [0.1, 0.3]")
    print(f"  - value_coef: [0.25, 1.0]")
    print(f"  - entropy_coef: [0.001, 0.05]")
    print(f"  - max_grad_norm: [0.3, 1.0]")
    print(f"  - n_steps: [1024, 2048, 4096]")
    print(f"  - batch_size: [128, 256, 512]")
    print(f"  - n_epochs: [3, 10]\n")
    
    # Create study
    study = optuna.create_study(
        study_name=study_name,
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50000)
    )
    
    # Run optimization
    print(f"Starting optimization with {n_trials} trials...")
    print(f"Expected time: ~{n_trials * 10} minutes\n")
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


def save_optimization_results(study):
    """Save optimization results and best hyperparameters."""
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60 + "\n")
    
    # Best trial
    best_trial = study.best_trial
    
    print(f"Best Trial:")
    print(f"  Trial number: {best_trial.number}")
    print(f"  Mean evaluation reward: {best_trial.value:.2f}")
    print(f"\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save results
    results = {
        'best_trial_number': best_trial.number,
        'best_mean_reward': float(best_trial.value),
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
        'baseline_reward': 561.50,  # Phase 6 best
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/phase7/optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Saved optimization results")
    
    # Save all trials for analysis
    trials_data = []
    for trial in study.trials:
        trial_data = {
            'number': trial.number,
            'value': float(trial.value) if trial.value is not None else None,
            'params': trial.params,
            'state': trial.state.name
        }
        trials_data.append(trial_data)
    
    with open('data/phase7/all_trials.json', 'w') as f:
        json.dump(trials_data, f, indent=2)
    
    print("✓ Saved all trial data")
    
    # Improvement analysis
    if best_trial.value > 561.50:
        improvement = ((best_trial.value - 561.50) / 561.50) * 100
        print(f"\n🎉 IMPROVEMENT FOUND!")
        print(f"  Baseline (Phase 6): 561.50")
        print(f"  Optimized: {best_trial.value:.2f}")
        print(f"  Improvement: +{improvement:.1f}%")
    else:
        print(f"\n📊 Results:")
        print(f"  Baseline (Phase 6): 561.50")
        print(f"  Optimized: {best_trial.value:.2f}")
        print(f"  Note: Short training (200k steps) for optimization speed")
        print(f"  Recommendation: Train best config for 1M steps")
    
    # Create visualization
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Optimization history
        trials = [t.number for t in study.trials if t.value is not None]
        values = [t.value for t in study.trials if t.value is not None]
        axes[0, 0].plot(trials, values, 'o-', linewidth=2, markersize=8, alpha=0.7)
        axes[0, 0].axhline(561.50, color='red', linestyle='--', linewidth=2, 
                          label='Baseline (Phase 6)', alpha=0.7)
        axes[0, 0].axhline(best_trial.value, color='green', linestyle='--', 
                          linewidth=2, label=f'Best: {best_trial.value:.1f}', alpha=0.7)
        axes[0, 0].set_xlabel('Trial', fontsize=12)
        axes[0, 0].set_ylabel('Mean Evaluation Reward', fontsize=12)
        axes[0, 0].set_title('Optimization History', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Parameter importance (if enough trials)
        if len(study.trials) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())[:8]  # Top 8
                importances = [importance[p] for p in params]
                
                axes[0, 1].barh(params, importances, color='steelblue', alpha=0.8)
                axes[0, 1].set_xlabel('Importance', fontsize=12)
                axes[0, 1].set_title('Hyperparameter Importance', fontsize=14, fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3, axis='x')
            except:
                axes[0, 1].text(0.5, 0.5, 'Not enough trials\nfor importance analysis', 
                               ha='center', va='center', fontsize=12)
                axes[0, 1].set_xlim(0, 1)
                axes[0, 1].set_ylim(0, 1)
        
        # Plot 3: Learning rate vs reward
        lr_trials = [(t.params['learning_rate'], t.value) for t in study.trials 
                     if t.value is not None]
        if lr_trials:
            lrs, rewards = zip(*lr_trials)
            axes[1, 0].scatter(lrs, rewards, s=100, alpha=0.6, c=range(len(lrs)), 
                              cmap='viridis')
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_xlabel('Learning Rate', fontsize=12)
            axes[1, 0].set_ylabel('Mean Reward', fontsize=12)
            axes[1, 0].set_title('Learning Rate Impact', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Clip epsilon vs reward
        clip_trials = [(t.params['clip_epsilon'], t.value) for t in study.trials 
                       if t.value is not None]
        if clip_trials:
            clips, rewards = zip(*clip_trials)
            axes[1, 1].scatter(clips, rewards, s=100, alpha=0.6, c=range(len(clips)), 
                              cmap='plasma')
            axes[1, 1].set_xlabel('Clip Epsilon', fontsize=12)
            axes[1, 1].set_ylabel('Mean Reward', fontsize=12)
            axes[1, 1].set_title('Clip Epsilon Impact', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('PPO Hyperparameter Optimization - Phase 7', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/phase7/optimization_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Saved optimization visualizations")
    except Exception as e:
        print(f"⚠ Could not create visualizations: {e}")
    
    return results


def train_best_config(best_params, num_steps=1_000_000):
    """
    Train agent with best hyperparameters for full duration.
    
    Args:
        best_params: Dictionary of best hyperparameters
        num_steps: Total training steps
    """
    print("\n" + "="*60)
    print("TRAINING WITH BEST HYPERPARAMETERS")
    print("="*60 + "\n")
    
    # Create environment
    env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=True)
    eval_env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=False)
    
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create agent with best hyperparameters
    agent = PPOAgent(
        observation_shape=observation_shape,
        n_actions=n_actions,
        learning_rate=best_params['learning_rate'],
        gamma=best_params['gamma'],
        gae_lambda=best_params['gae_lambda'],
        clip_epsilon=best_params['clip_epsilon'],
        value_coef=best_params['value_coef'],
        entropy_coef=best_params['entropy_coef'],
        max_grad_norm=best_params['max_grad_norm'],
        n_epochs=best_params['n_epochs'],
        batch_size=best_params['batch_size'],
        n_steps=best_params['n_steps'],
        device=device
    )
    
    print(f"Training Configuration:")
    print(f"  Total steps: {num_steps:,}")
    print(f"  Evaluation frequency: 50,000 steps")
    print(f"  Device: {device}\n")
    
    print("Best Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print()
    
    # Training loop
    from tqdm import tqdm
    
    eval_rewards = []
    eval_steps = []
    
    obs, _ = env.reset()
    total_steps = 0
    n_steps = best_params['n_steps']
    
    print(f"Starting training for {num_steps:,} steps...\n")
    pbar = tqdm(total=num_steps, desc="Training")
    
    while total_steps < num_steps:
        # Collect rollout
        for _ in range(n_steps):
            action, log_prob, value = agent.select_action(obs, training=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            agent.buffer.add(obs, action, log_prob, reward, value, done or truncated)
            
            total_steps += 1
            obs = next_obs
            
            if done or truncated:
                obs, _ = env.reset()
            
            # Evaluation
            if total_steps % 50_000 == 0:
                eval_reward = evaluate_agent(agent, eval_env, num_episodes=10)
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
    final_eval = evaluate_agent(agent, eval_env, num_episodes=20)
    
    # Save model
    agent.save('models/phase7/ppo_optimized.pt')
    
    # Save results
    results = {
        'num_steps': num_steps,
        'final_eval_reward': float(final_eval),
        'max_eval_reward': float(max(eval_rewards)) if eval_rewards else 0,
        'eval_rewards': [float(r) for r in eval_rewards],
        'eval_steps': [int(s) for s in eval_steps],
        'hyperparameters': best_params,
        'baseline_phase6': 561.50,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/phase7/optimized_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    env.close()
    eval_env.close()
    
    print(f"\n✓ Training complete!")
    print(f"  Final eval reward: {final_eval:.2f}")
    print(f"  Max eval reward: {results['max_eval_reward']:.2f}")
    print(f"  Baseline (Phase 6): 561.50")
    
    if results['max_eval_reward'] > 561.50:
        improvement = ((results['max_eval_reward'] - 561.50) / 561.50) * 100
        print(f"  🎉 Improvement: +{improvement:.1f}%")
    
    return results


def main():
    """Main execution function."""
    create_output_dirs()
    
    # Check for Optuna
    try:
        import optuna
        print("✓ Optuna available for optimization")
    except ImportError:
        print("❌ Optuna not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'optuna'])
        print("✓ Optuna installed")
    
    # Run optimization
    study = run_optimization(n_trials=20, study_name='ppo_phase7')
    
    # Save results
    results = save_optimization_results(study)
    
    print("\n" + "="*60)
    print("✅ PHASE 7 OPTIMIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  Data:")
    print("    - data/phase7/optimization_results.json")
    print("    - data/phase7/all_trials.json")
    print("  Visualizations:")
    print("    - outputs/phase7/optimization_results.png")
    print("\nNext Steps:")
    print("  1. Review best hyperparameters")
    print("  2. Optional: Run train_best_config() for 1M steps")
    print("  3. Move to Phase 8: Advanced Analysis")
    print("\n")


if __name__ == "__main__":
    main()
