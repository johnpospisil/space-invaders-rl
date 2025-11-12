"""
Phase 9: Model Interpretability

This script analyzes what the trained PPO agent learned through:
- Saliency maps (gradient-based attention visualization)
- Activation analysis (what CNN layers detect)
- Policy behavior analysis (action preferences)
- Feature importance

Helps understand the agent's decision-making process.
"""

import gymnasium as gym
import ale_py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from collections import defaultdict

import sys
sys.path.insert(0, '.')

from preprocessing import make_atari_env
from ppo_agent import PPOAgent

# Register ALE environments
gym.register_envs(ale_py)


def create_output_dirs():
    """Create necessary output directories."""
    dirs = ['outputs/phase9', 'outputs/phase9/saliency_maps', 
            'outputs/phase9/activations', 'outputs/phase9/policy_analysis']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Created output directories")


def load_trained_agent(model_path='models/phase6/ppo_final.pt'):
    """Load the trained PPO agent."""
    print("\n" + "="*60)
    print("PHASE 9: MODEL INTERPRETABILITY")
    print("="*60 + "\n")
    
    print(f"Loading trained PPO agent from {model_path}...")
    
    # Create environment to get observation shape
    env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4, clip_rewards=False)
    
    observation_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Create and load agent
    agent = PPOAgent(
        observation_shape=observation_shape,
        n_actions=n_actions,
        device='cpu'
    )
    
    agent.load(model_path)
    agent.network.eval()  # Set to evaluation mode
    
    print(f"✓ Loaded agent successfully")
    print(f"  Observation shape: {observation_shape}")
    print(f"  Number of actions: {n_actions}")
    
    return agent, env


def compute_saliency_map(agent, observation, action=None):
    """
    Compute saliency map using gradients.
    
    Shows which pixels the agent "pays attention to" when making decisions.
    
    Args:
        agent: Trained PPO agent
        observation: Current observation (numpy array)
        action: Specific action to compute saliency for (None = use predicted action)
        
    Returns:
        saliency_map: Gradient magnitudes showing important pixels
        predicted_action: Action the agent would take
        action_probs: Action probability distribution
    """
    # Convert to tensor and enable gradient
    obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(agent.device)
    obs_tensor.requires_grad = True
    
    # Forward pass
    with torch.set_grad_enabled(True):
        action_logits, value = agent.network(obs_tensor)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Use specified action or predicted action
        if action is None:
            action = action_probs.argmax(dim=-1).item()
        
        # Get score for the action
        action_score = action_logits[0, action]
        
        # Backward pass to get gradients
        action_score.backward()
    
    # Get saliency map (absolute gradient values)
    saliency = obs_tensor.grad.abs().squeeze().cpu().numpy()
    
    # Aggregate across channels (take max)
    if len(saliency.shape) == 3:
        saliency = np.max(saliency, axis=0)
    
    return saliency, action, action_probs.detach().cpu().numpy()[0]


def visualize_saliency_maps(agent, env, num_steps=10):
    """Generate and save saliency map visualizations."""
    print("\n" + "="*60)
    print("GENERATING SALIENCY MAPS")
    print("="*60 + "\n")
    
    print(f"Collecting {num_steps} timesteps with saliency maps...")
    
    obs, _ = env.reset()
    
    fig, axes = plt.subplots(num_steps, 4, figsize=(16, 4*num_steps))
    if num_steps == 1:
        axes = axes.reshape(1, -1)
    
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    
    for step in range(num_steps):
        # Compute saliency map
        saliency, action, action_probs = compute_saliency_map(agent, obs)
        
        # Get current frame (last frame of stack)
        if len(obs.shape) == 3:  # (H, W, C)
            current_frame = obs[:, :, -1]
        else:  # (C, H, W)
            current_frame = obs[-1, :, :]
        
        # Plot original frame
        axes[step, 0].imshow(current_frame, cmap='gray')
        axes[step, 0].set_title(f'Step {step+1}: Original Frame', fontweight='bold')
        axes[step, 0].axis('off')
        
        # Plot saliency map
        im = axes[step, 1].imshow(saliency, cmap='hot', interpolation='bilinear')
        axes[step, 1].set_title(f'Saliency Map', fontweight='bold')
        axes[step, 1].axis('off')
        plt.colorbar(im, ax=axes[step, 1], fraction=0.046)
        
        # Plot overlay
        axes[step, 2].imshow(current_frame, cmap='gray', alpha=0.7)
        axes[step, 2].imshow(saliency, cmap='hot', alpha=0.5, interpolation='bilinear')
        axes[step, 2].set_title(f'Overlay', fontweight='bold')
        axes[step, 2].axis('off')
        
        # Plot action probabilities
        axes[step, 3].barh(action_names, action_probs, color='steelblue', alpha=0.8)
        axes[step, 3].barh(action_names[action], action_probs[action], 
                          color='red', alpha=0.9, label='Selected')
        axes[step, 3].set_xlabel('Probability', fontweight='bold')
        axes[step, 3].set_title(f'Action: {action_names[action]}', fontweight='bold')
        axes[step, 3].set_xlim(0, 1)
        axes[step, 3].legend(fontsize=8)
        axes[step, 3].grid(True, alpha=0.3, axis='x')
        
        # Take action
        obs, reward, done, truncated, _ = env.step(action)
        
        if done or truncated:
            obs, _ = env.reset()
    
    plt.suptitle('Saliency Maps: What the Agent Pays Attention To', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/phase9/saliency_maps/saliency_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved saliency map visualizations")


def analyze_cnn_activations(agent, env, num_samples=5):
    """Analyze what CNN layers detect."""
    print("\n" + "="*60)
    print("ANALYZING CNN ACTIVATIONS")
    print("="*60 + "\n")
    
    print(f"Analyzing activations from {num_samples} game states...")
    
    # Collect observations
    observations = []
    obs, _ = env.reset()
    
    for _ in range(num_samples):
        observations.append(obs.copy())
        action = agent.select_action(obs, training=False)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()
    
    # Hook to capture activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks on conv layers
    agent.network.conv1.register_forward_hook(get_activation('conv1'))
    agent.network.conv2.register_forward_hook(get_activation('conv2'))
    agent.network.conv3.register_forward_hook(get_activation('conv3'))
    
    # Forward pass with first observation
    obs_tensor = torch.FloatTensor(observations[0]).unsqueeze(0).to(agent.device)
    _ = agent.network(obs_tensor)
    
    # Visualize activations
    fig, axes = plt.subplots(3, 8, figsize=(20, 9))
    
    layer_names = ['conv1', 'conv2', 'conv3']
    
    for layer_idx, layer_name in enumerate(layer_names):
        act = activations[layer_name].squeeze().cpu().numpy()
        
        # Show first 8 filters
        num_filters = min(8, act.shape[0])
        
        for i in range(num_filters):
            axes[layer_idx, i].imshow(act[i], cmap='viridis')
            axes[layer_idx, i].axis('off')
            if layer_idx == 0:
                axes[layer_idx, i].set_title(f'Filter {i+1}', fontsize=10)
        
        # Hide unused subplots
        for i in range(num_filters, 8):
            axes[layer_idx, i].axis('off')
        
        # Add layer label
        axes[layer_idx, 0].set_ylabel(f'{layer_name}\n({act.shape[0]} filters)', 
                                      fontsize=11, fontweight='bold', rotation=0, 
                                      ha='right', va='center')
    
    plt.suptitle('CNN Layer Activations: What Features Are Detected', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/phase9/activations/cnn_activations.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved CNN activation visualizations")


def analyze_policy_behavior(agent, env, num_episodes=10):
    """Analyze policy behavior and action preferences."""
    print("\n" + "="*60)
    print("ANALYZING POLICY BEHAVIOR")
    print("="*60 + "\n")
    
    print(f"Running {num_episodes} episodes to analyze behavior...")
    
    action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    
    # Collect statistics
    action_counts = defaultdict(int)
    action_values = defaultdict(list)
    episode_rewards = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Get action and value
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                action_logits, value = agent.network(obs_tensor)
                action_probs = F.softmax(action_logits, dim=-1)
                action = action_probs.argmax(dim=-1).item()
            
            action_counts[action] += 1
            action_values[action].append(value.item())
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Action frequency
    actions = list(range(len(action_names)))
    counts = [action_counts[a] for a in actions]
    
    axes[0, 0].bar([action_names[a] for a in actions], counts, 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Action Frequency Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    total = sum(counts)
    for i, (action, count) in enumerate(zip([action_names[a] for a in actions], counts)):
        pct = (count / total) * 100 if total > 0 else 0
        axes[0, 0].text(i, count, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Value estimates by action
    box_data = [action_values[a] for a in actions if len(action_values[a]) > 0]
    box_labels = [action_names[a] for a in actions if len(action_values[a]) > 0]
    
    bp = axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', alpha=0.7),
                            medianprops=dict(color='red', linewidth=2))
    axes[0, 1].set_ylabel('Value Estimate', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Value Estimates by Action', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Episode rewards
    axes[1, 0].hist(episode_rewards, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(episode_rewards):.1f}')
    axes[1, 0].set_xlabel('Episode Reward', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'Episode Rewards (n={num_episodes})', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Episode lengths
    axes[1, 1].hist(episode_lengths, bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(np.mean(episode_lengths), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(episode_lengths):.1f}')
    axes[1, 1].set_xlabel('Episode Length (steps)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'Episode Lengths (n={num_episodes})', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Policy Behavior Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/phase9/policy_analysis/policy_behavior.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved policy behavior analysis")
    
    # Print summary
    print(f"\nBehavior Summary:")
    print(f"  Mean episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"\nAction Preferences:")
    for action in actions:
        if counts[action] > 0:
            pct = (counts[action] / total) * 100
            print(f"  {action_names[action]:<12}: {counts[action]:>5} ({pct:>5.1f}%)")


def create_interpretability_summary(agent, env):
    """Create comprehensive interpretability summary."""
    print("\n" + "="*60)
    print("CREATING INTERPRETABILITY SUMMARY")
    print("="*60 + "\n")
    
    # Run a single episode and capture key moments
    obs, _ = env.reset()
    done = False
    truncated = False
    step = 0
    total_reward = 0
    
    key_moments = []
    
    while not (done or truncated) and step < 100:
        # Compute saliency
        saliency, action, action_probs = compute_saliency_map(agent, obs)
        
        # Get current frame
        if len(obs.shape) == 3:
            current_frame = obs[:, :, -1]
        else:
            current_frame = obs[-1, :, :]
        
        # Take action
        next_obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        
        # Save key moments (when reward is non-zero or high confidence action)
        max_prob = np.max(action_probs)
        if reward != 0 or max_prob > 0.9:
            key_moments.append({
                'step': step,
                'frame': current_frame,
                'saliency': saliency,
                'action': action,
                'action_probs': action_probs,
                'reward': reward,
                'confidence': max_prob
            })
        
        obs = next_obs
        step += 1
    
    # Visualize key moments
    if key_moments:
        num_moments = min(5, len(key_moments))
        fig, axes = plt.subplots(num_moments, 3, figsize=(15, 5*num_moments))
        if num_moments == 1:
            axes = axes.reshape(1, -1)
        
        action_names = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        
        for i, moment in enumerate(key_moments[:num_moments]):
            # Frame
            axes[i, 0].imshow(moment['frame'], cmap='gray')
            axes[i, 0].set_title(f"Step {moment['step']}: Reward={moment['reward']:.1f}", 
                               fontweight='bold')
            axes[i, 0].axis('off')
            
            # Saliency overlay
            axes[i, 1].imshow(moment['frame'], cmap='gray', alpha=0.7)
            axes[i, 1].imshow(moment['saliency'], cmap='hot', alpha=0.5, interpolation='bilinear')
            axes[i, 1].set_title(f"Attention Map", fontweight='bold')
            axes[i, 1].axis('off')
            
            # Action
            action = moment['action']
            axes[i, 2].barh(action_names, moment['action_probs'], color='steelblue', alpha=0.6)
            axes[i, 2].barh(action_names[action], moment['action_probs'][action], 
                          color='red', alpha=0.9)
            axes[i, 2].set_xlabel('Probability')
            axes[i, 2].set_title(f"Action: {action_names[action]} ({moment['confidence']:.2f})", 
                               fontweight='bold')
            axes[i, 2].set_xlim(0, 1)
        
        plt.suptitle(f'Key Decision Moments (Episode Reward: {total_reward:.0f})', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('outputs/phase9/key_moments.png', dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved key moments visualization ({num_moments} moments)")
    
    print(f"  Episode reward: {total_reward:.0f}")
    print(f"  Total steps: {step}")


def main():
    """Main execution function."""
    create_output_dirs()
    
    # Load trained agent
    agent, env = load_trained_agent()
    
    # Generate saliency maps
    visualize_saliency_maps(agent, env, num_steps=5)
    
    # Analyze CNN activations
    analyze_cnn_activations(agent, env, num_samples=3)
    
    # Analyze policy behavior
    analyze_policy_behavior(agent, env, num_episodes=20)
    
    # Create summary
    create_interpretability_summary(agent, env)
    
    env.close()
    
    print("\n" + "="*60)
    print("✅ PHASE 9 COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  Saliency Maps:")
    print("    - outputs/phase9/saliency_maps/saliency_analysis.png")
    print("  CNN Activations:")
    print("    - outputs/phase9/activations/cnn_activations.png")
    print("  Policy Analysis:")
    print("    - outputs/phase9/policy_analysis/policy_behavior.png")
    print("  Summary:")
    print("    - outputs/phase9/key_moments.png")
    print("\nKey Insights:")
    print("  - Saliency maps show what pixels influence decisions")
    print("  - CNN activations reveal learned features")
    print("  - Policy analysis shows action preferences")
    print("\nNext: Phase 10 (Deployment & Documentation)")
    print("\n")


if __name__ == "__main__":
    main()
