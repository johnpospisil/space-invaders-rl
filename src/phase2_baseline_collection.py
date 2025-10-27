"""
Phase 2: Data Collection & Baseline Performance

This script collects comprehensive baseline data by running a random agent for 100 episodes:
1. Runs random agent for 100 episodes
2. Tracks detailed metrics (rewards, lengths, survival time)
3. Creates comprehensive visualizations
4. Generates statistical analysis
5. Saves results for future comparison
"""

import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import scipy.stats as stats

# Register ALE environments
gym.register_envs(ale_py)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)


def create_output_dirs():
    """Create necessary output directories."""
    dirs = ['outputs/phase2', 'data/phase2', 'outputs/phase2/plots']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print("✓ Created output directories")


def collect_baseline_data(num_episodes=100):
    """Collect baseline data from random agent."""
    print("\n" + "="*60)
    print("PHASE 2: BASELINE DATA COLLECTION")
    print("="*60 + "\n")
    
    env = gym.make('ALE/SpaceInvaders-v5', render_mode='rgb_array')
    print(f"✓ Environment created: {env.spec.id}")
    print(f"✓ Running {num_episodes} episodes with random agent...\n")
    
    # Storage for metrics
    episode_data = []
    
    for episode in tqdm(range(num_episodes), desc="Collecting episodes"):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        # Track action distribution
        actions_taken = []
        rewards_per_step = []
        
        while not (done or truncated):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            actions_taken.append(action)
            rewards_per_step.append(reward)
        
        # Store episode data
        episode_data.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'episode_length': episode_length,
            'avg_reward_per_step': episode_reward / episode_length if episode_length > 0 else 0,
            'actions': actions_taken,
            'rewards': rewards_per_step,
            'max_single_reward': max(rewards_per_step) if rewards_per_step else 0,
            'non_zero_rewards': sum(1 for r in rewards_per_step if r > 0)
        })
    
    env.close()
    
    print(f"\n✓ Completed {num_episodes} episodes")
    return episode_data


def analyze_baseline_data(episode_data):
    """Perform comprehensive statistical analysis."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60 + "\n")
    
    # Extract basic metrics
    rewards = [ep['total_reward'] for ep in episode_data]
    lengths = [ep['episode_length'] for ep in episode_data]
    avg_rewards = [ep['avg_reward_per_step'] for ep in episode_data]
    max_rewards = [ep['max_single_reward'] for ep in episode_data]
    non_zero_rewards = [ep['non_zero_rewards'] for ep in episode_data]
    
    # Calculate statistics
    stats_dict = {
        'num_episodes': len(episode_data),
        'timestamp': datetime.now().isoformat(),
        
        # Reward statistics
        'reward_stats': {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'median': float(np.median(rewards)),
            'min': float(np.min(rewards)),
            'max': float(np.max(rewards)),
            'q25': float(np.percentile(rewards, 25)),
            'q75': float(np.percentile(rewards, 75)),
            'iqr': float(np.percentile(rewards, 75) - np.percentile(rewards, 25)),
            'cv': float(np.std(rewards) / np.mean(rewards)) if np.mean(rewards) > 0 else 0
        },
        
        # Episode length statistics
        'length_stats': {
            'mean': float(np.mean(lengths)),
            'std': float(np.std(lengths)),
            'median': float(np.median(lengths)),
            'min': int(np.min(lengths)),
            'max': int(np.max(lengths)),
            'q25': float(np.percentile(lengths, 25)),
            'q75': float(np.percentile(lengths, 75))
        },
        
        # Performance metrics
        'performance_stats': {
            'avg_reward_per_step_mean': float(np.mean(avg_rewards)),
            'avg_reward_per_step_std': float(np.std(avg_rewards)),
            'max_single_reward_mean': float(np.mean(max_rewards)),
            'avg_non_zero_rewards_per_episode': float(np.mean(non_zero_rewards)),
            'zero_reward_episodes': int(sum(1 for r in rewards if r == 0)),
            'high_performing_episodes': int(sum(1 for r in rewards if r > np.mean(rewards) + np.std(rewards)))
        },
        
        # Raw data
        'episode_rewards': rewards,
        'episode_lengths': lengths
    }
    
    # Print summary
    print("📊 REWARD STATISTICS:")
    print("-" * 60)
    print(f"Mean: {stats_dict['reward_stats']['mean']:.2f} ± {stats_dict['reward_stats']['std']:.2f}")
    print(f"Median: {stats_dict['reward_stats']['median']:.2f}")
    print(f"Range: [{stats_dict['reward_stats']['min']:.0f}, {stats_dict['reward_stats']['max']:.0f}]")
    print(f"IQR: {stats_dict['reward_stats']['iqr']:.2f}")
    print(f"Coefficient of Variation: {stats_dict['reward_stats']['cv']:.2f}")
    
    print("\n📏 EPISODE LENGTH STATISTICS:")
    print("-" * 60)
    print(f"Mean: {stats_dict['length_stats']['mean']:.1f} ± {stats_dict['length_stats']['std']:.1f}")
    print(f"Median: {stats_dict['length_stats']['median']:.1f}")
    print(f"Range: [{stats_dict['length_stats']['min']}, {stats_dict['length_stats']['max']}]")
    
    print("\n🎯 PERFORMANCE METRICS:")
    print("-" * 60)
    print(f"Avg Reward/Step: {stats_dict['performance_stats']['avg_reward_per_step_mean']:.4f}")
    print(f"Avg Max Single Reward: {stats_dict['performance_stats']['max_single_reward_mean']:.2f}")
    print(f"Avg Non-Zero Rewards/Episode: {stats_dict['performance_stats']['avg_non_zero_rewards_per_episode']:.1f}")
    print(f"Zero Reward Episodes: {stats_dict['performance_stats']['zero_reward_episodes']}")
    print(f"High Performing Episodes (>μ+σ): {stats_dict['performance_stats']['high_performing_episodes']}")
    
    return stats_dict, episode_data


def create_comprehensive_visualizations(stats_dict, episode_data):
    """Create comprehensive visualization suite."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    rewards = stats_dict['episode_rewards']
    lengths = stats_dict['episode_lengths']
    episodes = range(1, len(rewards) + 1)
    
    # 1. Main Dashboard
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Episode rewards over time
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(episodes, rewards, alpha=0.6, linewidth=1, color='steelblue')
    ax1.axhline(y=stats_dict['reward_stats']['mean'], color='red', linestyle='--', 
                label=f"Mean: {stats_dict['reward_stats']['mean']:.1f}", linewidth=2)
    ax1.axhline(y=stats_dict['reward_stats']['median'], color='orange', linestyle='--', 
                label=f"Median: {stats_dict['reward_stats']['median']:.1f}", linewidth=2)
    ax1.fill_between(episodes, 
                      stats_dict['reward_stats']['mean'] - stats_dict['reward_stats']['std'],
                      stats_dict['reward_stats']['mean'] + stats_dict['reward_stats']['std'],
                      alpha=0.2, color='red', label='±1 Std Dev')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Total Reward', fontsize=11)
    ax1.set_title('Episode Rewards Over Time', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Reward distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(rewards, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=stats_dict['reward_stats']['mean'], color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=stats_dict['reward_stats']['median'], color='orange', linestyle='--', linewidth=2)
    ax2.set_xlabel('Total Reward', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Reward Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Episode lengths over time
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(episodes, lengths, alpha=0.6, linewidth=1, color='coral')
    ax3.axhline(y=stats_dict['length_stats']['mean'], color='red', linestyle='--', 
                label=f"Mean: {stats_dict['length_stats']['mean']:.1f}", linewidth=2)
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Episode Length', fontsize=11)
    ax3.set_title('Episode Lengths Over Time', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # Length distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(lengths, bins=30, color='coral', alpha=0.7, edgecolor='black')
    ax4.axvline(x=stats_dict['length_stats']['mean'], color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Episode Length', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Length Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Box plots
    ax5 = fig.add_subplot(gs[2, 0])
    bp1 = ax5.boxplot([rewards], labels=['Rewards'], patch_artist=True)
    bp1['boxes'][0].set_facecolor('steelblue')
    bp1['boxes'][0].set_alpha(0.7)
    ax5.set_ylabel('Total Reward', fontsize=11)
    ax5.set_title('Reward Box Plot', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Moving average
    ax6 = fig.add_subplot(gs[2, 1])
    window = 10
    moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
    ax6.plot(episodes, rewards, alpha=0.3, linewidth=1, color='steelblue', label='Raw')
    ax6.plot(episodes, moving_avg, linewidth=2, color='darkblue', label=f'{window}-Episode MA')
    ax6.set_xlabel('Episode', fontsize=11)
    ax6.set_ylabel('Total Reward', fontsize=11)
    ax6.set_title('Reward Moving Average', fontsize=13, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Scatter: Reward vs Length
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.scatter(lengths, rewards, alpha=0.5, color='purple', s=30)
    z = np.polyfit(lengths, rewards, 1)
    p = np.poly1d(z)
    ax7.plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.8, linewidth=2, label='Trend')
    ax7.set_xlabel('Episode Length', fontsize=11)
    ax7.set_ylabel('Total Reward', fontsize=11)
    ax7.set_title('Reward vs Length', fontsize=13, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Random Agent Baseline Performance - Comprehensive Dashboard', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('outputs/phase2/comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved comprehensive dashboard")
    
    # 2. Statistical Analysis Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Q-Q plot for normality
    stats.probplot(rewards, dist="norm", plot=axes[0, 0])
    axes[0, 0].set_title('Q-Q Plot (Reward Distribution)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_rewards = np.sort(rewards)
    cumulative = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
    axes[0, 1].plot(sorted_rewards, cumulative, linewidth=2, color='steelblue')
    axes[0, 1].set_xlabel('Total Reward', fontsize=11)
    axes[0, 1].set_ylabel('Cumulative Probability', fontsize=11)
    axes[0, 1].set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Violin plot
    parts = axes[1, 0].violinplot([rewards], positions=[0], showmeans=True, showmedians=True)
    axes[1, 0].set_xticks([0])
    axes[1, 0].set_xticklabels(['Rewards'])
    axes[1, 0].set_ylabel('Total Reward', fontsize=11)
    axes[1, 0].set_title('Reward Violin Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance over time (binned)
    bin_size = 10
    num_bins = len(rewards) // bin_size
    binned_means = [np.mean(rewards[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)]
    binned_stds = [np.std(rewards[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)]
    bin_centers = [(i+0.5)*bin_size for i in range(num_bins)]
    
    axes[1, 1].errorbar(bin_centers, binned_means, yerr=binned_stds, 
                        fmt='o-', capsize=5, linewidth=2, markersize=8, color='steelblue')
    axes[1, 1].set_xlabel('Episode', fontsize=11)
    axes[1, 1].set_ylabel('Mean Reward (per 10 episodes)', fontsize=11)
    axes[1, 1].set_title('Performance Stability Over Time', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/phase2/statistical_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved statistical analysis plots")
    
    # 3. Action Distribution Analysis
    all_actions = []
    for ep in episode_data:
        all_actions.extend(ep['actions'])
    
    action_counts = np.bincount(all_actions, minlength=6)
    action_labels = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    axes[0].bar(action_labels, action_counts, color='teal', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Action', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Action Distribution (All Episodes)', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Pie chart
    axes[1].pie(action_counts, labels=action_labels, autopct='%1.1f%%', 
                startangle=90, colors=plt.cm.Set3.colors)
    axes[1].set_title('Action Distribution Percentage', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/phase2/action_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved action distribution analysis")


def save_detailed_data(stats_dict, episode_data):
    """Save detailed data for future reference."""
    print("\n" + "="*60)
    print("SAVING DATA")
    print("="*60 + "\n")
    
    # Save statistics
    with open('data/phase2/baseline_statistics.json', 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print("✓ Saved baseline statistics to data/phase2/baseline_statistics.json")
    
    # Save episode data as CSV
    df = pd.DataFrame([
        {
            'episode': ep['episode'],
            'total_reward': ep['total_reward'],
            'episode_length': ep['episode_length'],
            'avg_reward_per_step': ep['avg_reward_per_step'],
            'max_single_reward': ep['max_single_reward'],
            'non_zero_rewards': ep['non_zero_rewards']
        }
        for ep in episode_data
    ])
    df.to_csv('data/phase2/baseline_episodes.csv', index=False)
    print("✓ Saved episode data to data/phase2/baseline_episodes.csv")
    
    # Create summary report
    summary = f"""
# Phase 2: Baseline Performance Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Reward Statistics
- Mean: {stats_dict['reward_stats']['mean']:.2f} ± {stats_dict['reward_stats']['std']:.2f}
- Median: {stats_dict['reward_stats']['median']:.2f}
- Range: [{stats_dict['reward_stats']['min']:.0f}, {stats_dict['reward_stats']['max']:.0f}]
- IQR: {stats_dict['reward_stats']['iqr']:.2f}
- Coefficient of Variation: {stats_dict['reward_stats']['cv']:.2f}

## Episode Length Statistics
- Mean: {stats_dict['length_stats']['mean']:.1f} ± {stats_dict['length_stats']['std']:.1f}
- Median: {stats_dict['length_stats']['median']:.1f}
- Range: [{stats_dict['length_stats']['min']}, {stats_dict['length_stats']['max']}]

## Performance Metrics
- Avg Reward/Step: {stats_dict['performance_stats']['avg_reward_per_step_mean']:.4f}
- Avg Max Single Reward: {stats_dict['performance_stats']['max_single_reward_mean']:.2f}
- Avg Non-Zero Rewards/Episode: {stats_dict['performance_stats']['avg_non_zero_rewards_per_episode']:.1f}
- Zero Reward Episodes: {stats_dict['performance_stats']['zero_reward_episodes']}
- High Performing Episodes (>μ+σ): {stats_dict['performance_stats']['high_performing_episodes']}

## Key Insights
- Random agent baseline established with {stats_dict['num_episodes']} episodes
- High variance in performance (CV = {stats_dict['reward_stats']['cv']:.2f})
- This baseline will be used for comparison with trained agents
"""
    
    with open('data/phase2/summary_report.md', 'w') as f:
        f.write(summary)
    print("✓ Saved summary report to data/phase2/summary_report.md")


def main():
    """Main execution function."""
    # Create directories
    create_output_dirs()
    
    # Collect baseline data
    episode_data = collect_baseline_data(num_episodes=100)
    
    # Analyze data
    stats_dict, episode_data = analyze_baseline_data(episode_data)
    
    # Create visualizations
    create_comprehensive_visualizations(stats_dict, episode_data)
    
    # Save data
    save_detailed_data(stats_dict, episode_data)
    
    print("\n" + "="*60)
    print("✅ PHASE 2 COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  Data:")
    print("    - data/phase2/baseline_statistics.json")
    print("    - data/phase2/baseline_episodes.csv")
    print("    - data/phase2/summary_report.md")
    print("  Visualizations:")
    print("    - outputs/phase2/comprehensive_dashboard.png")
    print("    - outputs/phase2/statistical_analysis.png")
    print("    - outputs/phase2/action_distribution.png")
    print("\nNext Steps:")
    print("  Ready to move to Phase 3: Preprocessing & Feature Engineering")
    print("\n")


if __name__ == "__main__":
    main()
