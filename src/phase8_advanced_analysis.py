"""
Phase 8: Advanced Analysis & Visualization

This script creates comprehensive analysis and visualizations comparing
all three trained agents: Basic DQN, Improved DQN, and PPO.

Analyses include:
- Learning curves comparison
- Performance statistics
- Training efficiency
- Algorithm trade-offs
- Publication-quality visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from scipy import stats
import pandas as pd

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11


def create_output_dirs():
    """Create necessary output directories."""
    Path('outputs/phase8').mkdir(parents=True, exist_ok=True)
    Path('data/phase8').mkdir(parents=True, exist_ok=True)
    print("✓ Created output directories")


def load_training_data():
    """Load training data from all phases."""
    print("\n" + "="*60)
    print("PHASE 8: ADVANCED ANALYSIS & VISUALIZATION")
    print("="*60 + "\n")
    
    print("Loading training data from all phases...")
    
    data = {}
    
    # Load Phase 2: Baseline
    try:
        baseline_rewards = np.load('../data/phase2/baseline_rewards.npy')
        data['baseline'] = {
            'mean': float(np.mean(baseline_rewards)),
            'std': float(np.std(baseline_rewards)),
            'rewards': baseline_rewards.tolist()
        }
        print(f"✓ Loaded Phase 2: Random Baseline ({data['baseline']['mean']:.2f} ± {data['baseline']['std']:.2f})")
    except:
        print("⚠ Could not load Phase 2 baseline data")
        data['baseline'] = {'mean': 146.95, 'std': 93.14, 'rewards': []}
    
    # Load Phase 4: Basic DQN
    try:
        with open('../data/phase4/training_metrics.json', 'r') as f:
            phase4 = json.load(f)
        data['phase4'] = phase4
        print(f"✓ Loaded Phase 4: Basic DQN (max: {phase4['max_eval_reward']:.2f})")
    except Exception as e:
        print(f"⚠ Could not load Phase 4 data: {e}")
        data['phase4'] = None
    
    # Load Phase 5: Improved DQN
    try:
        with open('../data/phase5/training_metrics.json', 'r') as f:
            phase5 = json.load(f)
        data['phase5'] = phase5
        print(f"✓ Loaded Phase 5: Improved DQN (max: {phase5['max_eval_reward']:.2f})")
    except Exception as e:
        print(f"⚠ Could not load Phase 5 data: {e}")
        data['phase5'] = None
    
    # Load Phase 6: PPO
    try:
        with open('data/phase6/training_metrics.json', 'r') as f:
            phase6 = json.load(f)
        data['phase6'] = phase6
        print(f"✓ Loaded Phase 6: PPO (max: {phase6['max_eval_reward']:.2f})")
    except Exception as e:
        print(f"⚠ Could not load Phase 6 data: {e}")
        data['phase6'] = None
    
    return data


def create_comprehensive_comparison(data):
    """Create comprehensive comparison visualizations."""
    print("\n" + "="*60)
    print("CREATING COMPREHENSIVE VISUALIZATIONS")
    print("="*60 + "\n")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Learning Curves Comparison (Large, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    
    baseline_mean = data['baseline']['mean']
    ax1.axhline(baseline_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Random Baseline ({baseline_mean:.1f})', alpha=0.6, zorder=1)
    
    if data['phase4']:
        steps = data['phase4']['eval_steps']
        rewards = data['phase4']['eval_rewards']
        ax1.plot(steps, rewards, 'o-', linewidth=2.5, markersize=8, 
                label=f"Basic DQN (max: {data['phase4']['max_eval_reward']:.1f})", 
                color='orange', alpha=0.8, zorder=3)
    
    if data['phase5']:
        steps = data['phase5']['eval_steps']
        rewards = data['phase5']['eval_rewards']
        ax1.plot(steps, rewards, 's-', linewidth=2.5, markersize=8,
                label=f"Improved DQN (max: {data['phase5']['max_eval_reward']:.1f})", 
                color='blue', alpha=0.8, zorder=4)
    
    if data['phase6']:
        steps = data['phase6']['eval_steps']
        rewards = data['phase6']['eval_rewards']
        ax1.plot(steps, rewards, '^-', linewidth=2.5, markersize=8,
                label=f"PPO (max: {data['phase6']['max_eval_reward']:.1f})", 
                color='green', alpha=0.8, zorder=5)
    
    ax1.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Evaluation Reward', fontsize=13, fontweight='bold')
    ax1.set_title('Learning Curves: Algorithm Comparison', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(left=0)
    
    # Plot 2: Final Performance Bar Chart
    ax2 = fig.add_subplot(gs[0, 2])
    
    algorithms = ['Random\nBaseline']
    max_rewards = [baseline_mean]
    colors = ['red']
    
    if data['phase4']:
        algorithms.append('Basic\nDQN')
        max_rewards.append(data['phase4']['max_eval_reward'])
        colors.append('orange')
    
    if data['phase5']:
        algorithms.append('Improved\nDQN')
        max_rewards.append(data['phase5']['max_eval_reward'])
        colors.append('blue')
    
    if data['phase6']:
        algorithms.append('PPO')
        max_rewards.append(data['phase6']['max_eval_reward'])
        colors.append('green')
    
    bars = ax2.bar(algorithms, max_rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Max Reward', fontsize=12, fontweight='bold')
    ax2.set_title('Maximum Performance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for bar, reward in zip(bars, max_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{reward:.0f}', ha='center', va='bottom', 
                fontsize=12, fontweight='bold')
    
    # Plot 3: Improvement over Baseline
    ax3 = fig.add_subplot(gs[1, 0])
    
    improvements = []
    imp_labels = []
    imp_colors = []
    
    if data['phase4']:
        imp = ((data['phase4']['max_eval_reward'] - baseline_mean) / baseline_mean) * 100
        improvements.append(imp)
        imp_labels.append('Basic\nDQN')
        imp_colors.append('orange')
    
    if data['phase5']:
        imp = ((data['phase5']['max_eval_reward'] - baseline_mean) / baseline_mean) * 100
        improvements.append(imp)
        imp_labels.append('Improved\nDQN')
        imp_colors.append('blue')
    
    if data['phase6']:
        imp = ((data['phase6']['max_eval_reward'] - baseline_mean) / baseline_mean) * 100
        improvements.append(imp)
        imp_labels.append('PPO')
        imp_colors.append('green')
    
    bars = ax3.bar(imp_labels, improvements, color=imp_colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Improvement Over Baseline', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.axhline(0, color='black', linewidth=1)
    
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'+{imp:.0f}%', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Plot 4: Training Stability (Standard Deviation of Eval Rewards)
    ax4 = fig.add_subplot(gs[1, 1])
    
    stabilities = []
    stab_labels = []
    stab_colors = []
    
    if data['phase4']:
        std = np.std(data['phase4']['eval_rewards'])
        stabilities.append(std)
        stab_labels.append('Basic\nDQN')
        stab_colors.append('orange')
    
    if data['phase5']:
        std = np.std(data['phase5']['eval_rewards'])
        stabilities.append(std)
        stab_labels.append('Improved\nDQN')
        stab_colors.append('blue')
    
    if data['phase6']:
        std = np.std(data['phase6']['eval_rewards'])
        stabilities.append(std)
        stab_labels.append('PPO')
        stab_colors.append('green')
    
    bars = ax4.bar(stab_labels, stabilities, color=stab_colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Std Dev of Eval Rewards', fontsize=12, fontweight='bold')
    ax4.set_title('Training Stability (Lower is Better)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    for bar, std in zip(bars, stabilities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{std:.1f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    # Plot 5: Sample Efficiency (Steps to reach 300 reward)
    ax5 = fig.add_subplot(gs[1, 2])
    
    target_reward = 300
    efficiencies = []
    eff_labels = []
    eff_colors = []
    
    if data['phase4']:
        steps = data['phase4']['eval_steps']
        rewards = data['phase4']['eval_rewards']
        for s, r in zip(steps, rewards):
            if r >= target_reward:
                efficiencies.append(s)
                eff_labels.append('Basic\nDQN')
                eff_colors.append('orange')
                break
    
    if data['phase5']:
        steps = data['phase5']['eval_steps']
        rewards = data['phase5']['eval_rewards']
        for s, r in zip(steps, rewards):
            if r >= target_reward:
                efficiencies.append(s)
                eff_labels.append('Improved\nDQN')
                eff_colors.append('blue')
                break
    
    if data['phase6']:
        steps = data['phase6']['eval_steps']
        rewards = data['phase6']['eval_rewards']
        for s, r in zip(steps, rewards):
            if r >= target_reward:
                efficiencies.append(s)
                eff_labels.append('PPO')
                eff_colors.append('green')
                break
    
    if efficiencies:
        bars = ax5.bar(eff_labels, efficiencies, color=eff_colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax5.set_ylabel('Training Steps', fontsize=12, fontweight='bold')
        ax5.set_title(f'Sample Efficiency (Steps to {target_reward} Reward)', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        for bar, steps in zip(bars, efficiencies):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{steps/1000:.0f}k', ha='center', va='bottom', 
                    fontsize=11, fontweight='bold')
    
    # Plot 6: Evaluation Reward Distribution (Box Plot)
    ax6 = fig.add_subplot(gs[2, :2])
    
    box_data = []
    box_labels = []
    box_colors = []
    
    if data['phase4']:
        box_data.append(data['phase4']['eval_rewards'])
        box_labels.append('Basic DQN')
        box_colors.append('orange')
    
    if data['phase5']:
        box_data.append(data['phase5']['eval_rewards'])
        box_labels.append('Improved DQN')
        box_colors.append('blue')
    
    if data['phase6']:
        box_data.append(data['phase6']['eval_rewards'])
        box_labels.append('PPO')
        box_colors.append('green')
    
    bp = ax6.boxplot(box_data, labels=box_labels, patch_artist=True, 
                     showmeans=True, meanline=True,
                     boxprops=dict(linewidth=2),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2),
                     medianprops=dict(linewidth=2, color='red'),
                     meanprops=dict(linewidth=2, color='purple', linestyle='--'))
    
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax6.set_ylabel('Evaluation Reward', fontsize=12, fontweight='bold')
    ax6.set_title('Evaluation Reward Distributions', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax6.axhline(baseline_mean, color='red', linestyle='--', linewidth=1, alpha=0.3, label='Baseline')
    
    # Plot 7: Statistical Summary Table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    table_data = []
    
    if data['phase4']:
        table_data.append([
            'Basic DQN',
            f"{np.mean(data['phase4']['eval_rewards']):.1f}",
            f"{data['phase4']['max_eval_reward']:.1f}",
            f"{np.std(data['phase4']['eval_rewards']):.1f}",
            f"{len(data['phase4']['eval_rewards'])}"
        ])
    
    if data['phase5']:
        table_data.append([
            'Improved DQN',
            f"{np.mean(data['phase5']['eval_rewards']):.1f}",
            f"{data['phase5']['max_eval_reward']:.1f}",
            f"{np.std(data['phase5']['eval_rewards']):.1f}",
            f"{len(data['phase5']['eval_rewards'])}"
        ])
    
    if data['phase6']:
        table_data.append([
            'PPO',
            f"{np.mean(data['phase6']['eval_rewards']):.1f}",
            f"{data['phase6']['max_eval_reward']:.1f}",
            f"{np.std(data['phase6']['eval_rewards']):.1f}",
            f"{len(data['phase6']['eval_rewards'])}"
        ])
    
    table = ax7.table(cellText=table_data,
                     colLabels=['Algorithm', 'Mean', 'Max', 'Std', 'N Evals'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code rows
    colors_map = {'Basic DQN': 'orange', 'Improved DQN': 'blue', 'PPO': 'green'}
    for i, row in enumerate(table_data, 1):
        color = colors_map.get(row[0], 'white')
        table[(i, 0)].set_facecolor(color)
        table[(i, 0)].set_alpha(0.3)
    
    ax7.set_title('Statistical Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Plot 8: Learning Rate Analysis (if multiple algorithms)
    ax8 = fig.add_subplot(gs[3, :])
    
    # Create timeline showing all evaluation points
    all_points = []
    
    if data['phase4']:
        for step, reward in zip(data['phase4']['eval_steps'], data['phase4']['eval_rewards']):
            all_points.append((step, reward, 'Basic DQN', 'orange'))
    
    if data['phase5']:
        for step, reward in zip(data['phase5']['eval_steps'], data['phase5']['eval_rewards']):
            all_points.append((step, reward, 'Improved DQN', 'blue'))
    
    if data['phase6']:
        for step, reward in zip(data['phase6']['eval_steps'], data['phase6']['eval_rewards']):
            all_points.append((step, reward, 'PPO', 'green'))
    
    # Group by algorithm and plot
    for algo in ['Basic DQN', 'Improved DQN', 'PPO']:
        algo_points = [(s, r) for s, r, a, c in all_points if a == algo]
        if algo_points:
            steps, rewards = zip(*algo_points)
            color = [c for s, r, a, c in all_points if a == algo][0]
            ax8.scatter(steps, rewards, s=100, alpha=0.6, label=algo, color=color, edgecolors='black', linewidth=1)
    
    ax8.axhline(baseline_mean, color='red', linestyle='--', linewidth=2, alpha=0.4, label='Baseline')
    ax8.set_xlabel('Training Steps', fontsize=13, fontweight='bold')
    ax8.set_ylabel('Evaluation Reward', fontsize=13, fontweight='bold')
    ax8.set_title('All Evaluation Points: Complete Training History', fontsize=15, fontweight='bold')
    ax8.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax8.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('Space Invaders RL: Comprehensive Algorithm Comparison', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig('outputs/phase8/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved comprehensive analysis visualization")


def create_statistical_analysis(data):
    """Perform statistical analysis and save report."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60 + "\n")
    
    report = []
    report.append("="*60)
    report.append("STATISTICAL ANALYSIS REPORT")
    report.append("="*60)
    report.append("")
    
    baseline_mean = data['baseline']['mean']
    
    # Summary statistics
    report.append("1. SUMMARY STATISTICS")
    report.append("-" * 60)
    report.append(f"{'Algorithm':<20} {'Mean':<10} {'Max':<10} {'Std':<10} {'N':<5}")
    report.append("-" * 60)
    report.append(f"{'Random Baseline':<20} {baseline_mean:<10.2f} {'N/A':<10} {data['baseline']['std']:<10.2f} {len(data['baseline']['rewards']) if data['baseline']['rewards'] else 'N/A':<5}")
    
    if data['phase4']:
        mean_reward = np.mean(data['phase4']['eval_rewards'])
        max_reward = data['phase4']['max_eval_reward']
        std_reward = np.std(data['phase4']['eval_rewards'])
        n = len(data['phase4']['eval_rewards'])
        report.append(f"{'Basic DQN':<20} {mean_reward:<10.2f} {max_reward:<10.2f} {std_reward:<10.2f} {n:<5}")
    
    if data['phase5']:
        mean_reward = np.mean(data['phase5']['eval_rewards'])
        max_reward = data['phase5']['max_eval_reward']
        std_reward = np.std(data['phase5']['eval_rewards'])
        n = len(data['phase5']['eval_rewards'])
        report.append(f"{'Improved DQN':<20} {mean_reward:<10.2f} {max_reward:<10.2f} {std_reward:<10.2f} {n:<5}")
    
    if data['phase6']:
        mean_reward = np.mean(data['phase6']['eval_rewards'])
        max_reward = data['phase6']['max_eval_reward']
        std_reward = np.std(data['phase6']['eval_rewards'])
        n = len(data['phase6']['eval_rewards'])
        report.append(f"{'PPO':<20} {mean_reward:<10.2f} {max_reward:<10.2f} {std_reward:<10.2f} {n:<5}")
    
    report.append("")
    
    # Improvement analysis
    report.append("2. IMPROVEMENT OVER BASELINE")
    report.append("-" * 60)
    
    if data['phase4']:
        imp = ((data['phase4']['max_eval_reward'] - baseline_mean) / baseline_mean) * 100
        report.append(f"Basic DQN:      {data['phase4']['max_eval_reward']:.2f} vs {baseline_mean:.2f} = +{imp:.1f}%")
    
    if data['phase5']:
        imp = ((data['phase5']['max_eval_reward'] - baseline_mean) / baseline_mean) * 100
        report.append(f"Improved DQN:   {data['phase5']['max_eval_reward']:.2f} vs {baseline_mean:.2f} = +{imp:.1f}%")
    
    if data['phase6']:
        imp = ((data['phase6']['max_eval_reward'] - baseline_mean) / baseline_mean) * 100
        report.append(f"PPO:            {data['phase6']['max_eval_reward']:.2f} vs {baseline_mean:.2f} = +{imp:.1f}%")
    
    report.append("")
    
    # Best algorithm
    report.append("3. BEST PERFORMING ALGORITHM")
    report.append("-" * 60)
    
    best_algo = None
    best_reward = baseline_mean
    
    if data['phase4'] and data['phase4']['max_eval_reward'] > best_reward:
        best_reward = data['phase4']['max_eval_reward']
        best_algo = 'Basic DQN (Phase 4)'
    
    if data['phase5'] and data['phase5']['max_eval_reward'] > best_reward:
        best_reward = data['phase5']['max_eval_reward']
        best_algo = 'Improved DQN (Phase 5)'
    
    if data['phase6'] and data['phase6']['max_eval_reward'] > best_reward:
        best_reward = data['phase6']['max_eval_reward']
        best_algo = 'PPO (Phase 6)'
    
    report.append(f"Winner: {best_algo}")
    report.append(f"Max Reward: {best_reward:.2f}")
    report.append(f"Improvement: +{((best_reward - baseline_mean) / baseline_mean) * 100:.1f}% over baseline")
    report.append("")
    
    # Save report
    report_text = "\n".join(report)
    print(report_text)
    
    with open('data/phase8/statistical_analysis.txt', 'w') as f:
        f.write(report_text)
    
    print("\n✓ Saved statistical analysis report")
    
    return report_text


def save_summary_json(data):
    """Save summary data as JSON."""
    summary = {
        'baseline': data['baseline'],
        'algorithms': {}
    }
    
    if data['phase4']:
        summary['algorithms']['basic_dqn'] = {
            'mean_reward': float(np.mean(data['phase4']['eval_rewards'])),
            'max_reward': float(data['phase4']['max_eval_reward']),
            'std_reward': float(np.std(data['phase4']['eval_rewards'])),
            'n_evaluations': len(data['phase4']['eval_rewards'])
        }
    
    if data['phase5']:
        summary['algorithms']['improved_dqn'] = {
            'mean_reward': float(np.mean(data['phase5']['eval_rewards'])),
            'max_reward': float(data['phase5']['max_eval_reward']),
            'std_reward': float(np.std(data['phase5']['eval_rewards'])),
            'n_evaluations': len(data['phase5']['eval_rewards'])
        }
    
    if data['phase6']:
        summary['algorithms']['ppo'] = {
            'mean_reward': float(np.mean(data['phase6']['eval_rewards'])),
            'max_reward': float(data['phase6']['max_eval_reward']),
            'std_reward': float(np.std(data['phase6']['eval_rewards'])),
            'n_evaluations': len(data['phase6']['eval_rewards'])
        }
    
    with open('data/phase8/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("✓ Saved analysis summary JSON")


def main():
    """Main execution function."""
    create_output_dirs()
    
    # Load data
    data = load_training_data()
    
    # Create visualizations
    create_comprehensive_comparison(data)
    
    # Statistical analysis
    create_statistical_analysis(data)
    
    # Save summary
    save_summary_json(data)
    
    print("\n" + "="*60)
    print("✅ PHASE 8 ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  Visualizations:")
    print("    - outputs/phase8/comprehensive_analysis.png")
    print("  Analysis:")
    print("    - data/phase8/statistical_analysis.txt")
    print("    - data/phase8/analysis_summary.json")
    print("\nKey Findings:")
    
    # Print winner
    baseline_mean = data['baseline']['mean']
    best_reward = baseline_mean
    best_algo = 'Random Baseline'
    
    if data['phase4'] and data['phase4']['max_eval_reward'] > best_reward:
        best_reward = data['phase4']['max_eval_reward']
        best_algo = 'Basic DQN'
    
    if data['phase5'] and data['phase5']['max_eval_reward'] > best_reward:
        best_reward = data['phase5']['max_eval_reward']
        best_algo = 'Improved DQN'
    
    if data['phase6'] and data['phase6']['max_eval_reward'] > best_reward:
        best_reward = data['phase6']['max_eval_reward']
        best_algo = 'PPO'
    
    print(f"  🏆 Best Algorithm: {best_algo} ({best_reward:.2f} reward)")
    print(f"  📈 Improvement: +{((best_reward - baseline_mean) / baseline_mean) * 100:.1f}% over baseline")
    print("\nNext: Phase 9 (Model Interpretability) or Phase 10 (Deployment)")
    print("\n")


if __name__ == "__main__":
    main()
