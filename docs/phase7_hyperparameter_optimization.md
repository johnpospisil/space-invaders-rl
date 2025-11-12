# Phase 7: Hyperparameter Optimization

## Overview

Phase 7 uses **Optuna** (Bayesian optimization) to systematically tune PPO hyperparameters and improve upon the current best performance of **561.50 max reward**.

## Why Hyperparameter Optimization?

Current results show PPO performed best (561.50), but these hyperparameters were chosen based on common defaults. Systematic optimization can:

- **Find better configurations** that may significantly improve performance
- **Understand parameter sensitivity** - which hyperparameters matter most
- **Validate current choices** - confirm defaults are reasonable
- **Discover interactions** - how parameters affect each other

## Optimization Method: Optuna with TPE

### Tree-structured Parzen Estimator (TPE)

- **Bayesian optimization**: Builds probabilistic model of objective function
- **Efficient search**: Focuses on promising regions of hyperparameter space
- **Better than grid search**: Requires fewer trials to find good configurations
- **Adaptive**: Learns from previous trials to suggest better parameters

### Median Pruner

- **Early stopping**: Terminates unpromising trials
- **Saves time**: Don't waste resources on bad configurations
- **More trials**: Can explore more configurations in same time budget

## Hyperparameters Being Tuned

| Parameter         | Range              | Default (Phase 6) | Purpose                                        |
| ----------------- | ------------------ | ----------------- | ---------------------------------------------- |
| **learning_rate** | [1e-5, 1e-3]       | 3e-4              | Step size for gradient updates                 |
| **gamma**         | [0.95, 0.999]      | 0.99              | Discount factor for future rewards             |
| **gae_lambda**    | [0.9, 0.99]        | 0.95              | Bias-variance tradeoff in advantage estimation |
| **clip_epsilon**  | [0.1, 0.3]         | 0.2               | Trust region size for policy updates           |
| **value_coef**    | [0.25, 1.0]        | 0.5               | Weight of value loss in total loss             |
| **entropy_coef**  | [0.001, 0.05]      | 0.01              | Exploration bonus                              |
| **max_grad_norm** | [0.3, 1.0]         | 0.5               | Gradient clipping threshold                    |
| **n_steps**       | {1024, 2048, 4096} | 2048              | Rollout length                                 |
| **batch_size**    | {128, 256, 512}    | 256               | Mini-batch size for updates                    |
| **n_epochs**      | [3, 10]            | 4                 | Update epochs per rollout                      |

## Implementation Details

### Objective Function

```python
def objective(trial):
    # Suggest hyperparameters from ranges
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    # ... more parameters

    # Train agent with suggested hyperparameters
    agent = PPOAgent(learning_rate=lr, gamma=gamma, ...)

    # Evaluate and return mean reward
    return mean_evaluation_reward
```

### Training per Trial

- **Steps**: 200,000 (reduced from 1M for speed)
- **Evaluations**: Every 50,000 steps (5 episodes each)
- **Pruning**: Stop trial if performance is poor at checkpoints
- **Return**: Mean of all evaluation rewards

### Why Shorter Training?

- **Faster iteration**: Can test more configurations
- **Early signals**: Good hyperparameters show promise quickly
- **Final validation**: Best config will be trained for full 1M steps
- **Trade-off**: Some good configs may need longer training to show potential

## Files

### Core Scripts

- **`src/phase7_hyperparam_optimization.py`**: Full optimization (20 trials)
- **`run_phase7_quick_test.py`**: Quick test (5 trials)

### Functions

1. **`objective(trial)`**: Optuna objective function to maximize
2. **`run_optimization(n_trials)`**: Run optimization study
3. **`save_optimization_results(study)`**: Save and visualize results
4. **`train_best_config(params)`**: Train best config for full 1M steps

### Outputs

- **`data/phase7/optimization_results.json`**: Best hyperparameters and performance
- **`data/phase7/all_trials.json`**: All trial data for analysis
- **`outputs/phase7/optimization_results.png`**: Visualization of optimization process
- **`models/phase7/ppo_optimized.pt`**: Final model with best hyperparameters

## Usage

### Quick Test (Recommended First)

```bash
python run_phase7_quick_test.py
```

- **Trials**: 5
- **Time**: ~50 minutes
- **Purpose**: Validate setup, get initial insights

### Full Optimization

```bash
cd src
python phase7_hyperparam_optimization.py
```

- **Trials**: 20
- **Time**: ~3-4 hours
- **Purpose**: Thorough search for best configuration

### Train Best Configuration

After optimization completes, train the best hyperparameters for full duration:

```python
from phase7_hyperparam_optimization import train_best_config
import json

# Load best hyperparameters
with open('data/phase7/optimization_results.json', 'r') as f:
    results = json.load(f)

# Train for 1M steps
train_best_config(results['best_params'], num_steps=1_000_000)
```

## Visualizations Generated

1. **Optimization History**

   - Trial number vs reward
   - Shows learning progress of optimization
   - Baseline comparison

2. **Parameter Importance**

   - Which hyperparameters matter most
   - Based on functional ANOVA
   - Helps focus future tuning efforts

3. **Learning Rate Impact**

   - Scatter plot of learning rate vs reward
   - Shows optimal range
   - Log scale for better visualization

4. **Clip Epsilon Impact**
   - Scatter plot of clip epsilon vs reward
   - Shows trust region sensitivity
   - Critical for PPO stability

## Expected Outcomes

### Scenario 1: Improvement Found

```
Best Trial Reward: 620.5
Baseline (Phase 6): 561.50
Improvement: +10.5%
```

✅ Hyperparameter tuning successful!  
➡️ Train best config for 1M steps  
➡️ Move to Phase 8 with optimized agent

### Scenario 2: Similar Performance

```
Best Trial Reward: 558.3
Baseline (Phase 6): 561.50
Difference: -0.6%
```

✅ Defaults were already good!  
➡️ Phase 6 hyperparameters validated  
➡️ Move to Phase 8 with Phase 6 agent

### Scenario 3: Need More Training

```
Best Trial Reward: 485.2
Baseline (Phase 6): 561.50
Note: 200k steps vs 1M baseline
```

⚠️ Short training may not show full potential  
➡️ Train best config for 1M steps  
➡️ May match or exceed baseline with more training

## Key Insights

### What Makes Good Hyperparameters?

1. **Learning Rate**

   - Too high: Unstable learning, divergence
   - Too low: Slow learning, gets stuck
   - Sweet spot: Typically 1e-4 to 5e-4 for PPO

2. **Clip Epsilon**

   - Too high: Large policy changes, instability
   - Too low: Too conservative, slow learning
   - Sweet spot: 0.1 to 0.2 for most tasks

3. **GAE Lambda**

   - High (0.95-0.99): Low bias, high variance
   - Low (0.90-0.93): High bias, low variance
   - Trade-off: Depends on environment dynamics

4. **Entropy Coefficient**
   - High: More exploration, slower convergence
   - Low: Less exploration, may get stuck
   - Decay: Can start high and decrease

### Parameter Interactions

- **Learning rate × n_epochs**: Higher LR may need fewer epochs
- **Clip epsilon × learning rate**: Smaller clip needs smaller LR
- **n_steps × batch_size**: Larger rollouts can use larger batches
- **Entropy × clip epsilon**: Both affect exploration stability

## Troubleshooting

### All Trials Perform Poorly

- Check if environment is working
- Verify 200k steps is enough to show learning
- May need to adjust search ranges
- Consider longer training per trial

### High Variance in Results

- Inherent in RL (stochastic)
- Increase evaluation episodes
- Run multiple seeds for best config
- Use median instead of mean

### Optuna Errors

```bash
# Install optuna if missing
pip install optuna

# Update if needed
pip install --upgrade optuna
```

### Memory Issues

- Reduce batch_size search space
- Reduce n_steps options
- Close environments properly
- Use `del agent` between trials

## Comparison with Other Methods

| Method                | Pros                            | Cons                     | When to Use                    |
| --------------------- | ------------------------------- | ------------------------ | ------------------------------ |
| **Grid Search**       | Systematic, guaranteed coverage | Slow, exponential cost   | Few parameters, unlimited time |
| **Random Search**     | Simple, parallelizable          | Inefficient, no learning | Quick exploration              |
| **Bayesian (Optuna)** | Efficient, learns from trials   | Setup complexity         | Limited trials, want best      |
| **Evolutionary**      | Robust, global search           | Many evaluations needed  | Complex landscapes             |

## Next Steps

### After Quick Test (5 trials)

1. Review initial results
2. Adjust search ranges if needed
3. Run full optimization (20 trials)

### After Full Optimization

1. Analyze parameter importance
2. Train best config for 1M steps
3. Compare with Phase 6 baseline
4. Document findings

### Phase 8 Preparation

- Best performing agent (optimized or Phase 6)
- All training history
- Ready for advanced analysis

## References

- **Optuna Paper**: Akiba et al. (2019) - "Optuna: A Next-generation Hyperparameter Optimization Framework"
- **TPE Algorithm**: Bergstra et al. (2011) - "Algorithms for Hyper-Parameter Optimization"
- **PPO Paper**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- **Hyperparameter Tuning**: Henderson et al. (2018) - "Deep Reinforcement Learning that Matters"

## Tips for Success

1. **Start small**: Quick test validates setup
2. **Monitor progress**: Check trial results during optimization
3. **Be patient**: Full optimization takes time but is worth it
4. **Save everything**: All trials provide learning
5. **Validate findings**: Train best config fully before claiming improvement
6. **Document**: Note which parameters matter for future work

---

**Status**: 🔄 Optimization in progress (quick test running)

**Goal**: Find hyperparameters that exceed 561.50 max reward

**Next**: Analyze results and potentially run full optimization or move to Phase 8
