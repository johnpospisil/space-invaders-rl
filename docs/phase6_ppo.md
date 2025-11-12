# Phase 6: Alternative Algorithms - PPO

## Overview

Phase 6 implements **Proximal Policy Optimization (PPO)** as an alternative to the DQN-based approaches from Phases 4 and 5. This enables comparison between **value-based** (DQN) and **policy-gradient** (PPO) methods.

## Key Concepts

### PPO vs DQN: Fundamental Differences

| Aspect              | DQN (Phases 4-5)           | PPO (Phase 6)                   |
| ------------------- | -------------------------- | ------------------------------- |
| **Learning Type**   | Value-based                | Policy-gradient                 |
| **What it learns**  | Q(s,a) - value of actions  | π(a\|s) - policy directly       |
| **Update Strategy** | Off-policy (replay buffer) | On-policy (rollout collection)  |
| **Exploration**     | ε-greedy                   | Stochastic policy               |
| **Data Efficiency** | High (reuses experience)   | Lower (recent data only)        |
| **Stability**       | Can be unstable            | Clipped objective for stability |

### PPO Algorithm

PPO optimizes the policy directly using three key components:

1. **Actor-Critic Architecture**

   - **Actor**: Outputs policy π(a|s) - probability distribution over actions
   - **Critic**: Estimates V(s) - value of being in state s
   - **Shared CNN backbone** for efficient feature learning

2. **Generalized Advantage Estimation (GAE)**

   ```
   Advantage A(s,a) = Q(s,a) - V(s)
   GAE: A^GAE(s,a) = Σ(γλ)^t δ_t
   where δ_t = r_t + γV(s_{t+1}) - V(s_t)
   ```

   - Balances bias vs variance with λ parameter
   - λ=0: high bias, low variance (TD)
   - λ=1: low bias, high variance (Monte Carlo)
   - We use λ=0.95

3. **Clipped Surrogate Objective**
   ```
   L^CLIP = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
   where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
   ```
   - Prevents large policy updates
   - Clip ratio r_t to [1-ε, 1+ε] range
   - We use ε=0.2

### Implementation Details

**Network Architecture:**

```
Shared CNN:
  Conv2D(4, 32, 8x8, stride=4) → ReLU
  Conv2D(32, 64, 4x4, stride=2) → ReLU
  Conv2D(64, 64, 3x3, stride=1) → ReLU
  Flatten → FC(3136, 512) → ReLU

Actor Head (Policy):
  FC(512, n_actions) → Softmax

Critic Head (Value):
  FC(512, 1) → Linear
```

**Hyperparameters:**

- Learning rate: 3e-4 (Adam optimizer)
- Discount factor γ: 0.99
- GAE lambda λ: 0.95
- Clip epsilon ε: 0.2
- Value loss coefficient: 0.5
- Entropy bonus: 0.01 (encourages exploration)
- Rollout steps: 2,048 (collects full trajectories)
- Mini-batch size: 256
- Update epochs: 4 (per rollout)

## Files

### Core Implementation

- **`src/ppo_agent.py`** (372 lines)
  - `ActorCriticNetwork`: Shared CNN + actor/critic heads
  - `RolloutBuffer`: Stores trajectories for on-policy learning
  - `PPOAgent`: Full PPO implementation with GAE and clipped objective

### Training Scripts

- **`src/phase6_train_ppo.py`**: Full training (1M steps, ~2-3 hours)
- **`run_phase6_quick_test.py`**: Quick validation (100k steps, ~30 minutes)

### Outputs

- **Models:** `models/phase6/ppo_*.pt`
- **Metrics:** `data/phase6/training_metrics.json`
- **Visualizations:** `outputs/phase6/training_results.png`

## Usage

### Quick Test (Recommended First)

```bash
python run_phase6_quick_test.py
```

- Runs for 100k steps (~30 minutes)
- Validates implementation
- Saves to `models/phase6/ppo_quick_test.pt`

### Full Training

```bash
cd src
python phase6_train_ppo.py
```

- Runs for 1M steps (~2-3 hours)
- Evaluates every 50k steps
- Saves checkpoints every 100k steps
- Creates comparison with Phases 4 & 5

### Load and Evaluate

```python
from preprocessing import make_atari_env
from ppo_agent import PPOAgent

env = make_atari_env('ALE/SpaceInvaders-v5', frame_stack=4)
agent = PPOAgent(env.observation_space.shape, env.action_space.n)
agent.load('models/phase6/ppo_final.pt')

# Run episode
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = agent.select_action(obs, training=False)
    obs, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    done = done or truncated

print(f"Episode reward: {total_reward}")
```

## Expected Results

Based on typical PPO performance on Atari:

| Metric                  | Expected Range | Target |
| ----------------------- | -------------- | ------ |
| Final Eval Reward       | 300-500        | >350   |
| Max Eval Reward         | 400-600        | >450   |
| Improvement vs Baseline | 170-310%       | >200%  |

**Comparison with DQN:**

- PPO may have smoother learning curves (clipped updates)
- DQN typically more sample efficient (replay buffer)
- PPO better for environments requiring exploration
- Results depend on hyperparameter tuning

## Key Insights

### Advantages of PPO

1. **Stable Updates**: Clipped objective prevents destructive policy changes
2. **Direct Policy Learning**: No need for ε-greedy or max operators
3. **Stochastic Policy**: Natural exploration through probability distribution
4. **Parallelizable**: Can run multiple environments simultaneously (not implemented here)

### Limitations vs DQN

1. **Sample Efficiency**: Only uses recent data (on-policy)
2. **Memory**: Must store full rollouts before updating
3. **Computation**: Multiple epochs per rollout increases cost
4. **Tuning Sensitivity**: More hyperparameters to configure

### When to Use PPO vs DQN

- **Use PPO when:**

  - Need stable, monotonic improvement
  - Environment requires exploration
  - Can collect data efficiently
  - Want to parallelize environments

- **Use DQN when:**
  - Sample efficiency is critical
  - Have limited environment interactions
  - Want simpler implementation
  - Deterministic policy is acceptable

## Analysis Questions

1. **Learning Curves**: Does PPO show smoother learning than DQN?
2. **Sample Efficiency**: Steps needed to reach baseline performance?
3. **Final Performance**: Max reward vs DQN/Improved DQN?
4. **Stability**: Variance in evaluation rewards?
5. **Training Time**: Wall-clock time vs DQN (at equal steps)?

## Visualizations

The training script generates:

1. **Algorithm Comparison**: PPO vs DQN vs Improved DQN vs Baseline
2. **Training Progress**: Episode rewards with moving average
3. **Policy Loss**: Shows optimization stability
4. **Performance Bar Chart**: Final comparison of all agents

## Next Steps

### Phase 7: Hyperparameter Optimization

- Grid search or Bayesian optimization
- Test different learning rates, clip values, GAE lambdas
- Find optimal configuration

### Phase 8: Advanced Analysis

- Learning curve analysis
- State visitation patterns
- Policy visualization
- Ablation studies

### Phase 9: Model Interpretability

- Saliency maps (what the agent "sees")
- Action prediction analysis
- Feature visualization

## References

- **PPO Paper**: Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- **GAE Paper**: Schulman et al. (2016) - "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
- **OpenAI Spinning Up**: https://spinningup.openai.com/en/latest/algorithms/ppo.html

## Troubleshooting

**Training too slow:**

- Reduce `n_steps` (2048 → 1024)
- Reduce `n_epochs` (4 → 2)
- Use GPU if available

**Poor performance:**

- Increase training steps (1M → 2M)
- Tune learning rate (try 1e-4 or 5e-4)
- Adjust clip_epsilon (try 0.1 or 0.3)
- Tune entropy coefficient (exploration)

**Instability:**

- Reduce learning rate
- Increase batch size
- Reduce clip_epsilon
- Check for NaN values in losses

---

**Status**: ✅ Implementation complete, training in progress

**Performance**: Testing phase (quick test running)

**Next**: Full 1M step training, then Phase 7
