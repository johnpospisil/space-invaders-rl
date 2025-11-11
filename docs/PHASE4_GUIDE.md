# Phase 4: DQN Agent - Basic Implementation

## Overview

This phase implements a basic Deep Q-Network (DQN) agent following the Nature paper (Mnih et al., 2015). This is your first **learning agent** that improves through experience.

## What Was Implemented

### 1. Core Components (`src/`)

#### `replay_buffer.py`

- **Purpose**: Store and sample transitions for training
- **Key Features**:
  - Pre-allocated numpy arrays for memory efficiency
  - Fixed capacity with circular overwrite
  - Random sampling to break temporal correlations
  - Automatic device conversion (CPU/GPU)

#### `dqn_agent.py`

- **DQN Network**:
  - 3 convolutional layers (32, 64, 64 filters)
  - 2 fully connected layers (512, n_actions)
  - Follows Nature DQN architecture exactly
- **DQNAgent Class**:
  - Epsilon-greedy exploration with decay
  - Target network for stable learning
  - Huber loss with gradient clipping
  - Save/load functionality

#### `phase4_train_dqn.py`

- **Training Loop**:
  - 1M step training with progress tracking
  - Evaluation every 50k steps (10 episodes)
  - Checkpoint saving every 100k steps
  - Comprehensive metrics collection

### 2. Analysis Notebook

#### `notebooks/phase4_dqn_training.ipynb`

- Load and analyze training results
- Compare with random baseline (146.95 ± 93.14)
- Visualize agent gameplay
- Action distribution analysis
- Performance comparison charts

## Training Parameters

| Parameter       | Value     | Purpose                          |
| --------------- | --------- | -------------------------------- |
| Total Steps     | 1,000,000 | Full training duration           |
| Batch Size      | 32        | Training batch size              |
| Buffer Size     | 100,000   | Replay buffer capacity           |
| Learning Starts | 50,000    | Steps before training begins     |
| Target Update   | 10,000    | Steps between target net updates |
| Epsilon Start   | 1.0       | Initial exploration rate         |
| Epsilon End     | 0.01      | Final exploration rate           |
| Epsilon Decay   | 0.9999    | Per-step decay rate              |
| Gamma           | 0.99      | Discount factor                  |
| Learning Rate   | 1e-4      | Adam optimizer                   |

## Expected Outputs

### Models

- `models/phase4/dqn_final.pt` - Final trained model
- `models/phase4/dqn_checkpoint_*.pt` - Intermediate checkpoints

### Data

- `data/phase4/training_metrics.json` - Complete training metrics

### Visualizations

- `outputs/phase4/training_results.png` - 4-panel training summary
- `outputs/phase4/learning_curve.png` - Performance vs baseline
- `outputs/phase4/gameplay_sample.png` - Agent behavior frames
- `outputs/phase4/action_distribution.png` - Action usage analysis
- `outputs/phase4/baseline_comparison.png` - Bar chart comparison

## How to Run

### Training (Takes Several Hours)

```bash
python src/phase4_train_dqn.py
```

**Note**: Training 1M steps will take:

- **CPU**: 8-12 hours (depending on CPU)
- **GPU**: 2-4 hours (much faster!)

### Quick Test (Optional)

For a quick test to ensure everything works:

1. Reduce `num_steps` to 100,000 in `phase4_train_dqn.py`
2. Reduce `learning_starts` to 10,000
3. This will complete in ~30 minutes on CPU

### Analysis

After training completes, open the notebook:

```bash
jupyter notebook notebooks/phase4_dqn_training.ipynb
```

## What You'll Learn

1. **Experience Replay**: How storing and randomly sampling past experiences breaks correlation and stabilizes learning

2. **Target Networks**: Why a separate, slowly-updated network prevents instability during training

3. **Epsilon-Greedy**: Balancing exploration (trying new actions) vs exploitation (using learned policy)

4. **Deep Q-Learning**: How neural networks can approximate Q-values from high-dimensional observations

## Expected Performance

Based on typical DQN results:

- **Random Baseline**: 146.95 ± 93.14
- **Expected DQN**: 300-500 (2-3x improvement)
- **Best Case**: 600-800 (if training goes well)

Your actual results may vary! That's part of the learning process in RL.

## Key Insights

### Why DQN Works

1. **CNN processes visual input**: Learns spatial features like enemy positions
2. **Experience replay**: Reuses past data efficiently
3. **Target network**: Prevents moving target problem
4. **Reward clipping**: Normalizes reward scale across games

### Common Issues

- **High variance**: Some runs learn faster than others (this is normal)
- **Slow initial learning**: Agent needs to fill replay buffer first
- **Performance plateaus**: May need more training or better hyperparameters

## What's Next: Phase 5

After Phase 4, you'll have a working DQN agent. Phase 5 will improve it with:

1. **Double DQN**: Reduce Q-value overestimation
2. **Dueling Networks**: Separate value and advantage streams
3. **Prioritized Replay**: Sample important transitions more often
4. **Noisy Networks**: Better exploration mechanism

These improvements typically add 20-50% more performance!

## Files Created in This Phase

```
src/
  ├── replay_buffer.py          # Experience replay buffer
  ├── dqn_agent.py              # DQN network and agent
  └── phase4_train_dqn.py       # Training script

notebooks/
  └── phase4_dqn_training.ipynb # Analysis notebook

models/phase4/                  # (created during training)
data/phase4/                    # (created during training)
outputs/phase4/                 # (created during training)
```

## Troubleshooting

### Out of Memory

- Reduce `buffer_size` to 50,000
- Reduce `batch_size` to 16
- Use CPU instead of GPU for smaller memory footprint

### Training Too Slow

- Reduce `num_steps` to 500,000 for faster results
- Use GPU if available
- Reduce `eval_freq` to 100,000

### Agent Not Learning

- Check that preprocessing is working (Phase 3)
- Verify replay buffer is filling up
- Try running for more steps
- Check training loss is decreasing

## References

- Mnih et al. (2015) - "Human-level control through deep reinforcement learning" (Nature)
- van Hasselt et al. (2016) - "Deep Reinforcement Learning with Double Q-learning"
- Schaul et al. (2016) - "Prioritized Experience Replay"

---

**Commit this phase**: Once training completes and you've analyzed results, commit everything to GitHub before moving to Phase 5!
