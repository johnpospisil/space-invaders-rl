# Space Invaders Reinforcement Learning Project

A comprehensive RL portfolio project implementing and comparing various deep reinforcement learning algorithms on the Atari Space Invaders environment.

## Project Status

- [x] **Phase 1**: Environment Setup & Exploration
- [x] **Phase 2**: Data Collection & Baseline Performance
- [x] **Phase 3**: Preprocessing & Feature Engineering
- [x] **Phase 4**: DQN Agent - Part 1 (Basic Implementation) - **506 max reward**
- [x] **Phase 5**: DQN Agent - Part 2 (Improvements) - Ready to train!
- [ ] **Phase 6**: Alternative Algorithms
- [ ] **Phase 7**: Hyperparameter Optimization
- [ ] **Phase 8**: Advanced Analysis & Visualization
- [ ] **Phase 9**: Model Interpretability
- [ ] **Phase 10**: Deployment & Documentation
- [ ] **Phase 11**: Transfer Learning (Bonus)
- [ ] **Phase 12**: Multi-Agent or Curriculum Learning (Bonus)

## Setup

```bash
pip install -r requirements.txt
```

## Phase 1: Environment Setup & Exploration

Run the initial exploration:

```bash
python src/phase1_environment_setup.py
```

This will:

- Load the Space Invaders environment
- Run a random agent for 5 episodes
- Display environment statistics
- Save sample frames

**Results**: Baseline established with mean reward of 119.0 ± 79.0

## Phase 2: Data Collection & Baseline Performance

Run comprehensive baseline data collection:

```bash
python src/phase2_baseline_collection.py
```

This will:

- Run random agent for 100 episodes
- Collect detailed performance metrics
- Generate comprehensive statistical analysis
- Create visualization dashboards
- Save baseline data for comparison

**Results**:

- Mean Reward: 146.95 ± 93.14
- Mean Episode Length: 504.4 ± 155.9
- Coefficient of Variation: 0.63 (high variance)

## Phase 3: Preprocessing & Feature Engineering

Run preprocessing demonstration:

```bash
python src/phase3_preprocessing_demo.py
```

This will:

- Demonstrate preprocessing pipeline steps
- Compare raw vs preprocessed observations
- Visualize frame stacking and transformations
- Test preprocessed environment
- Create reusable preprocessing utilities

**Key Features**:

- 72% reduction in pixel dimensions (100,800 → 28,224)
- Grayscale conversion + resize to 84×84
- Frame stacking (4 frames) for temporal information
- Reward clipping to {-1, 0, 1}
- Normalization to [0, 1]
- Follows DQN Nature paper methodology

**Preprocessing Module**: `src/preprocessing.py` provides reusable wrappers for all phases

## Phase 4: DQN Agent - Part 1 (Basic Implementation)

Train a basic DQN agent:

```bash
python src/phase4_train_dqn.py
```

This will:

- Train DQN agent for 1M steps
- Implement experience replay with 100k buffer
- Use epsilon-greedy exploration (1.0 → 0.01)
- Save checkpoints every 100k steps
- Evaluate agent every 50k steps
- Generate training curves and visualizations

**Architecture**:

- CNN: 3 conv layers (32→64→64 filters) + 2 FC layers (512→n_actions)
- Optimizer: Adam with learning rate 1e-4
- Loss: Huber loss with gradient clipping
- Target network updated every 10k steps
- Follows Nature DQN paper (Mnih et al., 2015)

**Training Parameters**:

- Total steps: 1,000,000
- Batch size: 32
- Replay buffer: 100,000 transitions
- Learning starts: 50,000 steps
- Gamma (discount): 0.99

**Note**: Training will take several hours on CPU, much faster with GPU.

**Analysis**: Use `notebooks/phase4_dqn_training.ipynb` to analyze results

**Phase 4 Results**: Max reward 506 (244% improvement over baseline!)

## Phase 5: DQN Improvements

Train improved DQN with multiple enhancements:

```bash
python src/phase5_train_improved_dqn.py
```

Or for quick test (~30 min):

```bash
python run_phase5_quick_test.py
```

**Improvements**:

1. **Double DQN**: Reduces Q-value overestimation by using online network for action selection
2. **Dueling Networks**: Separates value function V(s) and advantage function A(s,a)
3. **Prioritized Experience Replay**: Samples important transitions more frequently

**Architecture**:

- Dueling network with separate value and advantage streams
- Same CNN backbone as Phase 4
- Prioritized replay with α=0.6, β annealing from 0.4 to 1.0
- All Phase 4 hyperparameters maintained for fair comparison

**Expected Results**: 10-30% improvement over Phase 4 (~560-660 reward)

## Project Structure

```
space_invaders_rl/
├── src/                    # Source code
├── notebooks/              # Jupyter notebooks for analysis
├── data/                   # Data and logs
├── models/                 # Saved models
├── outputs/                # Visualizations and results
└── requirements.txt        # Dependencies
```

## Author

Data Science Portfolio Project - 2025
