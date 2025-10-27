# Space Invaders Reinforcement Learning Project

A comprehensive RL portfolio project implementing and comparing various deep reinforcement learning algorithms on the Atari Space Invaders environment.

## Project Status

- [x] **Phase 1**: Environment Setup & Exploration
- [x] **Phase 2**: Data Collection & Baseline Performance
- [x] **Phase 3**: Preprocessing & Feature Engineering
- [ ] **Phase 4**: DQN Agent - Part 1 (Basic Implementation)
- [ ] **Phase 5**: DQN Agent - Part 2 (Improvements)
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
