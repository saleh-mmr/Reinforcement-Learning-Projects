#  Cartpole Training

A PyTorch-based Deep Q-Network (DQN) implementation for training RL agents on classic control environments from OpenAI Gymnasium. This project provides training pipeline for CartPole environments with support for model checkpointing and visualization.

## Project Overview

This project implements a DQN agent that learns to solve control tasks through reinforcement learning. The agent uses:
- **Deep Q-Learning**: A value-based reinforcement learning algorithm
- **Experience Replay**: Samples from a memory buffer to break correlations in training data
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation during learning
- **Neural Networks**: Feed-forward networks to approximate Q-value functions

## Project Structure

```
MountainCar_Cartpole_Separate_Training/
├── agents/                    # DQN agent implementation
│   └── agent.py              # Main DQNAgent class with learning logic
├── envs/                      # Environment wrappers
│   └── cartpole.py           # CartPole-v1 environment wrapper
├── learning/                  # Training orchestration
│   └── trainer.py            # Trainer class coordinating training loops
├── memory/                    # Experience replay buffer
│   └── replay_memory.py       # ReplayMemory class
├── network/                   # Neural network architectures
│   └── network.py            # DQNNetwork (feed-forward Q-network)
├── scripts/                   # Entry points and evaluation scripts
│   ├── script.py             # Main training/evaluation script
│   └── best_model_seed_49.pth # Saved model checkpoint
├── utils/                     # Configuration and utilities
│   └── config.py             # Global device configuration
└── README.md                  # This file
```

## Components

### Agent (`agents/agent.py`)
- **DQNAgent**: Implements the core DQN learning algorithm
  - Maintains a Q-network for action-value estimation
  - Uses experience replay memory for stable learning
  - Implements epsilon-greedy exploration strategy
  - Optimizes network weights using RMSprop optimizer

### Environments (`envs/cartpole.py`)
- **CartPoleEnv**: Wrapper around Gymnasium's CartPole-v1
  - Configurable time limits per episode
  - Reproducible seeds for deterministic behavior
  - Exposes action and observation spaces

### Training (`learning/trainer.py`)
- **Trainer**: Coordinates the training pipeline
  - Manages training episodes and hyperparameters
  - Tracks rewards and loss over time
  - Saves best-performing model checkpoints
  - Handles model evaluation

### Memory (`memory/replay_memory.py`)
- **ReplayMemory**: Experience replay buffer
  - Stores state transitions (state, action, reward, next_state, done)
  - Supports random sampling for mini-batch training
  - Configurable capacity with FIFO eviction

### Network (`network/network.py`)
- **DQNNetwork**: Feed-forward neural network
  - Two hidden layers with ReLU activation
  - Outputs Q-values for each action
  - PyTorch Module for GPU/CPU compatibility

### Configuration (`utils/config.py`)
- Global device management (CUDA/CPU)
- CUDA optimization flags for debugging
- Memory management

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Gymnasium
- NumPy
- Matplotlib

### Setup

1. Clone or navigate to the project directory
2. Install dependencies:
```bash
pip install torch gymnasium numpy matplotlib
```

## Usage

### Training a New Model

Edit `scripts/script.py` to set `train_mode = True`:

```bash
python scripts/script.py
```

This will:
- Train a DQN agent on CartPole-v1 for 600 episodes
- Track rewards and loss across episodes
- Save the best model checkpoint
- Generate performance plots

### Evaluating a Saved Model

Set `train_mode = False` in `scripts/script.py` and run:

```bash
python scripts/script.py
```

## Hyperparameters

Configure hyperparameters in `scripts/script.py`:

```python
hyperparams = {
    "discount_factor": 0.99,      # Gamma - future reward discount rate
    "batch_size": 100,            # Mini-batch size for training
    "memory_capacity": 10000,     # Max replay memory size
    "max_episodes": 600,          # Total training episodes
    "goal": 200,                  # Target episode reward/time limit
    "network_size": 20,           # Hidden layer size
    "epsilon_max": 1.0,           # Initial exploration rate
    "epsilon_min": 0.01,          # Final exploration rate
    "epsilon_decay": 0.00005,     # Decay rate per step
}
```

## Training Details

### DQN Algorithm
The agent learns by:
1. Taking actions using epsilon-greedy policy
2. Storing transitions in replay memory
3. Sampling mini-batches from memory
4. Computing target Q-values with future rewards
5. Minimizing MSE loss between predicted and target Q-values
6. Updating epsilon for annealed exploration

### Loss Function
Mean Squared Error (MSE) between predicted Q-values and TD targets

### Optimizer
RMSprop with learning rate 0.001

## Model Checkpoint

Trained model weights are saved in `scripts/best_model_seed_49.pth` using PyTorch's state_dict format. This allows quick evaluation without retraining.

## Performance

The agent learns to achieve high rewards on CartPole-v1 through DQN training. Performance is tracked via:
- Episode rewards over time
- Loss curves during training
- Running average reward windows
- 
## Notes

- GPU/CPU device is automatically detected and configured
- Seeds ensure reproducible training runs
- All paths use relative imports for modularity
- CUDA synchronization is enabled for easier debugging
