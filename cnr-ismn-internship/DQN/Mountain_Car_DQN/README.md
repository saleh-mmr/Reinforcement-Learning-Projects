# Deep Q-Learning for MountainCar-v0 (Gymnasium)

This project implements a Deep Reinforcement Learning (DRL) agent to solve the **MountainCar-v0** environment from the Gymnasium library.
The implementation includes core RL components such as experience replay, target networks, environment wrappers, and an enhanced version of Q-learning using **Double DQN**.

The project is structured with modular, reusable components that allow easy modification of network architectures, exploration strategies, or reward shaping.

---

## Table of Contents

1. Introduction
2. Environment Description
3. Project Architecture
4. Observation and Reward Wrappers
5. Deep Q-Network (DQN)
6. Double DQN: Motivation and Implementation
7. Training Procedure
8. Testing Procedure
9. Hyperparameters
10. Installation and Usage
11. File Structure
12. References

---

## 1. Introduction

The MountainCar-v0 environment presents a classic RL challenge where the agent must drive an underpowered car up a steep hill. The environment requires the agent to learn *non-myopic* behavior: moving left to gain momentum before successfully climbing the right hill.

Traditional approaches rely on tabular methods (such as SARSA or Q-learning) with hand-crafted features. This project extends the problem to the function-approximation regime using **Deep Q-Learning (DQN)** and its improved variant **Double DQN**, enabling the agent to learn from continuous state spaces.

The project includes:

* A replay memory buffer
* A neural-network-based Q-function
* A target network for stable learning
* Custom observation normalization
* Customizable reward shaping
* Double DQN for improved value estimation

---

## 2. Environment Description

MountainCar-v0 characteristics:

* Continuous state space: position and velocity
* Discrete action space: [push left, no push, push right]
* Sparse and delayed rewards
* Requires momentum-building behavior

The goal is to reach the flag at the top of the right hill positioned at approximately 0.5.
Episodes terminate after 200 steps if the goal is not reached.

---

## 3. Project Architecture

The project is composed of the following modules:

### ReplayMemory

Stores agent experiences `(state, action, next_state, reward, done)` and samples randomized mini-batches for training.

### DQNNetwork

Defines a fully connected neural network mapping state vectors to Q-values for each possible action.

### DQNAgent

Handles:

* Action selection via epsilon-greedy policy
* Learning updates
* Gradient clipping
* Target network synchronization
* Double DQN logic

### Environment Wrappers

* Observation normalization
* Reward shaping
* Combined wrapper for maintaining consistent agent inputs

### ModelTrainTest

High-level controller that manages:

* Training loops
* Logging
* Model saving
* Plotting
* Testing

### Main Entry Point

Configures hyperparameters and triggers training or testing.

---

## 4. Observation and Reward Wrappers

### ObservationWrapper

Normalizes raw environment observations from their physical ranges:

position in [-1.2, 0.6], velocity in [-0.07, 0.07]

to normalized values in [0, 1] for stable neural network inputs.

### RewardWrapper

Implements reward shaping techniques to provide dense learning signals.
Reward shaping encourages:

* Momentum generation
* Climbing progress
* Near-goal rewards
* Reduced step penalties

The wrapper converts normalized states back to physical coordinates before computing shaped rewards.

The final reward function is designed to stabilize learning while encouraging physically meaningful behavior, such as swinging to build momentum.

---

## 5. Deep Q-Network (DQN)

DQN approximates the action-value function:

Q(s, a; θ)

using a neural network with parameters θ.
Key features:

* Off-policy learning
* Experience replay
* Target network for stable training

### Vanilla DQN Target Update

In the original DQN algorithm, the target value is computed using:

y = r + γ max_a' Q_target(s', a')

where Q_target is a periodically updated copy of the main Q-network.

This formulation tends to produce **overestimation bias**, because the same network selects and evaluates the maximizing action.

---

## 6. Double DQN: Motivation and Implementation

### Overestimation in DQN

Vanilla DQN uses the target network both to:

1. Select the best next action via `argmax`
2. Evaluate the Q-value of that action

Because neural approximators contain noise, the max operator tends to select overestimated Q-values. Over many updates, this leads to systematic overestimation bias, slowing or preventing convergence.

This bias is amplified in tasks like MountainCar, where:

* Optimal behavior requires long-term planning
* Q-values are initially very similar across actions
* Small estimation errors dominate early learning

### Double DQN Solution

Double DQN decouples action selection from action evaluation.

Let θ be parameters of the main network, and θ' parameters of the target network.

### Double DQN Target Update

The improved target is:

y = r + γ Q_target(s', argmax_a Q_main(s', a; θ); θ')

This breaks the positive feedback loop responsible for overestimation.

### Implementation Change

In the learning step, the following was replaced:

Original DQN:

```python
next_target_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]
```

Double DQN:

```python
next_actions = self.main_network(next_states).argmax(dim=1, keepdim=True)
next_target_q = self.target_network(next_states).gather(1, next_actions)
```

This modification significantly stabilizes learning and is one of the key reasons the agent successfully learns the momentum-based strategy required to solve MountainCar.

---

## 7. Training Procedure

Training is performed over multiple episodes. For each episode:

1. Reset environment using the wrapped observation and reward functions
2. Select actions using an epsilon-greedy policy
3. Store transitions in replay memory
4. Begin learning once memory contains sufficient samples
5. Update the Q-network using Double DQN targets
6. Periodically update the target network
7. Log episode rewards and loss
8. Save models at defined intervals
9. Optionally render selected episodes for visualization

Epsilon decays over time to shift from exploration to exploitation.

---

## 8. Testing Procedure

Testing loads a pre-trained network and runs the environment without learning.
The agent selects actions exclusively via greedy policy, and rendering can be enabled to visualize its behavior.

The evaluation logs:

* Episode returns
* Steps taken
* Completion behavior

---

## 9. Hyperparameters

Typical hyperparameters used in this project include:

* Learning rate: 1e-4
* Discount factor: 0.99
* Batch size: 64–128
* Replay memory capacity: 100,000+
* Epsilon decay: 0.995–0.999
* Epsilon min: 0.01–0.02
* Target update frequency: 50–200 steps
* Gradient clipping: 5–10
* Maximum episodes: 900
* Maximum steps per episode: 200

These parameters may be adjusted depending on reward shaping and performance needs.

---

## 10. Installation and Usage

### Requirements

* Python 3.8+
* PyTorch
* Gymnasium
* NumPy
* Matplotlib
* Pygame

Install dependencies:

```
pip install -r requirements.txt
```

### Train the model:

```
python main.py
```

### Test the trained model:

Adjust the configuration in main.py:

```
train_mode = False
```

Then run:

```
python main.py
```

---

## 11. File Structure

```
DQN/
│
├── MountainCar/
│   ├── dqn_network.py
│   ├── dqn_agent.py
│   ├── replay_memory.py
│   ├── observation_wrapper.py
│   ├── reward_wrapper.py
│   ├── step_wrapper.py
│   ├── model_train_test.py
│   ├── config.py
│   ├── run.py
│
├── weights/
├── plots/
└── README.md
```

---

## 12. References

1. Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.
2. van Hasselt, H., Guez, A., & Silver, D. (2015). Deep Reinforcement Learning with Double Q-learning. AAAI.
3. Sutton, R. S., & Barto, A. G. Reinforcement Learning: An Introduction (2nd ed.).
4. Gymnasium Documentation: [https://gymnasium.farama.org](https://gymnasium.farama.org)
