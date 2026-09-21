# Multi-Weight-Molecular-Spintronic-Synapses-For-Reinforcement-Learning

A hardware-aware Deep Q-Learning project that simulates multi-weight molecular spintronic synapses for solving standard and modified CartPole reinforcement learning tasks.

## Overview

This repository explores how reinforcement learning can be adapted to emerging in-memory and neuromorphic hardware. The project uses a Deep Q-Network (DQN) agent for the CartPole control problem, but replaces the standard optimizer update with a custom conductance-inspired synaptic update rule.

The main idea is to represent neural-network weights using physical conductance states. A multi-weight molecular spintronic synapse can generate different effective weights for different tasks by combining resistive switching and magnetic-state selection.

## Key Ideas

- Deep Q-Learning for reinforcement learning
- Standard and modified CartPole environments
- Multi-task learning with task-specific effective weights
- Conductance-based synaptic weight representation
- Molecular spintronic synapse model
- Resistive switching for tuning conductance values
- Magnetoresistance for selecting parallel and antiparallel states
- Hardware-aware Manhattan-style update rule using gradient signs

## Method

The project uses one shared DQN architecture for two related CartPole tasks:

- Task 0: Standard CartPole
- Task 1: Modified CartPole with changed physical parameters

Each trainable network parameter is represented by a multi-weight synapse. Before action selection or learning, the synaptic controller loads the effective weight for the active task.

For two tasks, the effective weights are modeled as:

```text
w0 = alpha * (G_AP,0 + G_P,1  - G_bias)
w1 = alpha * (G_P,0  + G_AP,1 - G_bias)
```

The conductance values are calculated from an internal programmed state:

```text
G(x) = g * log10(x)
```

Instead of applying a floating-point optimizer step, the model uses only the sign of the gradient:

```text
Positive gradient -> increase bias conductance -> decrease weight
Negative gradient -> increase active task conductance -> increase weight
Zero gradient     -> no update
```

## Training Workflow

1. Initialize two CartPole environments.
2. Store experiences in separate replay memories.
3. Select actions using an epsilon-greedy policy.
4. Compute Bellman targets and MSE loss.
5. Use backpropagation to calculate gradients.
6. Apply sign-based synaptic updates through the custom controller.
7. Save training logs, model checkpoints, and evaluation results.

## Results

The DQN agent successfully learns both CartPole tasks. Training rewards gradually improve, and testing shows stable performance with maximum reward over repeated evaluation episodes. Weight and bias heatmaps show that the same shared architecture produces different task-specific effective parameters.

## Technologies

- Python
- PyTorch
- Gymnasium
- NumPy
- Matplotlib
- Reinforcement Learning
- Neuromorphic Computing
- Spintronic Synapse Modeling

## Repository Structure

```text
project
├── agent/       # Source code for DQN Agent with learn and selct_action method
├── controller/  # MultiWeightSynapse controller, a custumized optimizer
├── devices/     # design of Resistive Switching and Magnetoresistance of devices
├── env/         # Gymnasium envirnment
├── learning/    # core of training and testing loops
├── memory/      # buffers for storing transactions(states, actions, rewards, next states and dones)
├── network/     # DQN a basic neural network
├── scripts/     # tuning hyperparameters and run script
└── utils/       # config
```

## How to Run

Install the required packages:
 - pandas
 - numpy
 - matplotlib
 - gymnasium

Run training in script folder:

```bash
python train_script.py
```

Run evaluation in script folder:

```bash
python cartpole_test_script.py
```

## Project Goal

The goal of this project is to study how Deep Q-Learning can be made more compatible with physical synaptic hardware. Instead of relying only on ideal floating-point updates, the project uses a simplified hardware-aware update rule based on conductance changes and gradient signs.

## Suggested Repository Topics

```text
reinforcement-learning
deep-q-learning
cartpole
neuromorphic-computing
spintronics
memristive-devices
in-memory-computing
hardware-aware-ai
pytorch
gymnasium
```
