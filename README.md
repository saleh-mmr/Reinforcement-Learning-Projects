# **Deep Q-Learning on FrozenLake-v1**

This project implements a complete **Deep Q-Network (DQN)** for solving the **FrozenLake-v1** environment from Gymnasium.
It supports both **training** and **testing**, including experience replay, target network updates, epsilon-greedy exploration, and learning curve plotting.

---

## **Project Structure**

```
DQN/
│
├── run.py
├── model_train_test.py
├── dqn_agent.py
├── dqn_network.py
├── replay_memory.py
├── config.py
│
└── 4x4_weights/           # (Created during training)
      final_weights_3000.pth
```

---

## **Features**

* Deep Q-Learning with:

  * Experience Replay
  * Target Network
  * Epsilon-Greedy Exploration
  * Mini-batch Learning
* One-hot state encoding for FrozenLake
* Save & load trained models
* Training reward and loss plots
* Works on 4×4 or 8×8 FrozenLake maps

---

## **Training**

Enable training mode in `run.py`:

```python
train_mode = True
```

Then run:

```bash
python run.py
```

During training:

* models are saved every interval (default: 500 episodes)
* final model weights go to:

```
4x4_weights/final_weights_3000.pth
```

* reward and loss curve plots are generated automatically

---

## **Testing**

After training, switch to:

```python
train_mode = False
```

And specify the trained model path:

```python
"RL_load_path": "./4x4_weights/final_weights_3000.pth"
```

Then run:

```bash
python run.py
```

The agent will play FrozenLake using the learned policy.

---

## **Hyperparameters**

You can modify them in `run.py`:

```python
learning_rate     # Q-network optimizer LR
discount_factor   # γ discount
batch_size        # Replay mini-batch size
update_frequency  # Target network update rate
max_episodes      # Total episodes
memory_capacity   # Replay buffer size
max_epsilon       # Initial exploration rate
min_epsilon       # Final exploration rate
epsilon_decay     # Exploration decay
```

---

## **Outputs**

Training generates:

* `reward_plot.png`
* `loss_plot.png`
* saved models in `4x4_weights/`

---

## **DQN Overview**

The agent learns Q-values using:

```
y = r + γ * max_a' Q_target(s', a')
loss = MSE( Q_main(s, a), y )
```

It uses:

* **main network** for current Q-values
* **target network** (periodically updated) for stable training
* **replay memory** to break temporal correlation

---

## **Requirements**

Install dependencies:

```bash
pip install torch gymnasium pygame matplotlib numpy
```

---

## **License**
MIT License

Copyright (c) 2024 Mehdi Shahbazi Khojasteh
