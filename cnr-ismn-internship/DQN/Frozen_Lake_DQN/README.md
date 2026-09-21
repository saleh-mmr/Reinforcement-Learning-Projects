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


# **Implementation Details**

This project implements a complete **Deep Q-Network (DQN)** pipeline to train an agent to solve the **FrozenLake-v1** environment from Gymnasium.
Below is a detailed overview of the architecture, components, and training logic used in the implementation.

---

## **Environment: FrozenLake-v1**

FrozenLake-v1 is a grid-based environment where the agent must navigate from start **S** to goal **G** without falling into holes **H**.

* State space: **N discrete states** (e.g., 4×4 grid → 16 states)
* Action space: **4 actions** (up, down, left, right)
* Reward:

  * **+1** for reaching the goal
  * **0** otherwise

This implementation uses:

* `is_slippery=False` (deterministic)
* `max_episode_steps` adjustable
* One-hot state encoding to feed into the neural network

---

## **Architecture Overview**

The DQN system consists of:

1. **DQNNetwork (Neural Network)**
2. **ReplayMemory (Experience Buffer)**
3. **DQNAgent (Learning + Action Selection)**
4. **ModelTrainTest (Training & Testing Pipeline)**

Each component is explained in detail below.

---

# **Neural Network — `DQNNetwork`**

A simple feed-forward model:

```
State (one-hot) → FC(12) → ReLU → FC(8) → ReLU → FC(num_actions)
```

### **Purpose**

The network approximates the Q-function:

```
Q(s, a) ≈ expected return of taking action a in state s
```

### **Key features**

* He (Kaiming) initialization
* Output shape: **[num_actions]**
* Input shape: **[num_states]**

---

# **Experience Replay — `ReplayMemory`**

Stores transitions:

```
(state, action, next_state, reward, done)
```

Benefits:

✔ Breaks correlation between consecutive samples
✔ Stabilizes learning
✔ Allows mini-batch training

Replay memory uses **random sampling** to improve training stability.

---

# **DQN Agent — `DQNAgent`**

This is the core RL logic.
It contains:

* **Main network** – updated every step
* **Target network** – updated periodically
* **Epsilon-greedy policy** – for exploration
* **Learning algorithm** – computes loss + backprop

### **Action Selection (Epsilon-Greedy)**

```
if random < epsilon:
    explore (random action)
else:
    exploit (argmax Q)
```

### **Learning Step**

For each sampled transition:

```
Predicted Q: Q_main(s, a)

Target Q:
    if done:     target = r
    else:        target = r + γ * max_a' Q_target(s', a')
```

Loss:

```
MSE(predicted_q, target_q)
```

Then:

1. Backpropagation
2. Gradient clipping
3. Optimizer update

### **Target Network**

Updated with *hard update* every `update_frequency` steps:

```
target_network ← main_network
```

Prevents instability by keeping Q-targets fixed for a period.

---

# **Training Loop — `ModelTrainTest.train()`**

For each episode:

### **Step-by-step**

1. Reset environment
2. Convert state → one-hot
3. For each step:

   * Select action
   * Step environment
   * Store transition in replay memory
   * If enough samples → learn
   * Every N steps → update target network
4. Track rewards
5. Decay epsilon
6. Save model at intervals
7. Plot reward/loss curves


# **Key Techniques Implemented**

### ✔ Experience Replay

Prevents correlated updates.

### ✔ Target Network

Improves stability by separating prediction from target values.

### ✔ One-hot State Encoding

Simplifies learning for discrete FrozenLake states.

### ✔ Epsilon Annealing

Gradually shifts from exploration → exploitation.

### ✔ Gradient Clipping

Prevents exploding gradients.

### ✔ Deterministic Environment (is_slippery=False)

Agent receives consistent transitions, easier for learning.


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
https://github.com/MehdiShahbazi/DQN-Frozenlake-Gymnasium
