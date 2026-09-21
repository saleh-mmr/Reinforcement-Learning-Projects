# QMIX Algorithm: Theory and Implementation Report

## 1. QMIX Algorithm Overview

### 1.1 What is QMIX?

**QMIX** (Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning) is a value-based multi-agent reinforcement learning algorithm introduced by Rashid et al. (2018). It addresses the challenge of learning coordinated policies in cooperative multi-agent environments while maintaining the ability to execute policies in a decentralized manner.

### 1.2 The Core Problem

In multi-agent reinforcement learning, we need to balance two competing requirements:

1. **Centralized Training**: During training, agents can benefit from global information (full state) to learn coordinated strategies.
2. **Decentralized Execution**: During execution, each agent must act based only on its local observation, without communication.

**The Challenge**: How can we train agents with global information while ensuring they can act independently during execution?

### 1.3 QMIX's Solution

QMIX solves this by:
- Learning **individual agent Q-functions** Q_i(o_i, a_i) that depend only on local observations
- Learning a **mixing network** that combines these Q-values into a joint Q-value Q_tot(s, a_1, ..., a_n)
- Enforcing a **monotonicity constraint**: ∂Q_tot/∂Q_i ≥ 0 for all agents i

This constraint ensures that:
- If an agent's Q-value increases, the joint Q-value also increases
- The optimal joint action can be found by each agent independently selecting argmax_a_i Q_i(o_i, a_i)
- Decentralized execution is valid while benefiting from centralized training

---

## 2. How QMIX Works

### 2.1 Architecture Components

QMIX consists of three main components:

#### 2.1.1 Agent Q-Networks
- Each agent i has its own Q-network: Q_i(o_i, a_i)
- Takes local observation o_i as input
- Outputs Q-values for each action a_i
- **Decentralized**: Each agent's network only sees its own observation

#### 2.1.2 Mixing Network
- Takes individual agent Q-values [Q_1, Q_2, ..., Q_n] and global state s
- Combines them into joint Q-value: Q_tot(s, a_1, a_2, ..., a_n)
- Uses **hypernetworks** to generate state-dependent mixing weights
- **Centralized**: Has access to global state during training

#### 2.1.3 Centralized Replay Buffer
- Stores transitions: (s, o_1, ..., o_n, a_1, ..., a_n, r, s', o'_1, ..., o'_n, done)
- Includes both local observations and global state
- Enables off-policy learning with experience replay

### 2.2 The Monotonicity Constraint

The key innovation of QMIX is the **monotonicity constraint**:

**Mathematical Formulation:**
```
∂Q_tot/∂Q_i ≥ 0  for all agents i
```

**What this means:**
- The joint Q-value Q_tot is monotonically non-decreasing in each individual Q-value Q_i
- If agent i's Q-value increases, the joint Q-value cannot decrease
- This ensures that maximizing each Q_i independently will maximize Q_tot

**Why this matters:**
- Enables **decentralized execution**: Each agent can greedily select argmax_a_i Q_i(o_i, a_i)
- The resulting joint action will be optimal for Q_tot
- No need for communication or centralized action selection during execution

### 2.3 Hypernetworks

QMIX uses **hypernetworks** to enforce monotonicity:

- **Hypernetworks** are neural networks that generate weights for another network
- In QMIX, hypernetworks take the global state s as input
- They generate **positive weights** for the mixing network
- Positive weights ensure monotonicity: if all weights are positive, increasing any Q_i increases Q_tot

**Architecture:**
```
State s → Hypernetwork → Positive Weights W(s) → Mixing Network
```

### 2.4 Training Process

QMIX uses **centralized training with decentralized execution (CTDE)**:

1. **Collect Experience**: Agents interact with environment, storing transitions in centralized buffer
2. **Compute Q_tot**: For each transition, compute Q_tot(s, a_1, ..., a_n) using mixing network
3. **Compute Target**: Use target networks to compute Q_tot_target(s', a'_1, ..., a'_n)
4. **Update**: Minimize loss L = (Q_tot - (r + γ * Q_tot_target))²
5. **Update Targets**: Periodically copy main networks to target networks

**Key Insight**: The loss is computed on Q_tot, but gradients flow back to both:
- Agent Q-networks (through mixing network)
- Mixing network itself

This allows agents to learn coordinated strategies while maintaining decentralized execution capability.

### 2.5 Action Selection

**During Training (Exploration):**
- Each agent uses epsilon-greedy: select random action with probability ε, else argmax_a_i Q_i(o_i, a_i)

**During Execution (Exploitation):**
- Each agent greedily selects: a_i* = argmax_a_i Q_i(o_i, a_i)
- No communication needed - each agent acts independently
- Monotonicity ensures this joint action maximizes Q_tot

---

## 3. Implementation Details

### 3.1 Project Structure

The QMIX implementation is organized into several key files:

#### Core Algorithm Files:
- **`src/algos/qmix.py`**: Main QMIX coordinator class
- **`src/algos/qmix_agent.py`**: Individual agent Q-networks
- **`src/models/mixing_network.py`**: Mixing network with hypernetworks
- **`src/utils/qmix_replay_memory.py`**: Centralized replay buffer

#### Model Files:
- **`src/models/qvalue_network.py`**: Base Q-network architecture (shared with IQL/PS-DQN)

### 3.2 Key Implementation Components

#### 3.2.1 QMIXAgent (`qmix_agent.py`)

Each agent maintains:
- **Main Q-network**: Q_i(o_i, a_i) - estimates Q-values from local observations
- **Target Q-network**: Used for stable target computation
- **Action selection**: Epsilon-greedy policy

**Key Methods:**
- `select_action(obs, epsilon)`: Epsilon-greedy action selection
- `update_target_network()`: Copy main network to target

#### 3.2.2 MixingNetwork (`mixing_network.py`)

The mixing network implements the core QMIX mechanism:

**Architecture:**
```python
# Hypernetworks generate positive weights from state
hyper_w1: state → [n_agents × mixing_hidden_dim] weights
hyper_b1: state → [mixing_hidden_dim] bias
hyper_w2: state → [mixing_hidden_dim] weights  
hyper_b2: state → [1] bias

# Forward pass:
1. Generate positive weights: w1 = softplus(hyper_w1(state))
2. First mixing layer: hidden = ELU(Q_values @ w1 + b1)
3. Generate second layer weights: w2 = softplus(hyper_w2(state))
4. Final mixing: Q_tot = hidden @ w2 + b2
```

**Monotonicity Enforcement:**
- Uses `F.softplus()` to ensure all weights are strictly positive
- `softplus(x) = log(1 + exp(x))` is always > 0
- Better gradient flow than `abs()` for training stability

#### 3.2.3 QMIXReplayMemory (`qmix_replay_memory.py`)

Stores centralized transitions:
- **Global state** s: Full environment state
- **Local observations** o_i: Per-agent observations
- **Actions** a_i: Per-agent actions
- **Joint reward** r: Shared reward
- **Next state/observations**: For target computation
- **Done flag**: Episode termination

**Key Feature**: Deep-copies all inputs to prevent silent mutation that would corrupt training.

#### 3.2.4 QMIX Coordinator (`qmix.py`)

Orchestrates the entire algorithm:

**Initialization:**
- Creates n_agents QMIXAgent instances
- Creates main and target MixingNetwork instances
- Sets up optimizer with separate learning rates:
  - Agent networks: `learning_rate` (default: 1e-3)
  - Mixing network: `learning_rate * 0.5` (default: 5e-4)
- Initializes centralized replay buffer

**Training Step (`train_step()`):**

1. **Sample Batch**: Sample transitions from centralized buffer
2. **Extract Data**: Separate global states, observations, actions, rewards, etc.
3. **Current Q-values**: Compute Q_i(o_i, a_i) for each agent using main networks
4. **Compute Q_tot**: Mix agent Q-values using main mixing network
5. **Target Computation (Double-Q Learning)**:
   - Select next actions using **online networks**: a'_i = argmax Q_i_online(o'_i)
   - Evaluate using **target networks**: Q'_i = Q_i_target(o'_i, a'_i)
   - Mix target Q-values: Q_tot_target = MixingNetwork_target(Q'_1, ..., Q'_n, s')
6. **Compute Target**: r + γ * Q_tot_target * (1 - done)
7. **Compute Loss**: MSE(Q_tot, target)
8. **Backward Pass**: Compute gradients
9. **Gradient Clipping**: Clip all gradients (agent + mixing) to max_norm=10.0
10. **Optimizer Step**: Update all parameters

**Key Implementation Details:**

- **Double-Q Learning**: Reduces overestimation bias by using online networks for action selection and target networks for evaluation
- **Gradient Clipping**: Prevents exploding gradients by clipping norm across all parameters
- **State Normalization**: Global states are normalized to [0, 1] range by dividing by grid_size
- **Deep Copying**: All replay buffer inputs are deep-copied to prevent mutation

**Target Network Updates:**
- Updated every `target_update_freq` episodes (default: 50)
- Episode-based updates provide more stable learning than step-based

### 3.3 Training Loop

The training process follows this structure:

```python
for episode in range(max_episodes):
    obs, info = env.reset()
    state = get_global_state(env)
    
    for step in range(max_steps):
        # 1. Select actions (decentralized)
        actions = select_actions(obs, epsilon)
        
        # 2. Step environment
        next_obs, reward, done, info = env.step(actions)
        next_state = get_global_state(env)
        
        # 3. Store transition (centralized)
        store_transition(state, obs, actions, reward, next_state, next_obs, done)
        
        # 4. Train if buffer is large enough
        if len(buffer) >= min_buffer_size:
            loss = train_step()  # Centralized training on Q_tot
        
        # 5. Update state
        obs = next_obs
        state = next_state
    
    # 6. Update target networks periodically
    if episode % target_update_freq == 0:
        update_target_networks()
```

### 3.4 Hyperparameters

**Network Architecture:**
- Agent Q-networks: `input_dim (6) → hidden_dim (64) → hidden_dim (64) → num_actions (5)`
- Mixing network hidden dimension: `128` (increased from 64 for better capacity)

**Training Hyperparameters:**
- Learning rate: `1e-3` (agents), `5e-4` (mixing network)
- Batch size: `32`
- Replay buffer capacity: `10000`
- Discount factor (γ): `0.99`
- Target update frequency: `50 episodes`
- Min buffer size (warm-up): `3000`
- Epsilon decay: `1.0 → 0.05` over `75000` steps

**Stabilization Techniques:**
- Double-Q learning for target action selection
- Gradient clipping (max_norm=10.0) across all parameters
- Deep-copying replay buffer inputs
- State normalization to [0, 1] range
- Separate learning rates for agent and mixing networks
- Episode-based target network updates

### 3.5 Evaluation

During evaluation:
- Epsilon is set to 0 (greedy policy)
- Each agent independently selects: a_i = argmax Q_i(o_i, a_i)
- No communication or centralized coordination needed
- Success rate, episode length, and return are computed

---

## 4. Key Differences from IQL and PS-DQN

| Aspect | IQL | PS-DQN | QMIX |
|--------|-----|--------|------|
| **Network Structure** | Separate Q-networks per agent | Single shared Q-network | Separate Q-networks + Mixing network |
| **Training Signal** | Individual Q_i losses | Shared Q-network loss | Joint Q_tot loss |
| **State Information** | Local observations only | Local observations only | Global state + local observations |
| **Coordination** | Implicit (through environment) | Implicit (through sharing) | Explicit (through mixing network) |
| **Execution** | Decentralized | Decentralized | Decentralized |
| **Monotonicity** | N/A | N/A | Enforced via positive weights |

**QMIX Advantages:**
- Explicit coordination through mixing network
- Can leverage global state during training
- Maintains decentralized execution
- Theoretically grounded (monotonicity constraint)

**QMIX Challenges:**
- More complex architecture (mixing network + hypernetworks)
- Requires global state information
- More hyperparameters to tune
- Computationally more expensive

---

## 5. Summary

QMIX is a sophisticated MARL algorithm that enables centralized training while maintaining decentralized execution. The key innovation is the monotonicity constraint enforced through hypernetworks, which ensures that agents can act independently while still learning coordinated strategies.

**Theoretical Foundation:**
- Monotonic value function factorization
- Centralized training, decentralized execution (CTDE)
- Hypernetworks for state-dependent mixing

**Implementation Highlights:**
- Separate agent Q-networks for decentralized execution
- Mixing network with hypernetworks for coordination
- Centralized replay buffer with global state
- Double-Q learning and gradient clipping for stability
- State normalization and deep-copying for robustness

The implementation successfully applies QMIX to the Meeting Gridworld task, demonstrating how the algorithm can learn coordinated multi-agent policies while maintaining the ability to execute them in a fully decentralized manner.

