# Multi-Agent Reinforcement Learning: IQL and PS-DQN Implementation Report

## 1. Problem Statement and Goal

### Problem
The task is to solve a **cooperative multi-agent meeting problem** in a gridworld environment. Two agents must coordinate to simultaneously reach a shared goal location. This requires:
- **Coordination**: Both agents must arrive at the goal at the same time
- **Efficient navigation**: Agents must learn optimal paths while avoiding obstacles
- **Temporal coordination**: Agents must synchronize their movements to meet at the goal

### Environment: Meeting Gridworld
- **Grid size**: 5×5 discrete grid
- **Agents**: 2 agents that move simultaneously
- **Actions**: 5 actions per agent (up, down, left, right, stay)
- **Observations**: Each agent observes its own position, other agent's position, and goal position (6-dimensional vector)
- **Reward**: 
  - +10.0 if both agents reach the goal simultaneously
  - -0.01 per step (encourages efficiency)
- **Episode termination**: Success when both agents are on the goal, or timeout after 50 steps

### Goal
The goal is to compare different multi-agent reinforcement learning algorithms:
1. **Independent Q-Learning (IQL)**: Baseline where agents learn independently
2. **Parameter-Shared DQN (PS-DQN)**: Agents share a single Q-network
3. Evaluate which approach achieves higher success rates and better coordination

---

## 2. Independent Q-Learning (IQL)

### Algorithm Overview
**Independent Q-Learning (IQL)** is a baseline MARL algorithm where each agent learns independently, treating other agents as part of the environment. This approach:
- Treats the multi-agent problem as multiple single-agent problems
- Each agent maintains its own Q-function: Q_i(o_i, a_i)
- Agents learn from their own experiences independently
- Simple but effective, though may struggle with non-stationarity

### Key Characteristics
- **Decentralized learning**: Each agent has its own Q-network, optimizer, and replay buffer
- **Independent policies**: Agents don't explicitly coordinate during training
- **Non-stationarity issue**: As agents learn, the environment changes (other agents' policies change), making learning non-stationary

### Implementation Structure

#### Core Files

**`src/algos/iql.py`** (Main Algorithm Coordinator)
- `IQL` class: Orchestrates multiple independent agents
- Manages training loop, evaluation, and TensorBoard logging
- Coordinates agent actions and collects statistics
- Handles epsilon decay schedule across all agents
- **Key methods**:
  - `select_actions()`: Collects actions from all agents
  - `store_transitions()`: Stores experiences in each agent's buffer
  - `train()`: Main training loop with evaluation and logging

**`src/algos/iql_agent.py`** (Individual Agent Component)
- `IQLAgent` class: Represents a single independent agent
- Each agent has:
  - Main Q-network (`q_network`)
  - Target Q-network (`target_network`)
  - Optimizer (Adam)
  - Replay buffer (`replay_memory`)
- **Key methods**:
  - `select_action()`: Epsilon-greedy action selection
  - `learn()`: Performs DQN update using agent's own buffer
  - `update_target_network()`: Copies main network to target

**`src/utils/replay_memory.py`** (Shared Replay Buffer)
- `ReplayMemory` class: Stores dict-based transitions
- Stores: `(state_dict, action_dict, next_state_dict, reward, done)`
- Each agent extracts its own data from the dicts during sampling

**`src/utils/qvalue_network.py`** (Q-Network Architecture)
- `QValueNetwork` class: Feedforward neural network
- Architecture: `input_dim (6) → hidden_dim (64) → hidden_dim (64) → num_actions (5)`
- Used by all IQL agents

#### Training Script

**`scripts/train_iql.py`**
- Multi-seed training (seeds: 2025-2029)
- Aggregates results across seeds
- Saves results to JSON files
- TensorBoard logging support

### Learning Process
1. Each agent selects action independently using epsilon-greedy
2. Environment returns joint reward and next observations
3. Each agent stores transition in its own replay buffer
4. Each agent learns independently from its own buffer
5. Target networks updated periodically for stability

---

## 3. Parameter-Shared DQN (PS-DQN)

### Algorithm Overview
**Parameter-Shared DQN (PS-DQN)** extends IQL by sharing a single Q-network across all agents. This approach:
- Uses one shared Q-network for all agents
- Enables parameter sharing, improving sample efficiency
- Agents learn from all agent experiences through the shared network
- More sample-efficient than IQL due to shared learning

### Key Characteristics
- **Shared parameters**: Single Q-network used by all agents
- **Shared optimizer**: One optimizer updates the shared network
- **Shared replay buffer**: Stores joint transitions (dict-based)
- **Sample efficiency**: Each timestep contributes n_agents samples to learning

### Implementation Structure

#### Core Files

**`src/algos/ps_dqn.py`** (Main Algorithm Coordinator)
- `PS_DQN` class: Manages shared learning components
- Coordinates training, evaluation, and logging
- Handles epsilon schedule and action selection
- **Key methods**:
  - `select_actions()`: Uses shared network for all agents
  - `store_transitions()`: Stores joint transitions
  - `train()`: Training loop with shared network updates

**`src/algos/ps_dqn_agent.py`** (Shared Learning Component)
- `PS_DQNAgent` class: Contains shared learning components
- Single shared Q-network (main and target)
- Single shared optimizer
- Single shared replay buffer
- **Key methods**:
  - `select_actions()`: Action selection using shared network
  - `store_transitions()`: Stores dict-based transitions
  - `train_step()`: Extracts individual agent data from dicts and combines them
  - `update_target_network()`: Updates shared target network

**`src/utils/replay_memory.py`** (Shared Replay Buffer)
- Same `ReplayMemory` class as IQL
- Stores joint transitions as dicts
- During sampling, individual agent transitions are extracted and combined

**`src/utils/qvalue_network.py`** (Shared Q-Network)
- Same `QValueNetwork` architecture
- Shared across all agents

#### Training Script

**`scripts/train_ps_dqn.py`**
- Same structure as `train_iql.py`
- Multi-seed training with aggregation
- Results saved for comparison with IQL

### Learning Process
1. All agents select actions using the shared Q-network
2. Environment returns joint reward and next observations
3. Joint transition stored in shared replay buffer (one per timestep)
4. During training:
   - Sample batch of joint transitions
   - Extract individual agent data from dicts
   - Combine all agent transitions (batch_size × n_agents samples)
   - Update shared network using combined batch
5. Target network updated periodically

### Key Difference from IQL
- **IQL**: Each agent learns from its own experiences only
- **PS-DQN**: All agents learn from all agent experiences through parameter sharing
- **Benefit**: PS-DQN can be more sample-efficient, especially when agents are homogeneous

---

## 4. Experimental Results

### Experimental Setup
- **Environment**: 5×5 Meeting Gridworld with 2 agents
- **Training**: 1000 episodes per seed
- **Seeds**: 5 seeds (2025, 2026, 2027, 2028, 2029) for statistical significance
- **Evaluation**: Final metrics computed over last 100 episodes
- **Hyperparameters**: Same for both algorithms (learning rate: 1e-3, batch size: 32, gamma: 0.99)

### IQL Results

```
Aggregated Results (across all seeds):
  Number of seeds: 5
  Seeds used: [2025, 2026, 2027, 2028, 2029]

  Final Metrics (last 100 episodes, mean ± std):
    Success Rate: 62.80% ± 10.40%
    Episode Length: 29.8 ± 4.9
    Return: 5.99 ± 1.09
```

#### Analysis of IQL Results

**Success Rate: 62.80% ± 10.40%**
- **Performance**: Strong performance with 62.80% success rate
- **Consistency**: Moderate variance (10.40% std) indicates some variability across seeds, but generally stable
- **Interpretation**: IQL successfully learns to coordinate in most cases, with agents independently learning to reach the goal together

**Episode Length: 29.8 ± 4.9**
- **Efficiency**: Episodes complete in ~30 steps on average, well below the 50-step limit
- **Variance**: Moderate variance (4.9 steps) suggests some episodes are easier/harder
- **Interpretation**: Agents learn efficient paths, with successful episodes ending quickly

**Return: 5.99 ± 1.09**
- **Reward**: Positive average return indicates successful episodes outweigh step penalties
- **Calculation**: With 62.80% success rate, successful episodes give +10 reward, failed episodes give ~-0.5 (50 steps × -0.01)
- **Interpretation**: The algorithm is learning a viable policy that achieves the goal frequently

**Overall Assessment**: IQL demonstrates strong performance, successfully learning coordination despite treating other agents as part of the environment. The moderate variance suggests the algorithm is robust but may benefit from more training or hyperparameter tuning.

---

### PS-DQN Results

```
Aggregated Results (across all seeds):
  Number of seeds: 5
  Seeds used: [2025, 2026, 2027, 2028, 2029]

  Final Metrics (last 100 episodes, mean ± std):
    Success Rate: 6.20% ± 2.04%
    Episode Length: 48.4 ± 0.7
    Return: 0.14 ± 0.21
```

#### Analysis of PS-DQN Results

**Success Rate: 6.20% ± 2.04%**
- **Performance**: Poor performance with only 6.20% success rate
- **Consistency**: Low variance (2.04% std) indicates consistently poor performance across seeds
- **Interpretation**: PS-DQN struggles to learn effective coordination, despite parameter sharing

**Episode Length: 48.4 ± 0.7**
- **Efficiency**: Episodes nearly always hit the 50-step timeout (48.4 average)
- **Variance**: Very low variance (0.7 steps) confirms most episodes time out
- **Interpretation**: Agents are not learning to reach the goal efficiently, suggesting the shared network may be struggling with the task

**Return: 0.14 ± 0.21**
- **Reward**: Near-zero return indicates minimal learning progress
- **Calculation**: With only 6.20% success, most episodes time out with ~-0.5 return (50 steps × -0.01)
- **Interpretation**: The algorithm is barely learning, with returns close to random policy performance

**Overall Assessment**: PS-DQN performs significantly worse than IQL. This surprising result suggests that parameter sharing may not be beneficial for this specific task, or there may be implementation issues preventing effective learning.

---

### Comparative Analysis

| Metric | IQL | PS-DQN | Winner |
|--------|-----|--------|--------|
| **Success Rate** | 62.80% ± 10.40% | 6.20% ± 2.04% | **IQL** (10× better) |
| **Episode Length** | 29.8 ± 4.9 | 48.4 ± 0.7 | **IQL** (more efficient) |
| **Return** | 5.99 ± 1.09 | 0.14 ± 0.21 | **IQL** (much higher) |
| **Variance** | Moderate | Low (but poor) | - |

#### Key Findings

1. **IQL Outperforms PS-DQN**: Contrary to expectations, independent learning performs significantly better than parameter sharing for this task.

2. **Possible Reasons for PS-DQN Failure**:
   - **Task-specific**: The meeting task may require agents to learn distinct strategies that conflict with parameter sharing
   - **Sample efficiency trade-off**: While parameter sharing can improve sample efficiency, it may constrain the policy space too much
   - **Implementation**: The shared network may need different hyperparameters or architecture

3. **IQL Strengths**:
   - Agents can learn independently without constraints
   - Each agent adapts to the other's behavior naturally
   - Simple approach works well for this coordination task

4. **PS-DQN Weaknesses**:
   - Shared parameters may prevent agents from learning task-specific behaviors
   - The shared network may struggle to represent diverse strategies needed for coordination
   - May require more sophisticated mixing or different architecture

---

## Summary

Both algorithms use the same environment and evaluation protocol, enabling fair comparison. **IQL demonstrates strong performance (62.80% success rate)**, successfully learning coordination despite treating other agents as part of the environment. **PS-DQN performs poorly (6.20% success rate)**, suggesting that parameter sharing may not be beneficial for this specific coordination task, or requires different implementation strategies.

The results highlight that **algorithm choice matters significantly** for multi-agent tasks, and that seemingly more advanced approaches (parameter sharing) do not always outperform simpler baselines (independent learning). The modular implementation allows easy comparison and extension to other MARL algorithms like QMIX for further investigation.

