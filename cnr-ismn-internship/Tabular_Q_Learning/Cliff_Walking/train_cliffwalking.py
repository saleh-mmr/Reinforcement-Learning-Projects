from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import utils.config as config
import matplotlib.pyplot as plt
from envs.cliff_walking_env import CliffWalkingEnv
from devices.magnetoresistance import MagnetoresistanceParams
from devices.multiweight_synapse import MultiWeightSynapse, MultiWeightSynapseSpec
from learning.cliffwalking_step import cliffwalking_learning_step


# ============================================================
# Training configuration
# ============================================================

@dataclass
class TrainSpec:
    episodes: int = 3000
    max_steps: int = 200

    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_episodes: int = 1200

    gamma: float = 0.99
    seed: int = config.seed
    log_every: int = 1000


# ============================================================
# Utilities
# ============================================================

def linear_epsilon(ep: int, spec: TrainSpec) -> float:
    if ep >= spec.epsilon_decay_episodes:
        return spec.epsilon_end
    t = ep / max(1, spec.epsilon_decay_episodes)
    return spec.epsilon_start + t * (spec.epsilon_end - spec.epsilon_start)


def choose_action_eps_greedy(q: np.ndarray, epsilon: float, rng) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    return int(np.argmax(q))


# ============================================================
# Memristor Q-network
# ============================================================

def build_q_network(
    n_states: int,
    n_actions: int,
    scaling_factor: float,
    mr_params: MagnetoresistanceParams,
    rng,
):
    syn_spec = MultiWeightSynapseSpec(
        n_plus=1,
        scaling_factor=scaling_factor,
    )

    return [
        [
            MultiWeightSynapse(syn_spec, mr_params, rng)
            for _ in range(n_actions)
        ]
        for _ in range(n_states)
    ]


def read_q(phi_s, synapses, ap_index=0):
    s_idx = int(np.argmax(phi_s))
    a = np.array(
        [
            synapses[s_idx][a].weight(ap_index)[0]
            for a in range(len(synapses[s_idx]))
        ],
        dtype=np.float32,
    )
    return a


# ============================================================
# Training loop
# ============================================================

def train(sigma_pulse_noise):
    spec = TrainSpec()
    rng = np.random.default_rng(spec.seed)

    env = CliffWalkingEnv(seed=spec.seed)

    # -------- device parameters (stable for discrete tasks) --------
    mr_params = MagnetoresistanceParams(
        a=1.566e-8,
        b=0.350e-8,
        c=0.1,
        g_threshold=0.8,
        sigma_pulse_noise=sigma_pulse_noise,
        min_pulse_index_for_log=1)

    synapses = build_q_network(
        env.n_states,
        env.n_actions,
        scaling_factor=3e8,
        mr_params=mr_params,
        rng=rng)

    # --------------------------------------------------------
    # Episodes
    # --------------------------------------------------------
    all_episode_rewards = []
    all_cond = []
    for ep in range(spec.episodes):
        epsilon = linear_epsilon(ep, spec)
        _, phi_s, _ = env.reset()

        episode_step_counter = 0.0
        last_c = 0
        for _ in range(spec.max_steps):
            q = read_q(phi_s, synapses, ap_index=0)
            action = choose_action_eps_greedy(q, epsilon, rng)

            _, phi_s2, reward, terminated, truncated, _ = env.step(action)

            c = cliffwalking_learning_step(
                phi_s=phi_s,
                action=action,
                reward=reward,
                phi_s_next=phi_s2,
                terminated=terminated,
                synapses=synapses,
                gamma=spec.gamma,
                ap_index=0,
            )
            episode_step_counter += 1
            last_c = c


            phi_s = phi_s2
            if terminated or truncated:
                break
        episode_reward = spec.max_steps - episode_step_counter
        all_episode_rewards.append(episode_reward)
        all_cond.append(last_c)
    mean = np.mean(all_cond)
    env.close()
    return all_episode_rewards , mean

def run_sigma_sweep():
    sigmas = [
        # 1.7e-12,
        # 1.7e-11,
        1.7e-10,
        7.7e-10,
        9.7e-9,
        # 1.7e-7,
    ]
    curves = {}
    conductance = {}

    for sigma in sigmas:
        print(f"Training with sigma = {sigma:.1e}")
        curves[sigma], conductance[sigma]  = train(sigma)
    return curves, conductance


# ============================================================
# Plotting utilities
# ============================================================

def moving_average_last_k(values, k=50):
    """Return an array where each element i is the mean of values[max(0, i-k+1):i+1].
    The returned array has the same length as `values`.
    """
    vals = np.asarray(values, dtype=np.float32)
    n = len(vals)
    if n == 0:
        return vals
    ma = np.empty(n, dtype=np.float32)
    cumsum = np.cumsum(vals)
    for i in range(n):
        start = max(0, i - k + 1)
        total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        ma[i] = total / (i - start + 1)
    return ma


def plot_curves(curves, conductance, window):
    """Plot the moving-average (last `window` episodes) of rewards for each sigma on the same figure."""
    plt.figure(figsize=(12, 8))
    for sigma, values in curves.items():
        ma = moving_average_last_k(values, window)
        # Use scientific format for small sigma values
        plt.plot(ma, label=f"Programming Noise: {(sigma/conductance[sigma])*100:.1f}", linewidth=3.5)

    plt.title("Cliff Walking Online Training", fontsize=28, pad=20)
    plt.xlabel("Episode", fontsize=24)
    plt.ylabel(f"Mean Reward ({window}-episode MA)", fontsize=24)
    plt.legend(fontsize=20, handlelength=1)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # run training for different sigma values
    curves, conductance = run_sigma_sweep()

    # plot a 50-episode moving average of the rewards for each sigma
    plot_curves(curves, conductance, window=50)
