from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from envs.frozen_lake_env import FrozenLakeEnv, FrozenLakeSpec
from devices.magnetoresistance import MagnetoresistanceParams
from devices.multiweight_synapse import MultiWeightSynapse, MultiWeightSynapseSpec
from learning.multitask_step import (TaskSpec, multitask_learning_step)

# ============================================================
# Training configuration
# ============================================================

@dataclass
class TrainSpec:
    episodes: int = 3000
    max_steps: int = 100

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 1200

    gamma: float = 0.99
    seed: int = 2026
    log_every: int = 100


# ============================================================
# Utilities
# ============================================================

def linear_epsilon(ep: int, spec: TrainSpec) -> float:
    if ep >= spec.epsilon_decay_episodes:
        return spec.epsilon_end
    t = ep / max(1, spec.epsilon_decay_episodes)
    return spec.epsilon_start + t * (spec.epsilon_end - spec.epsilon_start)


def moving_average(x, window: int):
    if len(x) < window:
        return np.array([])
    return np.convolve(x, np.ones(window) / window, mode="valid")


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


def read_q(phi_s: np.ndarray, synapses, ap_index: int) -> np.ndarray:
    s_idx = int(np.argmax(phi_s))
    return np.array(
        [synapses[s_idx][a].weight(ap_index)[0] for a in range(len(synapses[s_idx]))],
        dtype=np.float32,
    )


# ============================================================
# Training loop
# ============================================================

def train(sigma_pulse_noise):
    spec = TrainSpec()
    rng = np.random.default_rng(spec.seed)

    env = FrozenLakeEnv(
        FrozenLakeSpec(
            map_name="4x4",
            is_slippery=True,
            seed=spec.seed,
        )
    )

    mr_params = MagnetoresistanceParams(
        a=1.566e-8,
        b=0.350e-8,
        c=0.1,
        g_threshold=0.8,
        sigma_pulse_noise=sigma_pulse_noise,
        min_pulse_index_for_log=1,
    )

    synapses = build_q_network(
        env.n_states,
        env.n_actions,
        scaling_factor=7e7,
        mr_params=mr_params,
        rng=rng,
    )

    tasks = [TaskSpec(name="FL", ap_index=0)]

    rewards = []
    logging_conductance = {}


    for ep in range(spec.episodes):
        epsilon = linear_epsilon(ep, spec)
        _, phi_s, _ = env.reset()
        total_reward = 0.0
        conductance_in_episode = []

        for _ in range(spec.max_steps):
            q = read_q(phi_s, synapses, ap_index=0)
            action = choose_action_eps_greedy(q, epsilon, rng)

            _, phi_s2, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            multitask_learning_step(
                tasks=tasks,
                experiences={
                    "FL": (phi_s, action, reward, phi_s2, terminated)
                },
                synapses=synapses,
                gamma=spec.gamma)

            s_idx = int(np.argmax(phi_s))
            _, conductance = synapses[s_idx][action].weight(ap_index=0)
            conductance_in_episode.append(conductance)

            phi_s = phi_s2
            if terminated or truncated:
                break

        rewards.append(total_reward)
        logging_conductance.update({ep:conductance_in_episode})

    env.close()
    return rewards, logging_conductance


def run_sigma_sweep():
    sigmas = [
        1.7e-10,
        7.7e-10,
        9.7e-9,
    ]
    curves = {}
    conductances = {}

    for sigma in sigmas:
        print(f"Training with sigma = {sigma:.1e}")
        r, c = train(sigma)
        last_values = [val[1] for val in c.values() if len(val) > 0]
        mean_of_lasts = np.mean(last_values)
        conductances[sigma] = mean_of_lasts
        curves[sigma] = moving_average(r, window=50)
    return curves, conductances


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    curves, conductance = run_sigma_sweep()
    print(conductance)

    for sigma, values in curves.items():
        # 'linewidth' makes the actual plot lines thicker
        percentage = (sigma / conductance[sigma]) * 100
        if percentage < 1:
            plt.plot(values, label=f"Programming Noise: {percentage:.1f}%", linewidth=3.5)
        else:
            plt.plot(values, label=f"Programming Noise: {percentage:.0f}%", linewidth=3.5)

    plt.title("Frozen Lake Online Training", fontsize=28, pad=20)
    plt.xlabel("Episode", fontsize=24)
    plt.ylabel("Mean Reward (50-episode MA)", fontsize=24)

    # 'fontsize' handles the text, 'handlelength' makes the colored lines in legend longer
    plt.legend(fontsize=20, handlelength=1)

    # Make tick marks (0.8, 1.0) larger as well
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.grid(True)
    plt.show()