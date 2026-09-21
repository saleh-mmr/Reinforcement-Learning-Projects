from __future__ import annotations

from dataclasses import dataclass
import numpy as np
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
    seed: int = 0
    log_every: int = 100


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


def read_q(phi_s, synapses, ap_index):
    s_idx = int(np.argmax(phi_s))
    return np.array(
        [synapses[s_idx][a].weight(ap_index)[0] for a in range(len(synapses[s_idx]))],
        dtype=np.float32,
    )


# ============================================================
# Training loop
# ============================================================

def train():
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
        g_threshold=0.5,
        sigma_pulse_noise=1.7e-14,
        min_pulse_index_for_log=1,
    )

    synapses = build_q_network(
        env.n_states,
        env.n_actions,
        scaling_factor=9e7,
        mr_params=mr_params,
        rng=rng,
    )

    tasks = [TaskSpec(name="FL", ap_index=0)]

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    logging_conductance = {}

    # --------------------------------------------------------
    # Episodes
    # --------------------------------------------------------

    for ep in range(spec.episodes):
        epsilon = linear_epsilon(ep, spec)
        _, phi_s, _ = env.reset()

        episode_reward = 0.0
        conductance_in_episode = []

        for _ in range(spec.max_steps):
            q = read_q(phi_s, synapses, ap_index=0)
            action = choose_action_eps_greedy(q, epsilon, rng)

            _, phi_s2, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            multitask_learning_step(
                tasks=tasks,
                experiences={
                    "FL": (phi_s, action, reward, phi_s2, terminated)
                },
                synapses=synapses,
                gamma=spec.gamma,
            )

            # log active weight
            s_idx = int(np.argmax(phi_s))
            _, conductance = synapses[s_idx][action].weight(ap_index=0)
            conductance_in_episode.append(conductance)

            phi_s = phi_s2
            if terminated or truncated:
                break

        logging_conductance.update({ep:conductance_in_episode})

        if (ep + 1) % spec.log_every == 0:
            print(
                f"Episode {ep+1:5d} | "
                f"eps={epsilon:.3f} | "
                f"reward={episode_reward:.2f}"
            )

    env.close()
    return logging_conductance


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    all_conductance = train()
    last_values = [val[1] for val in all_conductance.values() if len(val) > 0]
    mean_of_lasts = np.mean(last_values)
    print(f"Mean of last values: {mean_of_lasts}")

