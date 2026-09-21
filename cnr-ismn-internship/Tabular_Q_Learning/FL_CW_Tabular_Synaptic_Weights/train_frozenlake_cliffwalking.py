import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from devices.multiWeightSynapse import MultiWeightSynapseSpec
from devices.crosspointParams import CrossPointParams
from learning.memristiveQFunction import MemristiveQFunction
from envs.frozenLake import FrozenLakeEnv
from envs.cliffWalking import CliffWalkingEnv
from utils import config

def linear_epsilon(ep: int, epsilon_end, epsilon_start, epsilon_decay_episodes) -> float:
    if ep >= epsilon_decay_episodes:
        return epsilon_end
    t = ep / max(1, epsilon_decay_episodes)
    return epsilon_start + t * (epsilon_end - epsilon_start)


def choose_action_eps_greedy(q: np.ndarray, epsilon, rng) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    return int(np.argmax(q))


def build_q_function(n_states, n_actions, spec, params) -> MemristiveQFunction:
    q_function = MemristiveQFunction(state_size=n_states, action_size=n_actions, spec=spec, params=params)
    return q_function


def build_synapse_spec(n_problem, scaling_factor):
    return MultiWeightSynapseSpec(n_problem=n_problem, scaling_factor=scaling_factor)


def build_crosspoint_params(a, b, c, g_threshold, sigma_pulse_noise):
    return CrossPointParams(a=a, b=b, c=c, g_threshold=g_threshold, sigma_pulse_noise=sigma_pulse_noise)


def read_q_values(q_function: MemristiveQFunction, phi_s, ap_index) -> np.ndarray:
    q_values = np.array([])
    s_idx = int(np.argmax(phi_s))
    for action in range(len(q_function[s_idx])):
        synapse = q_function.get_synapse(phi_s, action)
        q_value = synapse.weight(ap_index)
        q_values = np.append(q_values, q_value)
    return q_values


def train():
    env_0 = FrozenLakeEnv()
    env_1 = CliffWalkingEnv()
    tasks = [
        env_0,
        env_1,
    ]
    n_states = env_0.n_states
    n_actions = env_0.n_actions
    n_problems = len(tasks)
    scaling_factor = 5e10
    a = 1.566e-8
    b = 0.350e-8
    c = 5e4
    g_threshold = 0.350e-8
    sigma_pulse_noise = 0.0
    rng = np.random.default_rng(seed=config.seed)

    # Build synapse spec and crosspoint params
    spec = build_synapse_spec(n_problem=n_problems, scaling_factor=scaling_factor)
    crosspoint_params = build_crosspoint_params(a, b, c, g_threshold, sigma_pulse_noise)

    # Build Q-function
    q_function = build_q_function(n_states, n_actions, spec, crosspoint_params)


    max_episode = 2
    max_step = 100
    for episode in range(max_episode):
        epsilon = linear_epsilon(episode, epsilon_start = 1.0, epsilon_end = 0.05, epsilon_decay_episodes = 50)
        episode_reward = 0.0

        for ap_index, env in enumerate(tasks):
            for _ in range(max_step):
                phi_s = env.reset()
                done = False
                trunc = False
                print(env)
                while not (done or trunc):
                    q_values = read_q_values(q_function, phi_s, ap_index=ap_index)  # Assuming ap_index=0 for simplicity
                    action = choose_action_eps_greedy(q_values, epsilon=epsilon, rng=rng)
                    _ , phi_next_state, reward, done , trunc, _ = env.step(action)
                    episode_reward += reward




train()