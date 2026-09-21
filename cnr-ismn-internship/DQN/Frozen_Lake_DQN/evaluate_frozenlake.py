import os
import csv
import numpy as np
import torch
import gymnasium as gym

from agent import DQNAgent
from config import device, seed


# ==============================
# Sigma sweep configuration
# ==============================

SIGMA_LIST = [
    1.7e-13,
    1.7e-12,
    1.7e-11,
    1.7e-10,
    1.7e-9,
]

RESULTS_DIR = "frozenlake_results"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")

EVAL_EPISODES = 200


# ==============================
# Environment configuration
# (MUST match training)
# ==============================

MAP_SIZE = 4
NUM_STATES = MAP_SIZE * MAP_SIZE
MAX_STEPS = 200


# ==============================
# Helpers
# ==============================

def sigma_to_str(sigma):
    return f"{sigma:.1e}".replace("+", "")


def make_env():
    env = gym.make(
        "FrozenLake-v1",
        map_name=f"{MAP_SIZE}x{MAP_SIZE}",
        is_slippery=False,
        max_episode_steps=MAX_STEPS,
        render_mode=None
    )
    return env


def state_preprocess(state):
    onehot = torch.zeros(NUM_STATES, dtype=torch.float32, device=device)
    onehot[state] = 1.0
    return onehot


# ==============================
# Evaluation logic
# ==============================


def evaluate_model(agent, env, episodes=200):
    rewards = []
    steps_list = []

    for _ in range(episodes):
        state, _ = env.reset(seed=seed)
        state = state_preprocess(state)

        done = False
        trunc = False
        ep_reward = 0
        steps = 0

        while not done and not trunc:
            action = agent.select_action(state, eval_mode=True)
            next_state, reward, done, trunc, _ = env.step(action)

            state = state_preprocess(next_state)
            ep_reward += reward
            steps += 1

        rewards.append(ep_reward)
        steps_list.append(steps)

    rewards = np.array(rewards)

    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(rewards > 0)),
        "avg_steps": float(np.mean(steps_list)),
        "std_reward": float(np.std(rewards)),
    }


# ==============================
# Main evaluation loop
# ==============================

if __name__ == "__main__":

    results = []

    for sigma in SIGMA_LIST:
        sigma_str = sigma_to_str(sigma)
        model_path = os.path.join(MODELS_DIR, f"sigma_{sigma_str}.pth")

        print("\n==============================")
        print(f"Evaluating sigma = {sigma}")
        print("==============================")

        env = make_env()

        agent = DQNAgent(
            env=env,
            epsilon_max=0.0,
            epsilon_min=0.0,
            epsilon_decay=1.0,
            clip_grad_norm=0.0,
            learning_rate=0.0,
            memory_capacity=1,
            discount=0.99,
            sigma=sigma,
        )

        agent.main_network.load_state_dict(
            torch.load(model_path, map_location=device)
        )
        agent.main_network.eval()

        metrics = evaluate_model(agent, env, episodes=EVAL_EPISODES)
        metrics["sigma"] = sigma

        results.append(metrics)

        print(metrics)

    # ==============================
    # Save evaluation summary
    # ==============================

    output_csv = os.path.join(RESULTS_DIR, "evaluation.csv")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sigma", "mean_reward", "success_rate", "avg_steps", "std_reward"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nEvaluation results saved to {output_csv}")
