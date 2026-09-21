"""
Training script for VDN on Meeting Gridworld.

Matches the structure and protocol used by existing training scripts (QMIX/PS-DQN).
"""

import os
import numpy as np
import torch
from typing import Dict, List, Any

from marl_meeting_task.src.env.meeting_gridworld import MeetingGridworldEnv
from marl_meeting_task.src.algos.vdn import VDN
from marl_meeting_task.src.utils.logger import Logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_training(seed, max_episodes, log_dir, verbose):
    set_seed(seed)

    hyperparameters: Dict[str, Any] = {
        'max_episodes': max_episodes,
        'learning_rate': 3e-4,
        'observation_dim': 4,
        'state_dim': 6,
        'network_size': 64,
        'num_actions': 5,
        'batch_size': 32,
        'memory_capacity': 10000,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.05,
        'epsilon_decay_steps': 75000,
        'target_update_freq': 50,
        'max_steps': 50,
        'min_buffer_size': 3000,
        'train_freq': 1,
        'eval_episodes': 200,
    }

    env = MeetingGridworldEnv(grid_size=5, n_agents=2, max_steps=hyperparameters['max_steps'])

    vdn = VDN(
        n_agents=2,
        input_dim=hyperparameters['observation_dim'],
        state_dim=hyperparameters['state_dim'],
        num_actions=hyperparameters['num_actions'],
        hidden_dim=hyperparameters['network_size'],
        learning_rate=hyperparameters['learning_rate'],
        memory_capacity=hyperparameters['memory_capacity'],
        gamma=hyperparameters['gamma'],
        epsilon_start=hyperparameters['epsilon_start'],
        epsilon_end=hyperparameters['epsilon_end'],
        epsilon_decay_steps=hyperparameters['epsilon_decay_steps'],
        batch_size=hyperparameters['batch_size'],
        target_update_freq=hyperparameters['target_update_freq'],
    )

    if log_dir is None:
        log_dir = f"runs/vdn_seed_{seed}"
    else:
        log_dir = f"{log_dir}/seed_{seed}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training VDN with seed: {seed}")
        print(f"{'='*60}\n")

    stats = vdn.train(
        env=env,
        max_episodes=max_episodes,
        max_steps=hyperparameters['max_steps'],
        train_freq=hyperparameters['train_freq'],
        min_buffer_size=hyperparameters['min_buffer_size'],
        verbose=verbose,
        log_dir=log_dir,
        eval_episodes=hyperparameters['eval_episodes'],
        env_seed=seed,
    )

    stats['seed'] = seed
    stats['hyperparameters'] = hyperparameters
    return stats


def aggregate_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n_seeds = len(all_results)
    all_rewards = [r['episode_rewards'] for r in all_results]
    all_lengths = [r['episode_lengths'] for r in all_results]
    all_successes = [r['episode_successes'] for r in all_results]

    max_episodes = max(len(r) for r in all_rewards)

    padded_rewards = []
    padded_lengths = []
    padded_successes = []
    for rewards, lengths, successes in zip(all_rewards, all_lengths, all_successes):
        padded_rewards.append(rewards + [np.nan] * (max_episodes - len(rewards)))
        padded_lengths.append(lengths + [np.nan] * (max_episodes - len(lengths)))
        padded_successes.append(successes + [np.nan] * (max_episodes - len(successes)))

    rewards_array = np.array(padded_rewards)
    lengths_array = np.array(padded_lengths)
    successes_array = np.array(padded_successes)

    aggregated = {
        'n_seeds': n_seeds,
        'seeds': [r['seed'] for r in all_results],
        'episode_rewards': {
            'mean': np.nanmean(rewards_array, axis=0).tolist(),
            'std': np.nanstd(rewards_array, axis=0).tolist(),
            'min': np.nanmin(rewards_array, axis=0).tolist(),
            'max': np.nanmax(rewards_array, axis=0).tolist(),
        },
        'episode_lengths': {
            'mean': np.nanmean(lengths_array, axis=0).tolist(),
            'std': np.nanstd(lengths_array, axis=0).tolist(),
            'min': np.nanmin(lengths_array, axis=0).tolist(),
            'max': np.nanmax(lengths_array, axis=0).tolist(),
        },
        'episode_successes': {
            'mean': np.nanmean(successes_array, axis=0).tolist(),
            'std': np.nanstd(successes_array, axis=0).tolist(),
        },
        'final_metrics': {
            'final_success_rate_mean': float(np.nanmean([np.mean(s[-100:]) for s in all_successes])),
            'final_success_rate_std': float(np.nanstd([np.mean(s[-100:]) for s in all_successes])),
            'final_episode_length_mean': float(np.nanmean([np.mean(l[-100:]) for l in all_lengths])),
            'final_episode_length_std': float(np.nanstd([np.mean(l[-100:]) for l in all_lengths])),
            'final_return_mean': float(np.nanmean([np.mean(r[-100:]) for r in all_rewards])),
            'final_return_std': float(np.nanstd([np.mean(r[-100:]) for r in all_rewards])),
        },
    }

    return aggregated


def main():
    SEEDS = [2025, 2026, 2027, 2028, 2029]
    MAX_EPISODES = 1000
    BASE_LOG_DIR = "runs/vdn_multi_seed"
    OUTPUT_DIR = "../results"
    VERBOSE = True

    # Initialize logger for training script
    logger = Logger(verbose=VERBOSE, log_dir=None)

    logger.summary(
        title="VDN Multi-Seed Training",
        items={
            "Seeds": SEEDS,
            "Max episodes per seed": MAX_EPISODES,
            "TensorBoard logs": BASE_LOG_DIR,
            "Results directory": OUTPUT_DIR,
        }
    )

    # Create a new run directory under results/vdn/{counter}
    run_dir = logger.create_run_dir('vdn', base_dir=OUTPUT_DIR)

    all_results = []
    for i, seed in enumerate(SEEDS, 1):
        logger.info(f"\n[{i}/{len(SEEDS)}] Starting VDN training with seed {seed}...")
        try:
            stats = run_training(seed=seed, max_episodes=MAX_EPISODES, log_dir=BASE_LOG_DIR, verbose=VERBOSE)
            all_results.append(stats)

            final_success_rate = np.mean(stats['episode_successes'][-100:])
            final_avg_length = np.mean(stats['episode_lengths'][-100:])
            final_avg_return = np.mean(stats['episode_rewards'][-100:])

            logger.info(f"\n[Seed {seed} Summary]")
            logger.info(f"  Final Success Rate (last 100): {final_success_rate:.2%}")
            logger.info(f"  Final Avg Length (last 100): {final_avg_length:.1f}")
            logger.info(f"  Final Avg Return (last 100): {final_avg_return:.2f}")
            logger.info(f"  Total Steps: {stats['total_steps']}")

            # Save per-seed JSON into the run directory
            results_data = {
                'seed': stats['seed'],
                'total_steps': stats['total_steps'],
                # 'episode_rewards': stats['episode_rewards'],
                # 'episode_lengths': stats['episode_lengths'],
                'episode_successes': stats['episode_successes'],
                'episode_losses': stats['episode_losses'],
                'final_eval_metrics': stats['final_eval_metrics'],
            }
            logger.save_seed_result_in_run(run_dir, seed=seed, hyperparameters=stats['hyperparameters'], results=results_data)

        except Exception as e:
            logger.info(f"\n[ERROR] VDN training failed for seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) == 0:
        logger.info("\n[ERROR] No successful VDN training runs!")
        return

    aggregated = aggregate_results(all_results)

    aggregated_save = {
        'hyperparameters': all_results[0]['hyperparameters'] if all_results else {},
        'aggregated': aggregated,
        'per_seed': all_results,
    }
    logger.save_aggregated_run(run_dir, aggregated_save)

    logger.aggregated_results(
        n_seeds=aggregated['n_seeds'],
        seeds=aggregated['seeds'],
        success_rate_mean=aggregated['final_metrics']['final_success_rate_mean'],
        success_rate_std=aggregated['final_metrics']['final_success_rate_std'],
        episode_length_mean=aggregated['final_metrics']['final_episode_length_mean'],
        episode_length_std=aggregated['final_metrics']['final_episode_length_std'],
        return_mean=aggregated['final_metrics']['final_return_mean'],
        return_std=aggregated['final_metrics']['final_return_std'],
    )

    logger.training_complete(BASE_LOG_DIR)


if __name__ == "__main__":
    main()

