"""
Logger class for handling all logging responsibilities in the MARL project.
Handles console logging, TensorBoard logging, and saving JSON result files.
"""

import os
import json
from typing import Optional, Dict, Any


class Logger:
    """
    Centralized logger for console and result saving.

    This class handles all logging responsibilities that remain enabled:
    - Console output (info, debug, progress, evaluation results)
    - Saving per-seed JSON result files with hyperparameters and metrics
    - Configuration (verbose mode)

    TensorBoard support has been removed per project preference.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize the logger.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print to console (default: True)
        log_dir : Optional[str]
            Ignored (kept for API compatibility). TensorBoard logging disabled.
        """
        self.verbose = verbose
        # log_dir parameter kept for backward compatibility but not used
        self.log_dir = None

    def info(self, message: str) -> None:
        """Log an info message to console."""
        if self.verbose:
            print(message)
    
    def debug(self, message: str) -> None:
        """Log a debug message to console."""
        if self.verbose:
            print(f"[DEBUG] {message}")
    
    def separator(self, char: str = "=", length: int = 60) -> None:
        """Print a separator line."""
        if self.verbose:
            print(char * length)
    
    def blank_line(self) -> None:
        """Print a blank line."""
        if self.verbose:
            print()
    
    def progress(
        self,
        episode: int,
        max_episodes: int,
        avg_reward: float,
        avg_length: float,
        success_rate: float,
        epsilon: float,
        total_steps: int,
    ) -> None:
        """
        Log training progress.
        
        Parameters:
        -----------
        episode : int
            Current episode number
        max_episodes : int
            Maximum number of episodes
        avg_reward : float
            Average reward over last 100 episodes
        avg_length : float
            Average episode length over last 100 episodes
        success_rate : float
            Success rate over last 100 episodes
        epsilon : float
            Current epsilon value
        total_steps : int
            Total training steps
        """
        if self.verbose:
            print(f"Episode {episode}/{max_episodes} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Success Rate: {success_rate:.2%} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Total Steps: {total_steps}")
    
    def evaluation(
        self,
        episode: Optional[int],
        success_rate: float,
        avg_episode_length: float,
        avg_return: float,
        is_final: bool = False,
    ) -> None:
        """
        Log evaluation results.
        
        Parameters:
        -----------
        episode : Optional[int]
            Episode number (None for final evaluation)
        success_rate : float
            Success rate
        avg_episode_length : float
            Average episode length
        avg_return : float
            Average return
        is_final : bool
            Whether this is the final evaluation (default: False)
        """
        if not self.verbose:
            return
        
        if is_final:
            self.separator()
            self.info("Running final evaluation...")
            self.separator()
            self.blank_line()
            self.info(f"[Final Evaluation] "
                      f"Success Rate: {success_rate:.2%} | "
                      f"Avg Length: {avg_episode_length:.1f} | "
                      f"Avg Return: {avg_return:.2f}")
        else:
            self.blank_line()
            self.info(f"[Evaluation at Episode {episode}] "
                      f"Success Rate: {success_rate:.2%} | "
                      f"Avg Length: {avg_episode_length:.1f} | "
                      f"Avg Return: {avg_return:.2f}")
            self.blank_line()
    
    def summary(
        self,
        title: str,
        items: Dict[str, Any],
    ) -> None:
        """
        Print a formatted summary with title and key-value pairs.
        
        Parameters:
        -----------
        title : str
            Title of the summary
        items : Dict[str, Any]
            Dictionary of key-value pairs to print
        """
        if not self.verbose:
            return
        
        self.blank_line()
        self.separator()
        self.info(title)
        self.separator()
        self.blank_line()
        
        for key, value in items.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.2f}")
            elif isinstance(value, (list, tuple)):
                self.info(f"  {key}: {value}")
            else:
                self.info(f"  {key}: {value}")
    
    def aggregated_results(
        self,
        n_seeds: int,
        seeds: list,
        success_rate_mean: float,
        success_rate_std: float,
        episode_length_mean: float,
        episode_length_std: float,
        return_mean: float,
        return_std: float,
    ) -> None:
        """
        Print aggregated results across multiple seeds.
        
        Parameters:
        -----------
        n_seeds : int
            Number of seeds
        seeds : list
            List of seed values
        success_rate_mean : float
            Mean success rate
        success_rate_std : float
            Std success rate
        episode_length_mean : float
            Mean episode length
        episode_length_std : float
            Std episode length
        return_mean : float
            Mean return
        return_std : float
            Std return
        """
        if not self.verbose:
            return
        
        self.blank_line()
        self.separator()
        self.info("Aggregating results across seeds...")
        self.separator()
        self.blank_line()
        
        self.info("Aggregated Results (across all seeds):")
        self.info(f"  Number of seeds: {n_seeds}")
        self.info(f"  Seeds used: {seeds}")
        self.blank_line()
        self.info("  Final Metrics (last 100 episodes, mean ± std):")
        self.info(f"    Success Rate: {success_rate_mean:.2%} ± {success_rate_std:.2%}")
        self.info(f"    Episode Length: {episode_length_mean:.1f} ± {episode_length_std:.1f}")
        self.info(f"    Return: {return_mean:.2f} ± {return_std:.2f}")

    def training_complete(self, log_dir: str) -> None:
        """
        Print training completion message.

        Parameters:
        -----------
        log_dir : str
            Directory where TensorBoard logs are saved
        """
        if not self.verbose:
            return

        self.blank_line()
        self.separator()
        self.info("Training completed!")
        self.separator()
        self.blank_line()

    # ------------------------------------------------------------------
    # JSON result saving utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _get_next_counter(algorithm_dir: str, seed: int) -> int:
        """
        Get the next counter value for a seed folder.
        
        The counter is inferred from existing files named 'result_{i}.json'.
        """
        seed_dir = os.path.join(algorithm_dir, str(seed))
        if not os.path.exists(seed_dir):
            return 0
        
        existing_files = [
            f for f in os.listdir(seed_dir)
            if f.startswith("result_") and f.endswith(".json")
        ]
        if not existing_files:
            return 0
        
        counters = []
        for filename in existing_files:
            try:
                counter = int(filename.replace("result_", "").replace(".json", ""))
                counters.append(counter)
            except ValueError:
                continue
        
        return max(counters) + 1 if counters else 0
    
    def save_seed_result(
        self,
        algorithm: str,
        seed: int,
        hyperparameters: Dict[str, Any],
        results: Dict[str, Any],
        base_dir: str = "results",
    ) -> str:
        """
        Save results for a single seed with hyperparameters to a JSON file.
        
        Directory structure:
            {base_dir}/{algorithm}/{seed}/result_{counter}.json
        
        Parameters:
        -----------
        algorithm : str
            Algorithm name (e.g., 'qmix', 'iql', 'ps_dqn')
        seed : int
            Seed number
        hyperparameters : Dict[str, Any]
            Dictionary of hyperparameters (taken from the actual config used)
        results : Dict[str, Any]
            Dictionary of results/metrics
        base_dir : str
            Base directory for results (default: "results")
        
        Returns:
        --------
        str
            Path to the saved JSON file.
        """
        # Create directory structure: base_dir/{algorithm}/{seed}/
        algorithm_dir = os.path.join(base_dir, algorithm)
        seed_dir = os.path.join(algorithm_dir, str(seed))
        os.makedirs(seed_dir, exist_ok=True)
        
        # Get next counter for this seed
        counter = self._get_next_counter(algorithm_dir, seed)
        
        # Build file path
        result_file = os.path.join(seed_dir, f"result_{counter}.json")
        
        # Prepare data
        data = {
            "hyperparameters": hyperparameters,
            "results": results,
        }
        
        # Save to JSON
        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Optional console message
        if self.verbose:
            self.info(f"Results saved to: {result_file}")
        
        return result_file
    
    def create_run_dir(self, algorithm: str, base_dir: str = "results") -> str:
        """
        Create a new run directory under base_dir/algorithm/{counter} where counter is
        an increasing integer inferred from existing run folders.

        Returns the path to the created run directory.
        """
        algorithm_dir = os.path.join(base_dir, algorithm)
        os.makedirs(algorithm_dir, exist_ok=True)

        # Find existing numeric subfolders and pick next counter
        existing = [d for d in os.listdir(algorithm_dir) if os.path.isdir(os.path.join(algorithm_dir, d))]
        counters = []
        for name in existing:
            try:
                counters.append(int(name))
            except ValueError:
                continue
        next_counter = max(counters) + 1 if counters else 0

        run_dir = os.path.join(algorithm_dir, str(next_counter))
        os.makedirs(run_dir, exist_ok=True)
        if self.verbose:
            self.info(f"Created run directory: {run_dir}")
        return run_dir

    def save_seed_result_in_run(
        self,
        run_dir: str,
        seed: int,
        hyperparameters: Dict[str, Any],
        results: Dict[str, Any],
    ) -> str:
        """
        Save per-seed JSON into the provided run directory using filename seed_{seed}.json.
        """
        result_file = os.path.join(run_dir, f"seed_{seed}.json")
        data = {
            "hyperparameters": hyperparameters,
            "results": results,
        }
        with open(result_file, "w") as f:
            json.dump(data, f, indent=2)
        if self.verbose:
            self.info(f"Seed results saved to: {result_file}")
        return result_file

    def save_aggregated_run(self, run_dir: str, aggregated: Dict[str, Any]) -> str:
        """
        Save aggregated results JSON into the run directory as aggregated.json.
        """
        agg_file = os.path.join(run_dir, "aggregated.json")
        with open(agg_file, "w") as f:
            json.dump(aggregated, f, indent=2)
        if self.verbose:
            self.info(f"Aggregated results saved to: {agg_file}")
        return agg_file

    def close(self) -> None:
        """Close resources (no-op: tensorboard disabled)."""
        return

    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - nothing to close."""
        self.close()

