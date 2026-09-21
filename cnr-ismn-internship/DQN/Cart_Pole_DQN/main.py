from config import device
from trainer import Trainer

train_mode = True                                  # True = Train DQN model, False = Load and Test
render = False                                     # Enable/disable environment visualization
max_episodes = 5000 if train_mode else 10

# ---------------- Hyperparameters ----------------
HPARAMS = {
    "RL_load_pth": "weights/best_cartpole.pth",     # Model checkpoint path for testing
    "learning_rate": 3e-4,                          # (not used for magnitude in Manhattan mode)
    "discount_factor": 0.99,                        # Gamma (future reward discount)
    "batch_size": 32,                               # Number of experiences per learning step
    "max_episodes": max_episodes,                   # number of episode for training or testing
    "max_steps": 500,                               # Episode timeout
    "render": render,                               # Set True to visually inspect
    "epsilon_max": 1.0,                             # Initial exploration rate
    "epsilon_min": 0.05,                            # Minimum allowed epsilon
    "epsilon_decay": 0.00000003,                         # Exploration decay speed
    "memory_capacity": 10000,                       # Replay buffer size
    "render_fps": 30,                               # Visualization frame rate
    "weight_datafile_path": "conductance/datafile.csv",         # path to your CSV file
}


if __name__ == "__main__":
    trainer = Trainer(HPARAMS)
    if train_mode:
        print(f"Running on device: {device}")
        trainer.train()
    else:
        trainer.test(max_episodes)