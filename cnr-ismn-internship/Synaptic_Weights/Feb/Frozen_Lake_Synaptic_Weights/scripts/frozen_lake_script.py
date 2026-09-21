import os
import sys
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.frozen_lake_trainer import Trainer


hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 64,
    "max_episodes": 1000,
    "max_steps": 100,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00005,
    "memory_capacity": 10000,
}
trainer = Trainer(hyperparams)
rewards_fl = trainer.train()



# Plot 1: FL rewards
plt.figure()
plt.plot(rewards_fl)
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title(f"FL Rewards per Episode (max={max(rewards_fl):.2f})")
plt.grid(True)
plt.show()

