import os
import sys
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.trainer import Trainer


hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 64,
    "max_episodes": 10,
    "max_steps": 100,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.00005,
    "memory_capacity": 10000,
}
trainer = Trainer(hyperparams)
rewards_cp, rewards_mc = trainer.train()



# Plot 1: CP rewards
plt.figure()
plt.plot(rewards_cp)
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title(f"CP Rewards per Episode (max={max(rewards_cp):.2f})")
plt.grid(True)
plt.show()

# Plot 2: MC rewards
plt.figure()
plt.plot(rewards_mc)
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.title(f"MC Rewards per Episode (max={max(rewards_mc):.2f})")
plt.grid(True)
plt.show()
