import os
import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from learning.trainer import Trainer

def plot_metric_compare(values, param_name, syn_index, metric_name, ap_indices=(0,1)):
    """Plot multiple AP indices of the same metric on a single axes with legend."""
    plt.figure()
    if metric_name == "bias_index" or metric_name == "g_bias":
        data = values[param_name][syn_index][metric_name]
        plt.plot(data, label=f"AP None")
        plt.title(f"{param_name} | Synapse {syn_index} | {metric_name}")
    else:
        for ap in ap_indices:
            data = values[param_name][syn_index][metric_name][ap]
            plt.plot(data, label=f"AP {ap}")
        plt.title(f"{param_name} | Synapse {syn_index} | {metric_name} (compare APs)")
    plt.xlabel("Step")
    plt.ylabel(metric_name)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_metric(values, ap_indices=(0,1)):
    """Plot multiple AP indices of the same metric on a single axes with legend."""
    plt.figure()
    for ap in ap_indices:
        data = values[ap]
        plt.plot(data, label=f"AP {ap}")
        plt.title(f"Loss (compare APs)")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

hyperparams = {
    "discount_factor": 0.99,
    "batch_size": 64,
    "max_episodes": 3000,
    "max_steps": 200,
    "epsilon_max": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.000005,
    "memory_capacity": 1000000,
    "train": True
}


def main():
    if hyperparams["train"]:
        trainer = Trainer(hyperparams)
        trainer.train()
        tracked = trainer.agent.weight_controller.track_values
        loss = trainer.agent.loss_history
        with open("tracked.pkl", "wb") as f:
            pickle.dump(tracked, f)
        with open("loss.pkl", "wb") as f:
            pickle.dump(loss, f)
    else:
        with open("tracked.pkl", "rb") as f:
            tracked = pickle.load(f)
        with open("loss.pkl", "rb") as f:
            loss = pickle.load(f)
        # Plot both AP indices on the same plot for easy comparison
        plot_metric_compare(tracked, "FC.0.weight", (1,1), "weight", ap_indices=(0,1))
        plot_metric(loss, ap_indices=(0,1))
        # print(track_values["FC.0.weight"][(0, 0)]["weight"])
        # print(track_values["FC.0.weight"][(0, 0)]["g_ap"])
        # print(track_values["FC.0.weight"][(0, 0)]["g_bias"])
        # print(track_values["FC.0.weight"][(0, 0)]["x_index"])
        # print(track_values["FC.0.weight"][(0, 0)]["bias_index"])


if __name__ == "__main__":
    main()