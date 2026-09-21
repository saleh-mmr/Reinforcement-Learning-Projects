from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from learning.online_update import apply_online_update


@dataclass(frozen=True)
class TaskSpec:
    name: str
    ap_index: int  # which '+' crosspoint is AP for this task


def read_q_from_memristors_for_task(phi_s, synapses, ap_index):
    """
    Q(s,a) readout for a specific task selection (ap_index).
    For one-hot phi, picks the active state row.
    """
    s_idx = int(np.argmax(phi_s))
    n_actions = len(synapses[s_idx])
    q = np.zeros(n_actions, dtype=np.float32)
    for a in range(n_actions):
        weight, _ = synapses[s_idx][a].weight(ap_index=ap_index)
        q[a] = float(weight)
    return q


def multitask_learning_step(tasks, experiences, synapses, gamma):
    """
    Performs ONE paper-style learning step across ALL tasks before returning.

    For each task:
      1) compute delta using memristor-read Q
      2) direction = sign(delta)
      3) update only the active weight (state, action) using paper rule:
            + => increment x_task
            - => increment x_B :contentReference[oaicite:4]{index=4}
    """
    for task in tasks:
        phi_s, action, reward, phi_s_next, terminated = experiences[task.name]

        q = read_q_from_memristors_for_task(phi_s, synapses, ap_index=task.ap_index)
        q_sa = float(q[action])

        q_next = read_q_from_memristors_for_task(phi_s_next, synapses, ap_index=task.ap_index)
        y = float(reward) if terminated else float(reward + gamma * np.max(q_next))
        delta = float(y - q_sa)

        direction = 1 if delta > 0 else (-1 if delta < 0 else 0)

        active_state = int(np.argmax(phi_s))
        apply_online_update(
            synapse=synapses[active_state][action],
            direction=direction,
            ap_index=task.ap_index)
