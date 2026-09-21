from __future__ import annotations
import numpy as np
from learning.online_update import apply_online_update


def cliffwalking_learning_step(
    phi_s,
    action,
    reward,
    phi_s_next,
    terminated,
    synapses,
    gamma,
    ap_index=0,
):
    """
    One learning step for DISCRETE CliffWalking.

    Assumptions:
      - φ(s) is one-hot
      - Tabular Q-values stored in memristive synapses
    """

    # -------- current state index --------
    s_idx = int(np.argmax(phi_s))

    # -------- Q(s,a) --------
    q_sa, c = synapses[s_idx][action].weight(ap_index=ap_index)

    # -------- target --------
    if terminated:
        y = float(reward)
    else:
        s_next = int(np.argmax(phi_s_next))
        q_next_vals = [
            synapses[s_next][a].weight(ap_index=ap_index)[0]
            for a in range(len(synapses[s_next]))
        ]
        y = float(reward + gamma * np.max(q_next_vals))

    # -------- TD error --------
    delta = float(y - q_sa)

    # -------- update direction --------
    direction = 1 if delta > 0 else (-1 if delta < 0 else 0)

    # -------- apply paper update --------
    apply_online_update(
        synapse=synapses[s_idx][action],
        direction=direction,
        ap_index=ap_index,
    )

    return c
