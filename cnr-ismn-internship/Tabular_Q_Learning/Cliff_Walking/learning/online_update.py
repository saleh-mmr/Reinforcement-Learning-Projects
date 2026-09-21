from __future__ import annotations
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def apply_online_update(synapse, direction, ap_index):
    """
    Apply ONE on-line update to ONE composite synapse.

    Parameters
    ----------
    synapse : MultiWeightSynapse
        The composite memristive synapse (one ANN weight).
    direction : int
        +1 -> weight should increase
        -1 -> weight should decrease
         0 -> no change
    ap_index : int
        Which '+' crosspoint corresponds to the current task (AP state).
        For FrozenLake-only: ap_index = 0.

    Paper rules:
      - If weight needs to be increased:
            increase x_n (corresponding '+' crosspoint)
      - If weight needs to be decreased:
            increase x_B (bias crosspoint)
      - Redraw noise when x changes
    """
    if direction > 0:
        # Increase corresponding '+' crosspoint
        synapse.increase_plus(ap_index)

    elif direction < 0:
        # Decrease weight by increasing bias
        synapse.increase_bias()


def batch_online_update(synapses, directions, ap_index):
    """
    Typical use case:
        - Updating all weights for one action
        - Or all synapses in a layer at once
    """
    assert len(synapses) == len(directions)

    for syn, d in zip(synapses, directions):
        apply_online_update(
            synapse=syn,
            direction=int(d),
            ap_index=ap_index,
        )
