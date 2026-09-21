# # Creating a test for the MemristiveQFunction class
#
# ___________________________imports___________________________
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from devices.multiWeightSynapse import MultiWeightSynapseSpec
from devices.crosspointParams import CrossPointParams
from learning.memristiveQFunction import MemristiveQFunction

current_state = [1, 0, 0]  # One-hot encoded state for state index 0
action = 0

# __________________________initialize params___________________________
# crosspoint parameters
a = 1.566e-8
b = 0.350e-8
c = 5e4
g_threshold = 0.350e-8
sigma_pulse_noise = 0.0
params = CrossPointParams(a=a, b=b, c=c, g_threshold=g_threshold, sigma_pulse_noise=sigma_pulse_noise)

#  initialize MemristiveQFunction
n_problem = 2
scaling_factor = 5e10
state_size = 3
action_size = 2
spec = MultiWeightSynapseSpec(n_problem=n_problem, scaling_factor=scaling_factor)
q_function = MemristiveQFunction(state_size=state_size, action_size=action_size, spec=spec, params=params)
print("Memristive Q-Function initialized.")



def log_iteration(q_function, current_state, action, sign_gradient, iteration):
    syn = q_function.get_synapse(current_state, action)

    # States
    idx_0 = syn.get_positive_crosspoint_state(0)[0]
    idx_1 = syn.get_positive_crosspoint_state(1)[0]
    idx_bias = syn.get_bias_crosspoint_state()[0]

    # Conductances
    G0_p  = syn.get_positive_crosspoint_conductance_p(0)
    G0_ap = syn.get_positive_crosspoint_conductance_ap(0)
    G1_p  = syn.get_positive_crosspoint_conductance_p(1)
    G1_ap = syn.get_positive_crosspoint_conductance_ap(1)
    G_bias = syn.get_bias_crosspoint_conductance()

    # Weights
    w0 = syn.weight(0)
    w1 = syn.weight(1)

    # Clean structured print
    print(f"\n{'='*70}")
    print(f"Iteration {iteration} | sign_gradient = {sign_gradient}")
    print(f"{'-'*70}")
    print(f"Indices   : index_0={idx_0}, index_1={idx_1}, index_bias={idx_bias}")
    print(f"Gradients : "
          f"G1_p={G0_p:.9f}, G1_ap={G0_ap:.9f} | "
          f"G2_p={G1_p:.9f}, G2_ap={G1_ap:.9f} | "
          f"G_bias={G_bias:.9f}")
    print(f"Weights   : w0={w0:.4f}, w1={w1:.4f}")


# Configuration: target weights and loop parameters
goal_w0 = -20.8
goal_w1 = 22.5
TOLERANCE = 0.05  # acceptable deviation from target
MAX_ITER = 10000000


def choose_signs(goal_w0, goal_w1, w0, w1):
    """Return sign vector [s0, s1] using the user's rule:
       if (goal - weight) > 0 => sign = -1
       if (goal - weight) < 0 => sign = +1
       if equal => 0
    """
    s0 = 0
    if (goal_w0 - w0) > 0:
        s0 = -1
    elif (goal_w0 - w0) < 0:
        s0 = 1

    s1 = 0
    if (goal_w1 - w1) > 0:
        s1 = -1
    elif (goal_w1 - w1) < 0:
        s1 = 1

    return [s0, s1]


def drive_to_goals(q_function, current_state, action, goal_w0, goal_w1, tol, max_iter):
    """Iteratively update the synapse until both weights reach their goals (within tol)
       or until max_iter is reached. Logs progress each update.

       Returns: (success:bool, w0_history:list, w1_history:list, idx0_history:list, idx1_history:list, idx_bias_history:list)
    """
    syn = q_function.get_synapse(current_state, action)
    w0 = syn.weight(0)
    w1 = syn.weight(1)

    # initial indices
    idx_0 = syn.get_positive_crosspoint_state(0)[0]
    idx_1 = syn.get_positive_crosspoint_state(1)[0]
    idx_bias = syn.get_bias_crosspoint_state()[0]

    # histories for plotting
    w0_history = [w0]
    w1_history = [w1]
    idx0_history = [idx_0]
    idx1_history = [idx_1]
    idx_bias_history = [idx_bias]

    print(f"\nStarting iterative drive to goals: w0={goal_w0}, w1={goal_w1}, tol={tol}, max_iter={max_iter}\n")

    iteration = 1
    while iteration <= max_iter:
        err0 = goal_w0 - w0
        err1 = goal_w1 - w1

        if abs(err0) <= tol and abs(err1) <= tol:
            print(f"Reached goals within tolerance after {iteration-1} updates.")
            print(f"Final weights: w0={w0:.4f}, w1={w1:.4f}")
            return True, w0_history, w1_history, idx0_history, idx1_history, idx_bias_history

        sign = choose_signs(goal_w0, goal_w1, w0, w1)

        # Apply update using the user's requested API call
        q_function.update_q_value(current_state, action, sign)

        # Log iteration (log_iteration reads current synapse state)
        log_iteration(q_function, current_state, action, sign, iteration)

        # Refresh weights and indices
        syn = q_function.get_synapse(current_state, action)
        w0 = syn.weight(0)
        w1 = syn.weight(1)

        idx_0 = syn.get_positive_crosspoint_state(0)[0]
        idx_1 = syn.get_positive_crosspoint_state(1)[0]
        idx_bias = syn.get_bias_crosspoint_state()[0]

        # append to histories
        w0_history.append(w0)
        w1_history.append(w1)
        idx0_history.append(idx_0)
        idx1_history.append(idx_1)
        idx_bias_history.append(idx_bias)

        iteration += 1

    # If we exhaust the loop
    print(f"Failed to reach targets within {max_iter} iterations. Last weights: w0={w0:.4f}, w1={w1:.4f}")
    return False, w0_history, w1_history, idx0_history, idx1_history, idx_bias_history


if __name__ == "__main__":
    # Allow quick local testing by setting FAST_TEST=1 in environment
    fast_test = os.environ.get('FAST_TEST', '0') == '1'
    run_max = 50 if fast_test else MAX_ITER

    (success, w0_hist, w1_hist,
     idx0_hist, idx1_hist, idx_bias_hist) = drive_to_goals(q_function, current_state, action, goal_w0, goal_w1, TOLERANCE, run_max)
    print("Done.")

    # Plotting: attempt to import matplotlib and create two plots
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available; install it to see plots: pip install matplotlib")
        raise

    # 1) weights vs iteration
    plt.figure(figsize=(8, 4))
    iters = list(range(len(w0_hist)))
    plt.plot(iters, w0_hist, label='w0')
    plt.plot(iters, w1_hist, label='w1')
    plt.xlabel('Iteration')
    plt.ylabel('Weight Value')
    plt.title('Weights vs Iteration')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out1 = 'weights_vs_iteration.png'
    plt.savefig(out1)
    print(f"Saved plot: {out1}")
    plt.close()

    # 2) indices vs iteration (idx_0, idx_1, idx_bias)
    try:
        plt.figure(figsize=(8, 4))
        iters_idx = list(range(len(idx0_hist)))
        plt.plot(iters_idx, idx0_hist, label='idx_0')
        plt.plot(iters_idx, idx1_hist, label='idx_1')
        plt.plot(iters_idx, idx_bias_hist, label='idx_bias')
        plt.xlabel('Iteration')
        plt.ylabel('Index Value')
        plt.title('Crosspoint Indices vs Iteration')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out2 = 'indices_vs_iteration.png'
        plt.savefig(out2)
        print(f"Saved plot: {out2}")
        plt.close()
    except Exception:
        # plotting shouldn't break main flow
        print('Failed to create indices plot (matplotlib error)')

    # Summarize
    if success:
        print(f"Drive finished successfully. Final weights saved/marked on plots.")
    else:
        print(f"Drive did not reach exact targets; last weights saved/marked on plots.")
