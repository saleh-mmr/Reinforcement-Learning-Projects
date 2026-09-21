# import gymnasium as gym
# import numpy as np
#
#
# class RewardWrapper(gym.RewardWrapper):
#     """
#     Wrapper class for modifying rewards in the MountainCar-v0 environment.
#
#     Args:
#         env (gym.Env): The environment to wrap.
#     """
#
#     def __init__(self, env):
#         super().__init__(env)
#
#     def reward(self, state):
#         """
#         Modifies the reward based on the current state of the environment.
#
#         Args:
#             state (numpy.ndarray): The current state of the environment.
#
#         Returns:
#             float: The modified reward.
#         """
#
#         current_position, current_velocity = state  # extract the position and current velocity based on the state
#
#         # Interpolate the value to the desired range (because the velocity normalized value would be in range of 0 to 1 and now it would be in range of -0.5 to 0.5)
#         current_velocity = np.interp(current_velocity, np.array([0, 1]), np.array([-0.5, 0.5]))
#
#         # (1) Calculate the modified reward based on the current position and velocity of the car.
#         degree = current_position * 360
#         degree2radian = np.deg2rad(degree)
#         modified_reward = np.cos(degree2radian) + 2 * np.abs(current_velocity)
#
#         # (2) Step limitation
#         modified_reward -= 0.5  # Subtract 0.5 to adjust the base reward (to limit useless steps).
#
#         # (3) Check if the car has surpassed a threshold of the path and is closer to the goal
#         if current_position > 0.98:
#             modified_reward += 20  # Add a bonus reward (Reached the goal)
#         elif current_position > 0.92:
#             modified_reward += 10  # So close to the goal
#         elif current_position > 0.82:
#             modified_reward += 6  # car is closer to the goal
#         elif current_position > 0.65:
#             modified_reward += 1 - np.exp(-2 * current_position)  # car is getting close. Thus, giving reward based on the position and the further it reached
#
#         # (4) Check if the car is coming down with velocity from left and goes with full velocity to right
#         initial_position = 0.40842572  # Normalized value of initial position of the car which is extracted manually
#
#         if current_velocity > 0.3 and current_position > initial_position + 0.1:
#             modified_reward += 1 + 2 * current_position  # Add a bonus reward for this desired behavior
#
#         return modified_reward
import gymnasium as gym
import numpy as np


class RewardWrapper(gym.RewardWrapper):
    """
    Effective and stable reward shaping for MountainCar-v0.
    - Encourages velocity strongly
    - Encourages climbing
    - Gives small potential-based boost
    - Smooth bonus near goal
    - Compatible with normalized observations
    """

    def __init__(self, env):
        super().__init__(env)

        # Real coordinate bounds
        self.pos_min, self.vel_min = env.observation_space.low
        self.pos_max, self.vel_max = env.observation_space.high

        # Discount
        self.gamma = 0.99


    def _denormalize(self, state):
        """Convert normalized state [0,1] -> real physics coordinates."""
        position = self.pos_min + state[0] * (self.pos_max - self.pos_min)
        velocity = self.vel_min + state[1] * (self.vel_max - self.vel_min)
        return position, velocity


    def reward(self, state):
        # Convert normalized to real
        position, velocity = self._denormalize(state)

        # ---------------------------
        # 1. Strong velocity shaping
        # ---------------------------
        momentum_reward = abs(velocity) * 30.0
        # ↑ this is the key for DQN to learn to swing


        # ---------------------------
        # 2. Position shaping (climbing reward)
        # ---------------------------
        progress_reward = (position - self.pos_min) / (self.pos_max - self.pos_min)
        progress_reward *= 2.0


        # ---------------------------
        # 3. Small per-step penalty
        # ---------------------------
        step_penalty = -0.1


        # ---------------------------
        # 4. Smooth goal bonus
        # ---------------------------
        goal_bonus = 0.0
        if position > 0.45:
            goal_bonus += 3.0
        if position > 0.50:
            goal_bonus += 10.0
        if position >= 0.52:   # goal threshold
            goal_bonus += 25.0    # strong enough to be meaningful


        # ---------------------------
        # Final shaped reward
        # ---------------------------
        final_reward = momentum_reward + progress_reward + goal_bonus + step_penalty

        return final_reward
