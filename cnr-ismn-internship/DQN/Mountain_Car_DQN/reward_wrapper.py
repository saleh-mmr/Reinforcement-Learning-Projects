import gymnasium as gym

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
        step_penalty = -1


        # ---------------------------
        # 4. Smooth goal bonus
        # ---------------------------
        goal_bonus = 0.0
        if position > 0.45:
            goal_bonus += 3.0
        if position > 0.50:
            goal_bonus += 10.0
        if position >= 0.65:
            goal_bonus += 35.0
        if position > 0.95:
            goal_bonus += 100.0


        # ---------------------------
        # Final shaped reward
        # ---------------------------
        final_reward = momentum_reward + progress_reward + goal_bonus + step_penalty

        return final_reward
