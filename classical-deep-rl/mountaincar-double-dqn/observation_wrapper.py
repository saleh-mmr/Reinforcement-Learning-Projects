import gymnasium as gym
import numpy as np


class ObservationWrapper(gym.ObservationWrapper):


    def __init__(self,env):
        super().__init__(env)
        self.min_value = env.observation_space.low
        self.max_value = env.observation_space.high


    def observation(self, state):

        state = np.clip(state, self.min_value, self.max_value)
        normalized_state = (state - self.min_value) / (self.max_value - self.min_value)
        return  normalized_state

