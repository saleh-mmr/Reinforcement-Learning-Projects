import gymnasium as gym

from DQN.MountainCar.observation_wrapper import ObservationWrapper
from DQN.MountainCar.reward_wrapper import RewardWrapper


class StepWrapper(gym.Wrapper):


    def __init__(self, env):

        super().__init__(env)
        self.observation_wrapper = ObservationWrapper(env)
        self.reward_wrapper = RewardWrapper(env)


    def step(self, action):
        state, reward, done, truncation, info = self.env.step(action)

        modified_state = self.observation_wrapper.observation(state)
        modified_reward = self.reward_wrapper.reward(modified_state)

        return modified_state, modified_reward, done, truncation, info

    def reset(self, seed):
        state, info = self.env.reset(seed=seed)
        modified_state = self.observation_wrapper.observation(state)
        return modified_state, info





