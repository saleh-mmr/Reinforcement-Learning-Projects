import gymnasium as gym

from observation_wrapper import ObservationWrapper
# from reward_wrapper import RewardWrapper  # Commented out: disabling reward shaping per request


class StepWrapper(gym.Wrapper):


    def __init__(self, env):

        super().__init__(env)
        self.observation_wrapper = ObservationWrapper(env)
        # self.reward_wrapper = RewardWrapper(env)  # Commented out: disabling reward shaping per request


    def step(self, action):
        state, reward, done, truncation, info = self.env.step(action)

        modified_state = self.observation_wrapper.observation(state)
        # modified_reward = self.reward_wrapper.reward(modified_state)  # Commented out: use original env reward

        # Return normalized observation and the original environment reward
        return modified_state, reward, done, truncation, info

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        modified_state = self.observation_wrapper.observation(state)
        return modified_state, info
