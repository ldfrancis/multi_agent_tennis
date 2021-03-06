import gym
from unityagents import UnityEnvironment


class ReacherEnv(gym.Env):

    def __init__(self, filename):
        super(ReacherEnv, self).__init__()
        self.unity_env = UnityEnvironment(file_name=filename)
        self.brain_name = self.unity_env.brain_names[0]
        self.brain = self.unity_env.brains[self.brain_name]
        action_size = int(self.brain.vector_action_space_size)
        self.action_space = gym.spaces.Box(-1, 1, (action_size,))
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(37,))
        self.num_agents = 20

    def reset(self, train_mode=True):
        env_info = self.unity_env.reset(train_mode=True)[self.brain_name]
        state = env_info.vector_observations
        return state

    def step(self, action):
        env_info = self.unity_env.step(action)[self.brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        return next_state, reward, done, env_info

    def close(self):
        return self.unity_env.close()

    def render(self, mode='human'):
        pass

