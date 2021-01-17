import gym
from typing import List, Optional, Tuple, Any

from gym.vector import AsyncVectorEnv, SyncVectorEnv

from codes.d_agents.a0_base_agent import BaseAgent
from codes.e_utils.experience import ExperienceSource, ExperienceSourceFirstLast
from codes.e_utils.tests.gym_vec_test.utils import make_env


class ToyEnv(gym.Env):
    """
    Environment with observation 0..4 and actions 0..2
    Observations are rotated sequentialy mod 5, reward is equal to given action.
    Episodes are having fixed length of 10
    """

    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=3)
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return [self.step_index]

    def step(self, action):
        is_done = self.step_index == 10
        if is_done:
            reward = 10.0
            return [self.step_index % self.observation_space.n], reward, True, {}
        else:
            self.step_index += 1
            reward = -1
            return [self.step_index % self.observation_space.n], reward, False, {}


class DullAgent(BaseAgent):
    """
    Agent always returns the fixed action
    """
    def __init__(self, action: int):
        super(DullAgent, self).__init__()
        self.action = action

    def __call__(self, observations: List[Any], agent_state=None):
        return [self.action for _ in observations], agent_state


if __name__ == "__main__":
    # env = ToyEnv()
    # state = env.reset()
    # print("env.reset() -> %s" % state)
    #
    # next_state, reward, done, info = env.step(1)
    # print(f"env.step(1) -> {next_state}, {reward}, {done}, {info}")
    #
    # next_state, reward, done, info = env.step(2)
    # print(f"env.step(2) -> {next_state}, {reward}, {done}, {info}")
    #
    # for _ in range(10):
    #     r = env.step(0)
    #     print(r)
    #
    # agent = DullAgent(action=1)
    # print("agent:", agent([1, 2])[0])
    #
    # env = ToyEnv()
    # agent = DullAgent(action=1)
    # experience_source = ExperienceSource(env=env, agent=agent, n_step=2)
    # for idx, exp in enumerate(experience_source):
    #     if idx > 15:
    #         break
    #     print(exp)
    #
    # print("!!!!!!!!!!!")
    #
    # experience_source = ExperienceSource(env=env, agent=agent, n_step=4)
    # print(next(iter(experience_source)))
    #
    # print("!!!!!!!!!!! @@@@@")
    # experience_source = ExperienceSource(env=ToyEnv(), agent=agent, n_step=2)
    # for idx, exp in enumerate(experience_source):
    #     if idx > 4:
    #         break
    #     print(idx, exp)
    #
    # print("!!!!!!!!!!! @@@@@")
    # experience_source = ExperienceSource(env=[ToyEnv(), ToyEnv()], agent=agent, n_step=2)
    # for idx, exp in enumerate(experience_source):
    #     if idx > 4:
    #         break
    #     print(idx, exp)
    # print("!!!!!!!!!!! ###########################")

    env_fns = [make_env('CartPole-v0', i) for i in range(3)]
    env = SyncVectorEnv(env_fns)
    assert env.num_envs == 3

    # env = [ToyEnv(), ToyEnv(), ToyEnv()]
    agent = DullAgent(action=1)

    experience_source = ExperienceSourceFirstLast(env=env, agent=agent, n_step=2, gamma=0.99)
    for idx, exp in enumerate(experience_source):
        if idx > 100:
            break
        print(idx, exp)

    # print("!!!!!!!!!!!")
    #
    # print("ExperienceSourceFirstLast")
    # env = ToyEnv()
    # experience_source = ExperienceSourceFirstLast(env, agent, gamma=1.0, n_step=1)
    # for idx, exp in enumerate(experience_source):
    #     print(exp)
    #     if idx > 10:
    #         break
