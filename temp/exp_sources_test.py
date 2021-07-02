import gym
from codes.a_config.parameters_general import PARAMETERS_GENERAL
from codes.e_utils import rl_utils
from codes.e_utils.names import EnvironmentName

gym.logger.set_level(40)

from typing import List, Optional, Tuple, Any

from gym.vector import AsyncVectorEnv, SyncVectorEnv

from codes.d_agents.a0_base_agent import BaseAgent
from codes.e_utils.experience import ExperienceSource, ExperienceSourceFirstLast
from codes.e_utils.tests.gym_vec_test.utils import make_env


class TOY_PARAMETERS(PARAMETERS_GENERAL):
    ENVIRONMENT_ID = EnvironmentName.TOY_V0
    wandb_project = "rl"
    wandb_entity = "bluebibi"
    WANDB = False
    NUM_ENVIRONMENTS = 2


class DullAgent(BaseAgent):
    """
    Agent always returns the fixed action
    """
    def __init__(self, action_space):
        super(DullAgent, self).__init__(
            worker_id=0, params=None, action_shape=action_space, device="cpu",
            action_min=0, action_max=1
        )
        self.action_ = 0

    def __call__(self, observations: List[Any], agent_state=None):
        actions = [self.action_ for _ in observations]
        self.action_ = 1 - self.action_
        return actions, agent_state


def basic_single_env_test(params):
    vec_env = rl_utils.get_environment(params=params)
    env = vec_env.envs[0]
    state = env.reset()
    print(f"[Step {0:>3}] env.reset() -> Next State: %s" % state)

    action_ = 0
    done = False
    step = 0
    while not done:
        step += 1
        next_state, reward, done, info = env.step(action_)
        print(f"[Step {step:>3}] env.step({action_}) "
              f"-> Next State: {next_state}, Reward: {reward:>4}, Done: {done:>4}, Info: {info}")
        action_ = 1 - action_


def agent_experience_source_test(params):
    env = rl_utils.get_environment(params=params)
    agent = DullAgent(action_space=env.action_space.shape)
    print("agent:", agent([1, 2])[0])

    experience_source = ExperienceSource(env=env, agent=agent, n_step=2)
    for idx, exp in enumerate(experience_source):
        if idx >= 20:
            break
        print(exp)


def agent_experience_source_first_last_test(params, num_exps=200):
    env = rl_utils.get_environment(params=params)
    agent = DullAgent(action_space=env.action_space.shape)
    print("agent:", agent([1, 2])[0])

    experience_source = ExperienceSourceFirstLast(env=env, agent=agent, gamma=0.99, n_step=5)

    for idx, exp in enumerate(experience_source):
        if idx >= num_exps:
            break
        print(exp)


if __name__ == "__main__":
    # params = TOY_PARAMETERS()
    # basic_single_env_test(params=params)
    # print("!!!!!!!!!!!")

    # params = TOY_PARAMETERS()
    # agent_experience_source_test(params=params)
    # print("!!!!!!!!!!!")

    params = TOY_PARAMETERS()
    agent_experience_source_first_last_test(params=params)
    #print("!!!!!!!!!!!")
    #
    # params = TOY_PARAMETERS()
    # params.NUM_ENVIRONMENTS = 10
    # agent_experience_source_first_last_test(params=params, num_exps=100)


    # env_fns = [make_env('CartPole-v0', i) for i in range(3)]
    # env = SyncVectorEnv(env_fns)
    # assert env.num_envs == 3
    #
    # # env = [ToyEnv(), ToyEnv(), ToyEnv()]
    # agent = DullAgent(action=1)
    #
    # experience_source = ExperienceSourceFirstLast(env=env, agent=agent, n_step=2, gamma=0.99)
    # for idx, exp in enumerate(experience_source):
    #     if idx > 100:
    #         break
    #     print(idx, exp)

    # print("!!!!!!!!!!!")
    #
    # print("ExperienceSourceFirstLast")
    # env = ToyEnv()
    # experience_source = ExperienceSourceFirstLast(env, agent, gamma=1.0, n_step=1)
    # for idx, exp in enumerate(experience_source):
    #     print(exp)
    #     if idx > 10:
    #         break
