from link_rl.b_environments.ai_economist import foundation
from link_rl.b_environments.ai_economist.foundation.scenarios.utils import plotting
from typing import Optional
import numpy as np
import gym
from gym import spaces
import warnings
import matplotlib.pyplot as plt


_BIG_NUMBER = 1e20

class EconomistWrapper(gym.Env):
    def __init__(self):
        self.env_config = {
            # ===== SCENARIO CLASS =====
            # Which Scenario class to use: the class's name in the Scenario Registry (foundation.scenarios).
            # The environment object will be an instance of the Scenario class.
            'scenario_name': 'layout_from_file/simple_wood_and_stone',

            # ===== COMPONENTS =====
            # Which components to use (specified as list of ("component_name", {component_kwargs}) tuples).
            #   "component_name" refers to the Component class's name in the Component Registry (foundation.components)
            #   {component_kwargs} is a dictionary of kwargs passed to the Component class
            # The order in which components reset, step, and generate obs follows their listed order below.
            'components': [
                # (1) Building houses
                ('Build', {'skill_dist': "pareto", 'payment_max_skill_multiplier': 3}),
                # (2) Trading collectible resources
                ('ContinuousDoubleAuction', {'max_num_orders': 5}),
                # (3) Movement and resource collection
                ('Gather', {}),
            ],

            # ===== SCENARIO CLASS ARGUMENTS =====
            # (optional) kwargs that are added by the Scenario class (i.e. not defined in BaseEnvironment)
            'env_layout_file': 'quadrant_25x25_20each_30clump.txt',
            'starting_agent_coin': 10,
            'fixed_four_skill_and_loc': True,

            # ===== STANDARD ARGUMENTS ======
            # kwargs that are used by every Scenario class (i.e. defined in BaseEnvironment)
            'n_agents': 4,  # Number of non-planner agents (must be > 1)
            'world_size': [25, 25],  # [Height, Width] of the env world
            'episode_length': 1000,  # Number of timesteps per episode

            # In multi-action-mode, the policy selects an action for each action subspace (defined in component code).
            # Otherwise, the policy selects only 1 action.
            'multi_action_mode_agents': False,
            'multi_action_mode_planner': True,

            # When flattening observations, concatenate scalar & vector observations before output.
            # Otherwise, return observations with minimal processing.
            'flatten_observations': False,
            # When Flattening masks, concatenate each action subspace mask into a single array.
            # Note: flatten_masks = True is required for masking action logits in the code below.
            'flatten_masks': True,
        }

        self.env = foundation.make_env_instance(**self.env_config)

        obs = self.env.reset()

        self.observation_space = self._dict_to_obs_spaces_dict(obs)
        self.action_space = self._dict_to_action_spaces_dict(obs)

    def _dict_to_obs_spaces_dict(self, obs):
        dict_of_spaces = {}
        for k, v in obs.items():

            # list of lists are listified np arrays
            _v = v
            if isinstance(v, list):
                _v = np.array(v)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                _v = np.array([v])

            # assign Space
            if isinstance(_v, np.ndarray):
                x = float(_BIG_NUMBER)
                # Warnings for extreme values
                if np.max(_v) > x:
                    warnings.warn("Input is too large!")
                if np.min(_v) < -x:
                    warnings.warn("Input is too small!")
                box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                # This loop avoids issues with overflow to make sure low/high are good.
                while not low_high_valid:
                    x = x // 2
                    box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                    low_high_valid = (box.low < 0).all() and (box.high > 0).all()

                dict_of_spaces[k] = box

            elif isinstance(_v, dict):
                dict_of_spaces[k] = self._dict_to_obs_spaces_dict(_v)
            else:
                raise TypeError

        return spaces.Dict(dict_of_spaces)

    def _dict_to_action_spaces_dict(self, obs):
        dict_of_spaces = {}

        for k, v in obs.items():
            if k != 'p' and self.env.world.agents[int(k)].multi_action_mode:
                dict_of_spaces[k] = spaces.MultiDiscrete(
                    self.env.get_agent(str(k)).action_spaces
                )
            elif k == 'p' and self.env.world.planner.multi_action_mode:
                dict_of_spaces[k] = spaces.MultiDiscrete(
                    self.env.get_agent("p").action_spaces
                )
            else:
                dict_of_spaces[k] = spaces.Discrete(
                    self.env.get_agent(str(k)).action_spaces
                )

        return spaces.Dict(dict_of_spaces)


    # def n_agents(self):
    #     return self.env.n_agents

    def reset(self, *, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        observation = self.env.reset()

        info = self.get_info()

        if return_info:
            return observation, info
        else:
            return observation

    def get_info(self):
        info = None
        return info

    def step(self, action_dict):
        obs, rew, done, info = self.env.step(action_dict)

        return obs, rew, done, info

#
# class Dummy_Agent:
#     def sample_random_action(self, agent, mask):
#         """Sample random UNMASKED action(s) for agent."""
#         # Return a list of actions: 1 for each action subspace
#
#         return np.random.choice(np.arange(agent.action_spaces), p=mask / mask.sum())  # possible action 1, unpossible action 0
#
#     def get_action(self, env, observation):
#         assert observation is not None
#         """Samples random UNMASKED actions for each agent in obs."""
#
#         actions = {
#             a_idx: self.sample_random_action(env.get_agent(a_idx), a_obs['action_mask'])
#             for a_idx, a_obs in observation.items()
#         }
#
#         return actions
#
#
# def run_env(plot_every=100):
#     env = Env().get_env()
#     agents = Dummy_Agent()
#
#     total_episodes = 10
#
#     for ep in range(1, total_episodes + 1):
#         fig, ax = plt.subplots(1, 1, figsize=(10, 10))
#
#         # Reset
#         obs = env.reset(force_dense_logging=False)
#
#         # Interaction loop (w/ plotting)
#         for t in range(env.episode_length):
#             actions = agents.get_action(env, obs)
#             obs, rew, done, info = env.step(actions)
#
#             if ((t + 1) % plot_every) == 0:
#                 plotting.plot_env_state(env, ax)
#                 ax.set_aspect('equal')
#                 fig.show()
#
#         if ((t + 1) % plot_every) != 0:
#             plotting.plot_env_state(env, ax)
#             ax.set_aspect('equal')
#             fig.show()
#
# def print_obs():
#     env = Env().get_env()
#     obs = env.reset(force_dense_logging=False)
#
#     for a_idx, a_obs in obs.items():
#         print(a_idx)
#         print("obs")
#         for key in a_obs:
#             print(key)
#         print()
#         #print(a_obs)
#
#
#
if __name__ == "__main__":
    env = EconomistWrapper()

    # print(env.observation_space)
    # print(len(env.observation_space))
    #
    # for key in env.observation_space:
    #     print("key : ", key)
    #     print(env.observation_space[key])

    print(env.action_space)
    #run_env()
    #print_obs()