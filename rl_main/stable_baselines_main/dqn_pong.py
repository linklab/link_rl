import gym
import numpy as np

from stable_baselines import DQN
from stable_baselines.common import atari_wrappers
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.results_plotter import load_results, ts2xy, X_TIMESTEPS
from stable_baselines.common.callbacks import BaseCallback

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

env = atari_wrappers.make_atari('PongNoFrameskip-v4')
env = atari_wrappers.wrap_deepmind(env)

MODEL_SAVE_DIR = "."

dqn_agent = DQN(
    policy='CnnPolicy', env=env,
    exploration_fraction=0.1, exploration_final_eps=0.01, exploration_initial_eps=1.0,
    train_freq=4, verbose=1
)


class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.best_mean_episode_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            print(self.locals["episode_rewards"][-100:])
            mean_100ep_reward = np.mean(self.locals["episode_rewards"][-100:])

            if self.verbose > 0:
                # print(self.episode_rewards[-100:])
                print("Steps: {} | Best mean reward: {:.2f} | Last mean reward per episode: {:.2f}".format(
                    self.num_timesteps,
                    self.best_mean_episode_reward,
                    mean_100ep_reward
                ))

            # New best model, you could save the agent here
            if mean_100ep_reward > self.best_mean_episode_reward:
                self.best_mean_episode_reward = mean_100ep_reward

                saved_file_name = MODEL_SAVE_DIR + "/dqn_pong_{0}_{1}".format(
                    self.n_calls, int(mean_100ep_reward)
                )

                # Example for saving best model
                if self.verbose > 0:
                    print("Saving new best model to {}".format(saved_file_name))

                self.model.save(saved_file_name)
                print()

        return True


callback = SaveOnBestTrainingRewardCallback(check_freq=1000)

MAX_STEPS = 1000000

dqn_agent = dqn_agent.learn(total_timesteps=MAX_STEPS, callback=callback)