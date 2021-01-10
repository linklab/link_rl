# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import pickle

import torch
import os, sys
import numpy as np

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model, print_environment_info
from codes.e_utils.experience_single import ExperienceSourceSingleEnvFirstLast
from codes.e_utils.experience_tracker import RewardTracker
from codes.e_utils.logger import get_logger
from codes.e_utils.names import DeepLearningModelName, RLAlgorithmName

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

my_logger = get_logger("openai_pendulum_ddpg")


def main(params):
    env = rl_utils.get_environment(owner="actual_worker", params=params)
    print_environment_info(env, params)

    agent, epsilon_tracker = rl_utils.get_rl_agent(env=env, worker_id=0, params=params, device=device)

    if params.DEEP_LEARNING_MODEL in [
        DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU,
        DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU_ATTENTION
    ]:
        step_length = params.RNN_STEP_LENGTH
    else:
        step_length = -1

    experience_source = ExperienceSourceSingleEnvFirstLast(
        env=env, agent=agent, gamma=params.GAMMA, steps_count=params.N_STEP, step_length=step_length
    )

    agent.set_experience_source_to_buffer(experience_source=experience_source)

    stat = None
    step_idx = 0
    loss_list = []
    episode_reward_list = []
    episode_mean_loss_list = []

    trajectory = []

    with RewardTracker(params=params, frame=False, stat=stat, early_stopping=None) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEP:
            step_idx += params.TRAIN_STEP_FREQ
            last_experience = agent.buffer.populate(params.TRAIN_STEP_FREQ)

            if epsilon_tracker:
                epsilon_tracker.udpate(step_idx)

            episode_rewards_and_done_step_lst = experience_source.pop_episode_reward_and_done_step_lst()
            solved = False

            if episode_rewards_and_done_step_lst:

                episode_rewards, done_steps = zip(*episode_rewards_and_done_step_lst)

                for current_episode_reward in episode_rewards:
                    epsilon = agent.action_selector.epsilon if hasattr(agent.action_selector, 'epsilon') else None
                    mean_loss = np.mean(loss_list) if len(loss_list) > 0 else 0.0

                    ##################################################
                    #####  FOR PAPER
                    ##################################################
                    episode_reward_list.append(current_episode_reward)
                    episode_mean_loss_list.append(mean_loss)
                    ##################################################

                    solved, mean_episode_reward = reward_tracker.set_episode_reward(
                        current_episode_reward, step_idx, epsilon, last_info=last_experience.info,
                        mean_loss=mean_loss, model=agent.model
                    )

                    loss_list.clear()

                    if solved:
                        save_model(
                            MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, mean_episode_reward
                        )
                        if solved:
                            break
            if solved:
                break

            if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_FAST_V0]:
                #print(last_experience[0])
                trajectory.append(last_experience)
                if len(trajectory) < params.TRAJECTORY_SIZE:
                    continue
            elif params.RL_ALGORITHM in [RLAlgorithmName.DDPG_FAST_V0, RLAlgorithmName.DQN_FAST_V0]:
                if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                    continue
            else:
                if len(agent.buffer) < params.BATCH_SIZE:
                    continue

            if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_FAST_V0]:
                _, last_loss, _ = agent.train_net(trajectory=trajectory)
                trajectory.clear()
            elif params.RL_ALGORITHM in [
                RLAlgorithmName.DDPG_FAST_V0,
                RLAlgorithmName.DISCRETE_A2C_FAST_V0,
                RLAlgorithmName.CONTINUOUS_A2C_FAST_V0
            ]:
                _, last_loss, _ = agent.train_net(step_idx=step_idx)
            elif params.RL_ALGORITHM == RLAlgorithmName.DQN_FAST_V0:
                _, last_loss = agent.train_net(step_idx=step_idx)
            else:
                raise ValueError()

            loss_list.append(last_loss)

        if params.SAVE_AT_MAX_GLOBAL_STEPS:
            save_model(
                MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, mean_episode_reward
            )

        ##################################################
        #####  FOR PAPER
        ##################################################
        with open('episode_reward_list.dump', 'wb') as f:
            pickle.dump(episode_reward_list, f)

        with open('episode_mean_loss_list.dump', 'wb') as f:
            pickle.dump(episode_mean_loss_list, f)
        ##################################################


if __name__ == "__main__":
    from codes.a_config.parameters import PARAMETERS as parameters
    params = parameters
    main(params)
