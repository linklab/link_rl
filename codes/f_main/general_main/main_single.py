# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
import torch
import torch.multiprocessing as mp
import os, sys

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model
from codes.e_utils.experience_single import ExperienceSourceSingleEnvFirstLast
from codes.e_utils.experience_tracker import RewardTracker
from codes.e_utils.logger import get_logger
from codes.e_utils.names import DeepLearningModelName

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

my_logger = get_logger("openai_pendulum_ddpg")


def main():
    mp.set_start_method('spawn')

    env = rl_utils.get_environment(owner="worker", params=params)
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    print("action_min: ", env.action_space.low[0], "action_max:", env.action_space.high[0])
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]

    agent, epsilon_tracker = rl_utils.get_rl_agent(
        env=env, worker_id=0, action_min=action_min, action_max=action_max, params=params
    )

    if params.DEEP_LEARNING_MODEL in [
        DeepLearningModelName.DETERMINISTIC_ACTOR_CRITIC_GRU,
        DeepLearningModelName.DETERMINISTIC_ACTOR_CRITIC_GRU_ATTENTION
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
    last_loss = 0.0

    with RewardTracker(params=params, frame=False, stat=stat, early_stopping=None) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEP:
            step_idx += params.TRAIN_STEP_FREQ
            last_entry = agent.buffer.populate(params.TRAIN_STEP_FREQ)
            epsilon_tracker.udpate(step_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()

            solved = False
            if episode_rewards:
                for current_episode_reward in episode_rewards:
                    solved, mean_episode_reward = reward_tracker.set_episode_reward(
                        current_episode_reward, step_idx, agent.action_selector.epsilon, last_info=last_entry.info,
                        last_loss=last_loss, model=agent.model
                    )

                    if solved:
                        save_model(
                            MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent.model, step_idx, mean_episode_reward
                        )
                        if solved:
                            break
            if solved:
                break

            if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                continue

            _, last_loss, _ = agent.train_net(step_idx=step_idx)


if __name__ == "__main__":
    main()