#!/usr/bin/env python3
# PARAMETERS_FAST_RL_PENDULUM_A2C
import time
import torch
import torch.multiprocessing as mp
import sys, os

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.fast_rl.rl_agent import float32_preprocessor
from config.names import PROJECT_HOME
from rl_main import rl_utils
from common.logger import get_logger
from common.fast_rl import experience, rl_agent
from common.fast_rl.common import statistics, utils
from config.parameters import PARAMETERS as params

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

my_logger = get_logger("openai_minitaur_bullet_a2c")

print(torch.__version__)


def play_func(exp_queue, env, net):
    print(env.action_space.low[0], env.action_space.high[0])
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]

    agent = rl_agent.ContinuousActorCriticAgent(
        net, action_min=action_min, action_max=action_max, device=device, preprocessor=float32_preprocessor
    )

    experience_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params.GAMMA, steps_count=params.N_STEP)

    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForPolicyBasedRL(method="policy_gradient")
    else:
        stat = None

    step_idx = 0
    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEP:
            # 1 스텝 진행하고 exp를 exp_queue에 넣음
            step_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            episode_rewards = experience_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    episode_rewards[0], step_idx, epsilon=0.0
                )

                if step_idx >= next_save_frame_idx:
                    rl_agent.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, step_idx, mean_episode_reward
                    )
                    next_save_frame_idx += params.MODEL_SAVE_STEP_PERIOD

                if solved:
                    rl_agent.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, step_idx, mean_episode_reward
                    )
                    break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    env = rl_utils.get_environment(owner="worker", params=params)
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)

    rl_algorithm = rl_utils.get_rl_algorithm(env=env, worker_id=0, logger=my_logger, params=params)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(exp_queue, env, rl_algorithm.model))
    play_proc.start()

    time.sleep(0.5)

    step_idx = 0

    while play_proc.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        exp = None
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            rl_algorithm.buffer._add(exp)

        if len(rl_algorithm.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
            continue

        if exp is not None and exp.last_state is None:
            rl_algorithm.train_net(step_idx=step_idx)


if __name__ == "__main__":
    main()