# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
import torch
import torch.multiprocessing as mp
import os, sys

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))
print("PROJECT_HOME:", PROJECT_HOME)
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.logger import get_logger
from config.names import DeepLearningModelName
from rl_main import rl_utils
from common.fast_rl.rl_agent import float32_preprocessor

print("PyTorch Version", torch.__version__)

from common.fast_rl import actions, experience, rl_agent, experience_single
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

my_logger = get_logger("openai_pendulum_ddpg")


def play_func(exp_queue, env, net):
    print(env.action_space.low[0], env.action_space.high[0])
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]

    action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=params.ACTION_SCALE)

    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )

    agent = rl_agent.AgentDDPG(
        net, n_actions=1, action_selector=action_selector,
        action_min=action_min, action_max=action_max, device=device, preprocessor=float32_preprocessor
    )

    if params.DEEP_LEARNING_MODEL in [DeepLearningModelName.DDPG_ACTOR_CRITIC_GRU, DeepLearningModelName.DDPG_ACTOR_CRITIC_GRU_ATTENTION]:
        step_length = params.RNN_STEP_LENGTH
    else:
        step_length = -1

    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(
        env, agent, gamma=params.GAMMA, steps_count=params.N_STEP, step_length=step_length
    )

    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForPolicyBasedRL(method="policy_gradient")
    else:
        stat = None

    step_idx = 0

    best_mean_episode_reward = 0.0

    with utils.RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEPS:
            # 1 스텝 진행하고 exp를 exp_queue에 넣음
            step_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            epsilon_tracker.udpate(step_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()
            if episode_rewards:
                current_episode_reward = episode_rewards[0]

                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    current_episode_reward, step_idx, epsilon=action_selector.epsilon
                )

                model_save_condition = [
                    reward_tracker.mean_episode_reward > best_mean_episode_reward,
                    step_idx > params.EPSILON_MIN_STEP
                ]

                if reward_tracker.mean_episode_reward > best_mean_episode_reward:
                    best_mean_episode_reward = reward_tracker.mean_episode_reward

                if all(model_save_condition) or solved:
                    rl_agent.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, step_idx, mean_episode_reward
                    )
                    if solved:
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
            for _ in range(3):
                rl_algorithm.train_net(step_idx=step_idx)


if __name__ == "__main__":
    main()