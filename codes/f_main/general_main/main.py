# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
from collections import deque

import torch
import torch.multiprocessing as mp
import os, sys
import numpy as np
import wandb

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model, print_environment_info
from codes.e_utils.experience_tracker import RewardTracker
from codes.e_utils.logger import get_logger
from codes.e_utils.names import DeepLearningModelName, RLAlgorithmName, EnvironmentName
from codes.e_utils.experience import ExperienceSourceFirstLast

WANDB_DIR = os.path.join(PROJECT_HOME, "out", "wandb")
if not os.path.exists(WANDB_DIR):
    os.makedirs(WANDB_DIR)

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

my_logger = get_logger("main")


def play_func(exp_queue, loss_queue, agent, epsilon_tracker):
    if params.WANDB:
        configuration = {key: getattr(params, key) for key in dir(params) if not key.startswith("__")}
        wandb.init(
            project=params.wandb_project,
            entity=params.wandb_entity,
            dir=WANDB_DIR,
            config=configuration
        )
        run_name = wandb.run.name
        run_number = run_name.split("-")[-1]
        wandb.run.name = "{0}_{1}_{2}_{3}".format(
            run_number, params.ENVIRONMENT_ID.value, agent.__name__, agent.model.__name__
        )
        wandb.run.save()

    env = rl_utils.get_environment(params=params)
    print_environment_info(env, params)

    if params.DEEP_LEARNING_MODEL in [
        DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU,
        DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU_ATTENTION
    ]:
        step_length = params.RNN_STEP_LENGTH
    else:
        step_length = -1

    experience_source = ExperienceSourceFirstLast(
        env=env, agent=agent, gamma=params.GAMMA, n_step=params.N_STEP, vectorized=True
    )

    exp_source_iter = iter(experience_source)
    stat = None
    step_idx = 0
    loss_dequeue = deque(maxlen=100)

    solved = False

    if params.WANDB:
        wandb.watch(agent.model.base)

    episode = 0
    with RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        try:
            while step_idx < params.MAX_GLOBAL_STEP:
                # 1 스텝 진행하고 exp를 exp_queue에 넣음
                step_idx += 1
                exp = next(exp_source_iter)
                exp_queue.put(exp)

                loss = loss_queue.get()
                if loss is None:
                    solved = True
                    break
                else:
                    loss_dequeue.append(loss)

                if epsilon_tracker:
                    epsilon_tracker.udpate(step_idx)

                episode_rewards, episode_steps = experience_source.pop_episode_reward_and_done_step_lst()

                if episode_rewards and episode_steps:
                    for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                        episode += 1

                        epsilon = agent.action_selector.epsilon if hasattr(agent.action_selector, 'epsilon') else None
                        mean_loss = np.mean(loss_dequeue) if len(loss_dequeue) > 0 else 0.0

                        solved, mean_episode_reward = reward_tracker.set_episode_reward(
                            episode_reward=current_episode_reward, episode_done_step=step_idx, epsilon=epsilon,
                            last_info=exp.info, current_episode_step=current_episode_step, mean_loss=mean_loss,
                            model=agent.model, wandb=wandb
                        )

                        if solved:
                            save_model(
                                MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, mean_episode_reward
                            )
                            if solved:
                                break

                if solved:
                    break

            if params.SAVE_AT_MAX_GLOBAL_STEPS:
                save_model(
                    MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, mean_episode_reward
                )
        finally:
            if params.ENVIRONMENT_ID in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
                env.stop()

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    env = rl_utils.get_single_environment(params=params)
    agent, epsilon_tracker = rl_utils.get_rl_agent(env=env, worker_id=0, params=params, device=device)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    loss_queue = mp.Queue(maxsize=params.AVG_EPISODE_SIZE_FOR_STAT)
    play_proc = mp.Process(target=play_func, args=(exp_queue, loss_queue, agent, epsilon_tracker))
    play_proc.start()

    time.sleep(0.5)

    step_idx = 0
    trajectory = []

    while play_proc.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if exp is None:
                loss_queue.put(None)
                play_proc.join()
                break
            agent.buffer._add(exp)
            if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0]:
                assert params.TRAIN_STEP_FREQ == 1 and exp is not None
                trajectory.append(exp)

        if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0]:
            if len(trajectory) < params.PPO_TRAJECTORY_SIZE:
                loss_queue.put(0.0)
                continue

            _, loss, _ = agent.train_net(trajectory=trajectory)
            trajectory.clear()

        elif params.RL_ALGORITHM in [RLAlgorithmName.DDPG_V0]:
            if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                loss_queue.put(0.0)
                continue
            _, loss, _ = agent.train_net(step_idx=step_idx)

        elif params.RL_ALGORITHM in [RLAlgorithmName.DQN_V0]:
            if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                loss_queue.put(0.0)
                continue
            _, loss = agent.train_net(step_idx=step_idx)

        elif params.RL_ALGORITHM in [RLAlgorithmName.DISCRETE_A2C_V0, RLAlgorithmName.CONTINUOUS_A2C_V0, RLAlgorithmName.SAC_V0]:
            if len(agent.buffer) < params.BATCH_SIZE:
                loss_queue.put(0.0)
                continue
            _, loss, _ = agent.train_net(step_idx=step_idx)

        else:
            raise ValueError()

        loss_queue.put(loss)

        if hasattr(params, "PER_RANK_BASED") and getattr(params, "PER_RANK_BASED"):
            if step_idx % 100 < params.TRAIN_STEP_FREQ:
                agent.buffer.rebalance()


if __name__ == "__main__":
    main()