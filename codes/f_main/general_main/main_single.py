# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
from collections import deque

import torch
import os, sys
import numpy as np
import wandb

from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.e_utils.experience import ExperienceSourceFirstLast
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model, print_environment_info, print_agent_info
from codes.e_utils.experience_tracker import RewardTracker
from codes.e_utils.logger import get_logger
from codes.e_utils.names import RLAlgorithmName, EnvironmentName


WANDB_DIR = os.path.join(PROJECT_HOME, "out", "wandb")
if not os.path.exists(WANDB_DIR):
    os.makedirs(WANDB_DIR)

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_logger = get_logger("main_single")


def main(params):
    train_env = rl_utils.get_environment(params=params)
    test_env = rl_utils.get_environment(params=params)
    print_environment_info(train_env, params)

    agent, epsilon_tracker = rl_utils.get_rl_agent(env=train_env, worker_id=0, params=params, device=device)
    print_agent_info(agent, epsilon_tracker, params)

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

    experience_source = ExperienceSourceFirstLast(
        env=train_env, agent=agent, gamma=params.GAMMA, n_step=params.N_STEP, vectorized=True
    )

    agent.set_experience_source_to_buffer(experience_source=experience_source)

    stat = None
    step_idx = 0
    loss_queue = deque(maxlen=100)
    solved = False

    if params.WANDB:
        wandb.watch(agent.model.base)

    episode = 0
    with RewardTracker(params=params, frame=False, stat=stat, early_stopping=None) as reward_tracker:
        try:
            while step_idx < params.MAX_GLOBAL_STEP:
                step_idx += params.TRAIN_STEP_FREQ
                last_experience = agent.buffer.populate(params.TRAIN_STEP_FREQ)

                if epsilon_tracker:
                    epsilon_tracker.udpate(step_idx)

                episode_rewards, episode_steps = experience_source.pop_episode_reward_and_done_step_lst()

                if len(episode_rewards) >= 1:
                    print(episode_rewards, episode_steps)

                if episode_rewards and episode_steps:
                    for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                        episode += 1
                        epsilon = agent.action_selector.epsilon if hasattr(agent.action_selector, 'epsilon') else None
                        mean_loss = np.mean(loss_queue) if len(loss_queue) > 0 else 0.0

                        solved, mean_episode_reward = reward_tracker.set_episode_reward(
                            episode_reward=current_episode_reward, episode_done_step=step_idx, epsilon=epsilon,
                            last_info=last_experience.info, current_episode_step=current_episode_step,
                            mean_loss=mean_loss, model=agent.model, wandb=wandb
                        )

                        if solved:
                            save_model(
                                MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, mean_episode_reward
                            )
                            break

                if solved:
                    break

                if isinstance(agent, OnPolicyAgent):
                    if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0, RLAlgorithmName.DISCRETE_PPO_V0]:
                        if len(agent.buffer) < params.PPO_TRAJECTORY_SIZE:
                            continue
                    else:
                        if len(agent.buffer) < params.BATCH_SIZE:
                            continue
                    _, last_loss, _ = agent.train_net(step_idx=step_idx)
                    agent.buffer.clear()
                elif isinstance(agent, OffPolicyAgent):
                    if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                        continue
                    _, last_loss, _ = agent.train_net(step_idx=step_idx)
                else:
                    raise ValueError()

                loss_queue.append(last_loss)

                if hasattr(params, "PER_RANK_BASED") and getattr(params, "PER_RANK_BASED"):
                    if step_idx % 100 < params.TRAIN_STEP_FREQ:
                        agent.buffer.rebalance()

            if params.SAVE_AT_MAX_GLOBAL_STEPS:
                save_model(
                    MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, mean_episode_reward
                )
        finally:
            if params.ENVIRONMENT_ID in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
                train_env.stop()
                test_env.stop()


if __name__ == "__main__":
    from codes.a_config.parameters import PARAMETERS as parameters
    params = parameters
    main(params)

