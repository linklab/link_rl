# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
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
from codes.e_utils.experience_tracker import RewardTracker, EarlyStopping
from codes.e_utils.logger import get_logger
from codes.e_utils.names import RLAlgorithmName, EnvironmentName, AgentMode

WANDB_DIR = os.path.join(PROJECT_HOME, "out", "wandb")
if not os.path.exists(WANDB_DIR):
    os.makedirs(WANDB_DIR)

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

my_logger = get_logger("main_single")


def train_main(params, train_env, test_env):
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

    step_idx = 0
    loss_queue = deque(maxlen=100)

    if params.WANDB:
        wandb.watch(agent.model.base)

    early_stopping = EarlyStopping(
        patience=params.STOP_PATIENCE_COUNT,
        evaluation_min_threshold=params.STOP_MEAN_EPISODE_REWARD,
        evaluation_min_step_idx=params.EPSILON_MIN_STEP if hasattr(params, "EPSILON_MIN_STEP") and params.EPSILON_MIN_STEP else None,
        verbose=True,
        delta=0.0,
        model_save_dir=MODEL_SAVE_DIR,
        model_save_file_prefix=params.ENVIRONMENT_ID.value,
        agent=agent
    )

    episode = 0
    solved = False
    with RewardTracker(params=params) as reward_tracker:
        try:
            while step_idx < params.MAX_GLOBAL_STEP:
                step_idx += params.TRAIN_STEP_FREQ
                last_experience = agent.buffer.populate(params.TRAIN_STEP_FREQ)

                if epsilon_tracker:
                    epsilon_tracker.udpate(step_idx)

                episode_rewards, episode_steps = experience_source.pop_episode_reward_and_done_step_lst()

                if episode_rewards and episode_steps:
                    for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                        episode += 1
                        epsilon = agent.train_action_selector.epsilon if hasattr(agent.train_action_selector, 'epsilon') else None
                        mean_loss = np.mean(loss_queue) if len(loss_queue) > 0 else 0.0

                        train_mean_episode_reward, speed = reward_tracker.set_episode_reward(
                            episode_reward=current_episode_reward, episode_done_step=step_idx, epsilon=epsilon,
                            last_info=last_experience.info, mean_loss=mean_loss
                        )

                    test_episode_reward_mean, test_episode_reward_std = test(params, test_env, agent)

                    print("[TEST EPISODES] EPISODE REWARD: {0:.4f}\u00B1{1:.4f}".format(
                        test_episode_reward_mean, test_episode_reward_std
                    ))

                    solved = early_stopping.evaluate(
                        evaluation_value=test_episode_reward_mean, model=agent.model, episode_done_step=step_idx
                    )

                    if params.WANDB:
                        wandb_info = {
                            "train episode reward": train_mean_episode_reward,
                            "mean_loss": mean_loss,
                            "steps/episode": current_episode_step,
                            "speed": speed,
                            "step_idx": step_idx,
                            "episode": episode
                        }
                        if epsilon:
                            wandb_info["epsilon"] = epsilon
                        wandb.log(wandb_info)

                if solved:
                    print("Solved in {0} steps and {1} episodes!".format(step_idx, episode))
                    break

                if isinstance(agent, OnPolicyAgent):
                    if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0, RLAlgorithmName.DISCRETE_PPO_V0]:
                        if len(agent.buffer) < params.PPO_TRAJECTORY_SIZE:
                            continue
                    else:
                        if len(agent.buffer) < params.BATCH_SIZE:
                            continue
                    _, last_loss, _ = agent.train(step_idx=step_idx)
                    agent.buffer.clear()
                elif isinstance(agent, OffPolicyAgent):
                    if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                        continue
                    _, last_loss, _ = agent.train(step_idx=step_idx)
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


def test(params, test_env, agent):
    agent.agent_mode = AgentMode.TEST

    num_step = 0

    episode_rewards = np.zeros(params.TEST_NUM_EPISODES)

    for test_episode in range(params.TEST_NUM_EPISODES):
        done = False
        episode_reward = 0

        state = test_env.reset()

        num_episode_step = 0
        while not done:
            test_env.render()

            num_step += 1
            num_episode_step += 1

            state = np.expand_dims(state, axis=0)

            action, _, = agent(state)

            next_state, reward, done, info = test_env.step(action[0])
            state = next_state
            episode_reward += reward

        episode_rewards[test_episode] = episode_reward

        print("[TEST EPISODE {0}] EPISODE STEPS: {1}, TOTAL STEPS: {2}, EPISODE REWARD: {3}".format(
            test_episode, num_episode_step, num_step, episode_reward
        ))

    return np.mean(episode_rewards), np.std(episode_rewards)


if __name__ == "__main__":
    from codes.a_config.parameters import PARAMETERS as parameters
    params = parameters

    train_env = rl_utils.get_environment(params=params)
    test_env = rl_utils.get_environment(params=params)
    print_environment_info(train_env, params)
    train_main(params, train_env, test_env)

