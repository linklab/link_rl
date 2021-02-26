# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
from collections import deque

import torch
import os, sys
import numpy as np
import wandb
from termcolor import colored

from codes.a_config.f_trade_parameters.parameters_trade_dqn import PARAMETERS_GENERAL_TRADE_DQN
from codes.b_environments.trade.trade_action_selector import EpsilonGreedyTradeDQNActionSelector, \
    ArgmaxTradeActionSelector
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils.rl_utils import get_environment_input_output_info, MODEL_SAVE_FILE_PREFIX, MODEL_ZOO_SAVE_DIR
from codes.e_utils.actions import EpsilonTracker

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model, print_environment_info, remove_models, agent_model_test, \
    print_agent_info, load_model
from codes.e_utils.train_tracker import SpeedTracker, EarlyStopping
from codes.e_utils.logger import get_logger
from codes.e_utils.names import DeepLearningModelName, RLAlgorithmName, EnvironmentName, ModelSaveMode, AgentMode
from codes.e_utils.experience import ExperienceSourceFirstLast
from codes.e_utils.names import OFF_POLICY_RL_ALGORITHMS, ON_POLICY_RL_ALGORITHMS

WANDB_DIR = os.path.join(PROJECT_HOME, "out", "wandb")
if not os.path.exists(WANDB_DIR):
    os.makedirs(WANDB_DIR)

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("DEVICE: {0}".format(device))

my_logger = get_logger("main")


def get_agent(env):
    input_shape, num_outputs, action_min, action_max = get_environment_input_output_info(env)

    agent = rl_utils.get_rl_agent(
        input_shape, num_outputs, action_min, action_max, worker_id=0, params=params, device=device
    )

    load_model(MODEL_ZOO_SAVE_DIR, MODEL_SAVE_FILE_PREFIX, agent, inquery=True)
    print_agent_info(agent, params)

    return agent


def set_wandb(agent):
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
    wandb.watch(agent.model.base, log="all")


def get_train_and_test_envs():
    train_env = rl_utils.get_environment(params=params)
    print_environment_info(train_env, params)
    if params.MODEL_SAVE_MODE == ModelSaveMode.TEST:
        test_env = rl_utils.get_single_environment(params=params, mode=AgentMode.TEST)
    else:
        test_env = None

    return train_env, test_env


def get_early_stopping(agent):
    early_stopping = EarlyStopping(
        patience=params.STOP_PATIENCE_COUNT,
        evaluation_min_threshold=params.STOP_MEAN_EPISODE_REWARD,
        verbose=True,
        delta=0.0,
        model_save_dir=MODEL_SAVE_DIR,
        model_save_file_prefix=params.ENVIRONMENT_ID.value,
        agent=agent,
        params=params
    )

    if hasattr(agent.train_action_selector, 'epsilon') and hasattr(params, "EPSILON_MIN_STEP"):
        early_stopping.evaluation_min_step_idx = params.EPSILON_MIN_STEP

    return early_stopping


def get_num_tests():
    if params.MODEL_SAVE_MODE == ModelSaveMode.TRAIN:
        num_tests = params.EARLY_STOPPING_TEST_EPISODE_PERIOD
    elif params.MODEL_SAVE_MODE == ModelSaveMode.TEST:
        num_tests = params.TEST_NUM_EPISODES
    else:
        num_tests = 0
    return num_tests


def process_episode(
        train_episode_reward_lst_for_test,
        train_episode_reward_lst_for_stat,
        current_episode_reward,
        agent,
        reward_tracker,
        step_idx,
        episode,
        test_env,
        num_tests,
        early_stopping,
        current_episode_step,
        exp
):
    train_episode_reward_lst_for_test.append(current_episode_reward)
    train_episode_reward_lst_for_stat.append(current_episode_reward)

    epsilon = agent.train_action_selector.epsilon if hasattr(agent.train_action_selector, 'epsilon') else None

    speed, elapsed_time = reward_tracker.set_episode_reward(
        episode_done_step=step_idx
    )

    solved = False
    test_mean_episode_reward = False

    if episode % params.EARLY_STOPPING_TEST_EPISODE_PERIOD == 0:
        if params.MODEL_SAVE_MODE in [ModelSaveMode.TRAIN, ModelSaveMode.TEST]:
            if params.MODEL_SAVE_MODE == ModelSaveMode.TRAIN:
                test_mean_episode_reward = np.mean(train_episode_reward_lst_for_test).item()
                test_std = np.std(train_episode_reward_lst_for_test).item()
                train_episode_reward_lst_for_test.clear()
                test_env_str = colored("TRAIN ENV", "yellow")
            else:
                test_mean_episode_reward, test_std = agent_model_test(params, test_env, agent)
                test_env_str = colored("TEST ENV", "yellow")

            mean_std_str = colored(
                "{0:7.2f}\u00B1{1:.2f}".format(test_mean_episode_reward, test_std), "yellow"
            )

            print("* MODEL SAVE & TRAIN STOP TEST for {0} *, EPISODE REWARD ({1} EPISODES): {2}".format(
                test_env_str, num_tests, mean_std_str
            ), end="")

            solved = early_stopping.evaluate(
                evaluation_value=test_mean_episode_reward,
                episode_done_step=step_idx
            )
        elif params.MODEL_SAVE_MODE == ModelSaveMode.FINAL_ONLY:
            test_mean_episode_reward = None
            solved = False
        else:
            raise ValueError()

    train_info_dict = {
        "train episode reward": current_episode_reward,
        "train mean_{0} episode reward".format(params.AVG_EPISODE_SIZE_FOR_STAT):
            np.mean(train_episode_reward_lst_for_stat),
        'test mean_{0} episode reward'.format(num_tests): test_mean_episode_reward,
        "steps/episode": current_episode_step,
        "speed": speed,
        "step_idx": step_idx,
        "episode": episode,
        'last_actions': exp.action,
        "elapsed_time": elapsed_time,
        "last_info": exp.info
    }
    if epsilon:
        train_info_dict["epsilon"] = epsilon

    return solved, train_info_dict


def last_model_save(agent, step_idx, train_episode_reward_lst_for_stat):
    remove_models(
        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent
    )
    save_model(
        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, np.mean(train_episode_reward_lst_for_stat)
    )


def agent_train(agent, step_idx, loss_dequeue, actor_objective_dequeue):
    is_loop_continue = False
    if isinstance(agent, OnPolicyAgent):
        if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0, RLAlgorithmName.DISCRETE_PPO_V0]:
            if len(agent.buffer) < params.PPO_TRAJECTORY_SIZE:
                return
        else:
            if len(agent.buffer) < params.BATCH_SIZE:
                return

        _, last_loss, actor_objective = agent.train(step_idx=step_idx)
        loss_dequeue.append(last_loss)
        if actor_objective:
            actor_objective_dequeue.append(actor_objective)
        # On-policy는 현재의 정책을 통해 산출된 경험정보만을 활용하여 NN을 업데이트해야 함.
        # 따라서, 현재 학습에 사용된 Buffer는 깨끗하게 지워야 함.
        agent.buffer.clear()

    elif isinstance(agent, OffPolicyAgent):
        if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
            return

        _, last_loss, actor_objective = agent.train(step_idx=step_idx)
        loss_dequeue.append(last_loss)
        if actor_objective:
            actor_objective_dequeue.append(actor_objective)
    else:
        raise ValueError()

    if hasattr(params, "PER_RANK_BASED") and getattr(params, "PER_RANK_BASED"):
        if step_idx % 100 < params.TRAIN_STEP_FREQ:
            agent.buffer.rebalance()

    return is_loop_continue


def print_performance(params, episode_done_step, done_episode, episode_reward, mean_episode_reward, epsilon,
                      elapsed_time, last_info, speed, mean_loss, mean_actor_objective, worker_id=None, last_action=None):

    if worker_id is not None:
        prefix = "[Worker ID: {0}]".format(worker_id)
    else:
        prefix = ""

    if isinstance(epsilon, tuple) or isinstance(epsilon, list):
        epsilon_str = " eps.: {0:5.3f}, {1:5.3f},".format(
            epsilon[0] if epsilon[0] else 0.0,
            epsilon[1] if epsilon[1] else 0.0
        )
    elif isinstance(epsilon, float):
        epsilon_str = " eps.: {0:5.3f},".format(
            epsilon if epsilon else 0.0,
        )
    else:
        epsilon_str = ""

    mean_episode_reward_str = "{0:9.3f}".format(mean_episode_reward)

    if isinstance(episode_reward, np.ndarray):
        episode_reward = episode_reward[0]

    print(
        "{0}[{1:6}/{2}] Ep. {3}, EPISODE REWARD: {4:9.3f}, MEAN_{5} EPSIODE REWARD: {6},{7} SPEED: {8:7.2f}steps/sec., {9}".format(
            prefix,
            episode_done_step,
            params.MAX_GLOBAL_STEP,
            done_episode,
            episode_reward,
            params.AVG_EPISODE_SIZE_FOR_STAT,
            mean_episode_reward_str,
            epsilon_str,
            speed,
            time.strftime("%Hh %Mm %Ss", time.gmtime(elapsed_time)),
    ), end="")

    if last_info and "action_count" in last_info:
        print(", {0}".format(last_info["action_count"]), end="")

    if mean_loss is not None:
        print(", mean (critic) loss {0:7.4f}".format(mean_loss), end="")

    if mean_actor_objective is not None:
        print(", mean actor obj. {0:7.4f}".format(mean_actor_objective), end="")

    if params.ENVIRONMENT_ID == EnvironmentName.TRADE_V0:
        print(", profit {0:8.1f}".format(last_info['profit']), end="")

    if last_action is not None:
        print(", last action {0}".format(last_action), end="")

    print("", flush=True)