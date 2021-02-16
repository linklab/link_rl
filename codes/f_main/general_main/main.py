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
from termcolor import colored

from codes.a_config.f_trade_parameters.parameters_trade_dqn import PARAMETERS_GENERAL_TRADE_DQN
from codes.b_environments.trade.trade_action_selector import EpsilonGreedyTradeDQNActionSelector, \
    ArgmaxTradeActionSelector
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils.rl_utils import get_environment_input_output_info
from codes.e_utils.actions import EpsilonTracker

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model, print_environment_info, remove_models, agent_model_test, \
    print_performance, print_agent_info
from codes.e_utils.experience_tracker import RewardTracker, EarlyStopping
from codes.e_utils.logger import get_logger
from codes.e_utils.names import DeepLearningModelName, RLAlgorithmName, EnvironmentName, ModelSaveMode, AgentMode
from codes.e_utils.experience import ExperienceSourceFirstLast

WANDB_DIR = os.path.join(PROJECT_HOME, "out", "wandb")
if not os.path.exists(WANDB_DIR):
    os.makedirs(WANDB_DIR)

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device, "!!!!")

my_logger = get_logger("main")


if isinstance(params, PARAMETERS_GENERAL_TRADE_DQN):
    model_save_file_prefix = "_".join([params.ENVIRONMENT_ID.value, params.COIN_NAME, params.TIME_UNIT])
else:
    model_save_file_prefix = params.ENVIRONMENT_ID.value


def play_func(exp_queue, agent):
    if params.ENVIRONMENT_ID in [EnvironmentName.TRADE_V0]:
        assert params.NUM_ENVIRONMENTS == 1
        train_env = rl_utils.get_environment(params=params)
        test_env = rl_utils.get_single_environment(params=params, mode=AgentMode.TEST)
        agent.train_action_selector = EpsilonGreedyTradeDQNActionSelector(
            epsilon=params.EPSILON_INIT, env=train_env.envs[0]
        )
        agent.epsilon_tracker = EpsilonTracker(
            action_selector=agent.train_action_selector,
            eps_start=params.EPSILON_INIT,
            eps_final=params.EPSILON_MIN,
            eps_frames=params.EPSILON_MIN_STEP
        )
        agent.test_and_play_action_selector = ArgmaxTradeActionSelector(env=test_env)
    else:
        train_env = rl_utils.get_environment(params=params)
        print_environment_info(train_env, params)
        if params.MODEL_SAVE_MODE == ModelSaveMode.TEST:
            test_env = rl_utils.get_single_environment(params=params, mode=AgentMode.TEST)
        else:
            test_env = None

    # if params.DEEP_LEARNING_MODEL in [
    #     DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU,
    #     DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU_ATTENTION
    # ]:
    #     step_length = params.RNN_STEP_LENGTH
    # else:
    #     step_length = -1

    experience_source = ExperienceSourceFirstLast(
        env=train_env, agent=agent, gamma=params.GAMMA, n_step=params.N_STEP
    )

    exp_source_iter = iter(experience_source)
    step_idx = 0

    early_stopping = EarlyStopping(
        patience=params.STOP_PATIENCE_COUNT,
        evaluation_min_threshold=params.STOP_MEAN_EPISODE_REWARD,
        verbose=True,
        delta=0.0,
        model_save_dir=MODEL_SAVE_DIR,
        model_save_file_prefix=model_save_file_prefix,
        agent=agent,
        params=params
    )

    if hasattr(agent.train_action_selector, 'epsilon') and hasattr(params, "EPSILON_MIN_STEP"):
        early_stopping.evaluation_min_step_idx = params.EPSILON_MIN_STEP

    episode = 0
    solved = False

    test_mean_episode_reward = None
    train_episode_reward_lst = []

    if params.MODEL_SAVE_MODE == ModelSaveMode.TRAIN:
        num_tests = params.EARLY_STOPPING_TEST_EPISODE_PERIOD
    elif params.MODEL_SAVE_MODE == ModelSaveMode.TEST:
        num_tests = params.TEST_NUM_EPISODES
    else:
        num_tests = 0

    with RewardTracker(params=params) as reward_tracker:
        try:
            while step_idx < params.MAX_GLOBAL_STEP:
                # 1 스텝 진행하고 exp를 exp_queue에 넣음
                step_idx += 1
                exp = next(exp_source_iter)
                exp_queue.put(exp)

                if hasattr(agent, "epsilon_tracker") and agent.epsilon_tracker:
                    agent.epsilon_tracker.udpate(step_idx)

                episode_rewards, episode_steps = experience_source.pop_episode_reward_and_done_step_lst()

                if episode_rewards and episode_steps:
                    for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                        episode += 1
                        train_episode_reward_lst.append(current_episode_reward)

                        epsilon = agent.train_action_selector.epsilon if hasattr(agent.train_action_selector, 'epsilon') else None

                        train_mean_episode_reward, speed, elapsed_time = reward_tracker.set_episode_reward(
                            episode_reward=current_episode_reward, episode_done_step=step_idx
                        )

                        if episode % params.EARLY_STOPPING_TEST_EPISODE_PERIOD == 0:
                            if params.MODEL_SAVE_MODE in [ModelSaveMode.TRAIN, ModelSaveMode.TEST]:
                                if params.MODEL_SAVE_MODE == ModelSaveMode.TRAIN:
                                    test_mean_episode_reward = np.mean(train_episode_reward_lst)
                                    test_std = np.std(train_episode_reward_lst)
                                    train_episode_reward_lst.clear()
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
                                train_mean_episode_reward,
                            "test mean_{0} episode reward".format(num_tests): test_mean_episode_reward,
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

                        exp_queue.put(train_info_dict)

                if solved:
                    print("Solved in {0} steps and {1} episodes!".format(step_idx, episode))
                    break

            if params.MODEL_SAVE_MODE == ModelSaveMode.FINAL_ONLY:
                remove_models(
                    MODEL_SAVE_DIR, model_save_file_prefix, agent
                )
                save_model(
                    MODEL_SAVE_DIR, model_save_file_prefix, agent, step_idx, train_mean_episode_reward
                )
        finally:
            if params.ENVIRONMENT_ID in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
                train_env.stop()

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"

    env = rl_utils.get_single_environment(params=params)
    input_shape, num_outputs, action_min, action_max = get_environment_input_output_info(env)
    agent = rl_utils.get_rl_agent(
        input_shape, num_outputs, action_min, action_max, worker_id=0, params=params, device=device
    )
    print_agent_info(agent, params)

    agent.model.share_memory()

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2) #params.TRAIN_STEP_FREQ * 2
    play_proc = mp.Process(target=play_func, args=(exp_queue, agent))
    play_proc.start()

    time.sleep(0.5)

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

        wandb.watch(agent.model.base, log="all")

    loss_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    actor_objective_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    step_idx = 0

    while play_proc.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if isinstance(exp, dict):
                train_info_dict = exp

                mean_loss = np.mean(loss_dequeue) if len(loss_dequeue) > 0 else 0.0
                mean_actor_objective = np.mean(actor_objective_dequeue) if len(actor_objective_dequeue) > 0 else 0.0

                print_performance(
                    params=params,
                    episode_done_step=train_info_dict["step_idx"],
                    done_episode=train_info_dict["episode"],
                    episode_reward=train_info_dict["train episode reward"],
                    mean_episode_reward=train_info_dict[
                        "train mean_{0} episode reward".format(params.AVG_EPISODE_SIZE_FOR_STAT)
                    ],
                    epsilon=train_info_dict["epsilon"] if "epsilon" in train_info_dict else None,
                    elapsed_time=train_info_dict["elapsed_time"],
                    last_info=train_info_dict["last_info"],
                    speed=train_info_dict["speed"],
                    mean_loss=mean_loss,
                    mean_actor_objective=mean_actor_objective,
                    last_action=train_info_dict["last_actions"]
                )

                if params.WANDB:
                    del train_info_dict["last_info"]
                    del train_info_dict["elapsed_time"]
                    train_info_dict["train mean (critic) loss"] = mean_loss
                    train_info_dict["train mean actor objective"] = mean_actor_objective

                    wandb.log(train_info_dict)

                continue

            if exp is None:
                play_proc.join()
                break
            agent.buffer._add(exp)

        if isinstance(agent, OnPolicyAgent):
            if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0, RLAlgorithmName.DISCRETE_PPO_V0]:
                if len(agent.buffer) < params.PPO_TRAJECTORY_SIZE:
                    continue
            else:
                if len(agent.buffer) < params.BATCH_SIZE:
                    continue
            _, last_loss, actor_objective = agent.train(step_idx=step_idx)
            loss_dequeue.append(last_loss)
            if actor_objective:
                actor_objective_dequeue.append(actor_objective)
            # On-policy는 현재의 정책을 통해 산출된 경험정보만을 활용하여 NN을 업데이트해야 함.
            # 따라서, 현재 학습에 사용된 Buffer는 깨끗하게 지워야 함.
            agent.buffer.clear()
        elif isinstance(agent, OffPolicyAgent):
            if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                continue
            _, last_loss, actor_objective = agent.train(step_idx=step_idx)
            loss_dequeue.append(last_loss)
            if actor_objective:
                actor_objective_dequeue.append(actor_objective)
        else:
            raise ValueError()

        if hasattr(params, "PER_RANK_BASED") and getattr(params, "PER_RANK_BASED"):
            if step_idx % 100 < params.TRAIN_STEP_FREQ:
                agent.buffer.rebalance()


if __name__ == "__main__":
    main()