import copy
import os, sys
import threading
from collections import deque
from multiprocessing import Pipe
from sys import platform as _platform
import torch.multiprocessing as mp

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.f_main.general_main.main_utils import *
from codes.b_environments.trade.trade_action_selector import EpsilonGreedyTradeDQNActionSelector, \
    ArgmaxTradeActionSelector
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.d_agents.on_policy.ppo.ppo_agent import AgentPPO

from codes.e_utils.common_utils import print_params
from codes.e_utils.experience import ExperienceSourceFirstLast
from codes.e_utils.names import OFF_POLICY_RL_ALGORITHMS, RLAlgorithmName, ON_POLICY_RL_ALGORITHMS
from codes.e_utils.train_tracker import SpeedTracker


if "win32" in _platform or "win64" in _platform:
    thread = True
    print("PLATFORM: WINDOWS")
else:
    thread = False
    print("PLATFORM: MAC or LINUX")


def actor_func(agent, current_model_version, exp_queue, child_pipe_conn):
    train_env, test_env = get_train_and_test_envs()

    if params.ENVIRONMENT_ID in [EnvironmentName.TRADE_V0]:
        assert params.NUM_ENVIRONMENTS == 1
        agent.train_action_selector = EpsilonGreedyTradeDQNActionSelector(
            epsilon=params.EPSILON_INIT, env=train_env.envs[0]
        )
        agent.test_and_play_action_selector = ArgmaxTradeActionSelector(env=test_env)

    experience_source = ExperienceSourceFirstLast(
        env=train_env, agent=agent, gamma=params.GAMMA, n_step=params.N_STEP
    )

    exp_source_iter = iter(experience_source)
    step_idx = 0

    episode = 0
    solved = False
    is_good_model_saved = False

    train_episode_reward_lst_for_stat = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    train_episode_reward_lst_for_test = deque(maxlen=params.TEST_NUM_EPISODES)

    episode_processor = EpisodeProcessor(test_env=test_env, agent=agent, params=params)

    with SpeedTracker(params=params) as speed_tracker:
        try:
            while step_idx < params.MAX_GLOBAL_STEP:
                # 1 ?????? ???????????? exp??? exp_queue??? ??????
                step_idx += 1

                if isinstance(agent, OnPolicyAgent):
                    agent.model_version = current_model_version.value

                if params.ENVIRONMENT_ID in [
                    EnvironmentName.REAL_DEVICE_DOUBLE_RIP,
                    EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0
                ] and hasattr(agent, "train_action_selector") and hasattr(agent.train_action_selector, "noise_std"):
                    if train_env.envs[0].is_upright:
                        agent.train_action_selector.noise_std = 0.25
                    else:
                        agent.train_action_selector.noise_std = 1.0

                exp = next(exp_source_iter)

                if thread:
                    exp_queue.put(exp)
                else:
                    child_pipe_conn.send(exp)

                if hasattr(agent, "epsilon_tracker") and agent.epsilon_tracker:
                    agent.epsilon_tracker.udpate(step_idx)

                episode_rewards, episode_steps = experience_source.pop_episode_reward_and_done_step_lst()

                if episode_rewards and episode_steps:
                    for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                        train_info_dict = episode_processor.process(
                            train_episode_reward_lst_for_test,
                            train_episode_reward_lst_for_stat,
                            current_episode_reward,
                            speed_tracker,
                            step_idx,
                            episode,
                            current_episode_step,
                            exp
                        )

                        if episode != 0 and episode % params.TEST_PERIOD_EPISODES == 0:
                            solved, good_model_saved, evaluation_msg = episode_processor.test(step_idx)

                            if good_model_saved:
                                is_good_model_saved = True

                            train_info_dict["evaluation_msg"] = evaluation_msg
                            train_info_dict["solved"] = solved
                        else:
                            train_info_dict["evaluation_msg"] = None
                            train_info_dict["solved"] = None

                        if params.ENVIRONMENT_ID in [
                            EnvironmentName.PENDULUM_MATLAB_V0,
                            EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0,
                            EnvironmentName.REAL_DEVICE_RIP,
                            EnvironmentName.REAL_DEVICE_DOUBLE_RIP,
                            # EnvironmentName.QUANSER_SERVO_2
                        ]:
                            train_info_dict["last_done_reason"] = train_env.envs[0].last_done_reason.value
                        else:
                            train_info_dict["last_done_reason"] = None

                        if thread:
                            exp_queue.put(train_info_dict)
                        else:
                            child_pipe_conn.send(train_info_dict)

                        episode += 1

                if solved:
                    print("Solved in {0} steps and {1} episodes!".format(step_idx, episode))
                    break

            if not is_good_model_saved:
                agent.test_model = copy.deepcopy(agent.model)
                last_model_save(agent, step_idx, train_episode_reward_lst_for_stat)
        finally:
            if params.ENVIRONMENT_ID in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
                print("send 'stop' message into train_env!")
                train_env.stop()

    if thread:
        exp_queue.put(None)
    else:
        child_pipe_conn.send(None)


def main():
    mp.set_start_method('spawn', force=True)
    os.environ['OMP_NUM_THREADS'] = "10"

    if params.ENVIRONMENT_ID in [
        EnvironmentName.PENDULUM_MATLAB_V0,
        EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0,
        EnvironmentName.REAL_DEVICE_RIP,
        EnvironmentName.REAL_DEVICE_DOUBLE_RIP,
        EnvironmentName.QUANSER_SERVO_2
    ]:
        tentative_env = None
    else:
        tentative_env = rl_utils.get_single_environment(params=params)

    agent = get_agent(tentative_env)
    current_model_version = mp.Value("i", 0)

    if thread:
        from queue import Queue
        exp_queue = Queue(maxsize=params.TRAIN_STEP_FREQ * 100) #params.TRAIN_STEP_FREQ * 2
        parent_pipe_conn = None
        actor = threading.Thread(target=actor_func, args=(agent, current_model_version, exp_queue, None))
    else:
        agent.model.share_memory()
        parent_pipe_conn, child_pipe_conn = Pipe()
        exp_queue = None
        actor = mp.Process(target=actor_func, args=(agent, current_model_version, None, child_pipe_conn))

    actor.start()
    time.sleep(0.5)

    if params.WANDB:
        set_wandb(agent)

    loss_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    actor_objective_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    step_idx = 0
    episode = 0

    last_mean_loss = None
    last_mean_actor_objective = None

    while actor.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        solved = False
        for _ in range(params.TRAIN_STEP_FREQ):
            if thread:
                exp = exp_queue.get()
            else:
                #print("!!!!!!! ", step_idx)
                exp = parent_pipe_conn.recv()

            if isinstance(exp, dict):
                train_info_dict = exp

                mean_loss = np.mean(loss_dequeue) if len(loss_dequeue) > 0 else 0.0
                mean_actor_objective = np.mean(actor_objective_dequeue) \
                    if len(actor_objective_dequeue) > 0 else 0.0

                print_performance(
                    params=params,
                    episode_done_step=train_info_dict["step_idx"],
                    done_episode=train_info_dict["episode"],
                    episode_reward=train_info_dict["### EVERY TRAIN EPISODE REWARDS ###"],
                    mean_episode_reward=train_info_dict[
                        "train mean ({0} episode rewards)".format(params.AVG_EPISODE_SIZE_FOR_STAT)
                    ],
                    epsilon=train_info_dict["epsilon"] if "epsilon" in train_info_dict else None,
                    elapsed_time=train_info_dict["elapsed_time"],
                    last_info=train_info_dict["last_info"],
                    speed=train_info_dict["speed"],
                    mean_loss=mean_loss,
                    mean_actor_objective=mean_actor_objective,
                    last_action=train_info_dict["last_actions"],
                    evaluation_msg=train_info_dict["evaluation_msg"],
                    last_done_reason=train_info_dict["last_done_reason"]
                )

                episode += 1

                if train_info_dict["solved"]:
                    solved = True

                if params.WANDB:
                    if last_mean_loss is None and last_mean_actor_objective is None:
                        last_mean_loss = mean_loss
                        last_mean_actor_objective = mean_actor_objective
                    else:
                        if abs(mean_loss - last_mean_loss) > 10.0:
                            mean_loss = last_mean_loss

                        if abs(mean_actor_objective - last_mean_actor_objective) > 10.0:
                            mean_loss = last_mean_loss

                        last_mean_loss = mean_loss
                        last_mean_actor_objective = mean_actor_objective

                    train_info_dict["train mean (critic) loss"] = mean_loss
                    train_info_dict["train mean actor objective"] = mean_actor_objective

                    if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_SAC_V0, RLAlgorithmName.DISCRETE_SAC_V0]:
                        train_info_dict["alpha"] = agent.alpha.item()

                    del train_info_dict["evaluation_msg"]
                    del train_info_dict["solved"]
                    del train_info_dict["last_done_reason"]
                    del train_info_dict["episode"]
                    del train_info_dict["elapsed_time"]

                    if params.RL_ALGORITHM in [
                        RLAlgorithmName.DDPG_V0,
                        RLAlgorithmName.TD3_V0,
                        RLAlgorithmName.CONTINUOUS_SAC_V0,
                        RLAlgorithmName.CONTINUOUS_PPO_V0,
                        RLAlgorithmName.CONTINUOUS_A2C_V0
                    ] and params.ENVIRONMENT_ID in [
                        EnvironmentName.QUANSER_SERVO_2,
                        EnvironmentName.REAL_DEVICE_RIP,
                        EnvironmentName.REAL_DEVICE_DOUBLE_RIP,
                        EnvironmentName.PENDULUM_MATLAB_V0,
                        EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0
                    ]:
                        if train_info_dict["last_actions"] < -1.0 or train_info_dict["last_actions"] > 1.0:
                            train_info_dict["last_actions"] = 0.0

                    wandb.log(train_info_dict)
            else:
                if exp is None:
                    solved = True
                    actor.join()
                    break
                else:
                    agent.buffer.add_sample(exp)

        if solved:
            print("Solved in {0} steps and {1} episodes!".format(step_idx, episode))
            break
        else:
            train(agent, step_idx, loss_dequeue, actor_objective_dequeue, current_model_version)


def train(agent, step_idx, loss_dequeue, actor_objective_dequeue, current_model_version):
    if params.RL_ALGORITHM in ON_POLICY_RL_ALGORITHMS:
        if isinstance(agent, AgentPPO):
            if len(agent.buffer) < params.PPO_TRAJECTORY_SIZE:
                return
        else:
            if len(agent.buffer) < params.BATCH_SIZE:
                return

        _, last_loss, actor_objective = agent.train_on_policy(
            step_idx=step_idx, current_model_version=current_model_version
        )

        # On-policy??? ????????? ????????? ?????? ????????? ?????????????????? ???????????? NN??? ?????????????????? ???.
        # ?????????, ?????? ????????? ????????? Buffer??? ???????????? ????????? ???.
        agent.buffer.clear()
    else:
        if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
            return

        _, last_loss, actor_objective = agent.train_off_policy(step_idx=step_idx)

        if hasattr(params, "PER_RANK_BASED") and getattr(params, "PER_RANK_BASED"):
            if step_idx % 100 < params.TRAIN_STEP_FREQ:
                agent.buffer.rebalance()

    loss_dequeue.append(last_loss)

    if actor_objective:
        actor_objective_dequeue.append(actor_objective)


if __name__ == "__main__":
    advance_check()
    print_params(params)

    #assert params.RL_ALGORITHM in OFF_POLICY_RL_ALGORITHMS

    main()
