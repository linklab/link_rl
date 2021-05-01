import multiprocessing
import os, sys
from collections import deque
import threading
from sys import platform as _platform
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)
os.environ['OMP_NUM_THREADS'] = "1"

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.f_main.general_main.a_common_main import *
from codes.b_environments.trade.trade_action_selector import EpsilonGreedyTradeDQNActionSelector, \
    ArgmaxTradeActionSelector
from codes.e_utils.common_utils import print_params
from codes.e_utils.experience import ExperienceSourceFirstLast
from codes.e_utils.names import OFF_POLICY_RL_ALGORITHMS
from codes.e_utils.train_tracker import SpeedTracker

if "win32" in _platform or "win64" in _platform:
    parent = threading.Thread
    print("PLATFORM: WINDOWS")
else:
    parent = mp.Process
    print("PLATFORM: MAC or LINUX")


class Actor(parent):
    def __init__(self, agent, exp_queue):
        super(Actor, self).__init__(name="Actor")
        self.agent = agent
        self.exp_queue = exp_queue

        self.train_env, self.test_env = get_train_and_test_envs()

        if params.ENVIRONMENT_ID in [EnvironmentName.TRADE_V0]:
            assert params.NUM_ENVIRONMENTS == 1
            agent.train_action_selector = EpsilonGreedyTradeDQNActionSelector(
                epsilon=params.EPSILON_INIT, env=self.train_env.envs[0]
            )
            from codes.d_agents.actions import EpsilonTracker
            agent.epsilon_tracker = EpsilonTracker(
                action_selector=agent.train_action_selector,
                eps_start=params.EPSILON_INIT,
                eps_final=params.EPSILON_MIN,
                eps_frames=params.EPSILON_MIN_STEP
            )
            agent.test_and_play_action_selector = ArgmaxTradeActionSelector(env=self.test_env)

        # if params.DEEP_LEARNING_MODEL in [
        #     DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU,
        #     DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU_ATTENTION
        # ]:
        #     step_length = params.RNN_STEP_LENGTH
        # else:
        #     step_length = -1

        self.experience_source = ExperienceSourceFirstLast(
            env=self.train_env, agent=self.agent, gamma=params.GAMMA, n_step=params.N_STEP
        )

        self.step_idx = 0

        self.early_stopping = get_early_stopping(agent)

        self.episode = 0
        self.solved = False

        self.train_episode_reward_lst_for_stat = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
        self.train_episode_reward_lst_for_test = deque(maxlen=params.TEST_NUM_EPISODES)

        self.num_tests = get_num_tests()

    def run(self):
        exp_source_iter = iter(self.experience_source)

        with SpeedTracker(params=params) as speed_tracker:
            try:
                while self.step_idx < params.MAX_GLOBAL_STEP:
                    # 1 스텝 진행하고 exp를 exp_queue에 넣음
                    self.step_idx += 1
                    exp = next(exp_source_iter)

                    if parent is mp.Process:
                        self.exp_queue.put(exp)
                    else:
                        self.agent.buffer._add(exp)

                    if hasattr(self.agent, "epsilon_tracker") and self.agent.epsilon_tracker:
                        self.agent.epsilon_tracker.udpate(self.step_idx)

                    episode_rewards, episode_steps = self.experience_source.pop_episode_reward_and_done_step_lst()

                    if episode_rewards and episode_steps:
                        for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                            self.episode += 1

                            self.solved, train_info_dict = process_episode(
                                self.train_episode_reward_lst_for_test,
                                self.train_episode_reward_lst_for_stat,
                                current_episode_reward,
                                self.agent,
                                speed_tracker,
                                self.step_idx,
                                self.episode,
                                self.test_env,
                                self.num_tests,
                                self.early_stopping,
                                current_episode_step,
                                exp
                            )

                            self.exp_queue.put(train_info_dict)

                    if self.solved:
                        print("Solved in {0} steps and {1} episodes!".format(self.step_idx, self.episode))
                        break

                if params.MODEL_SAVE_MODE == ModelSaveMode.FINAL_ONLY:
                    last_model_save(self.agent, self.step_idx, self.train_episode_reward_lst_for_stat)
            finally:
                if params.ENVIRONMENT_ID in [EnvironmentName.REAL_DEVICE_RIP, EnvironmentName.REAL_DEVICE_DOUBLE_RIP]:
                    print("send 'stop' message into train_env!")
                    self.train_env.stop()

            self.exp_queue.put(None)


def main():
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

    if parent is mp.Process:
        agent.model.share_memory()
        from torch.multiprocessing import Queue
    else:
        from queue import Queue

    exp_queue = Queue(maxsize=params.TRAIN_STEP_FREQ * 100) #params.TRAIN_STEP_FREQ * 2

    actor = Actor(agent, exp_queue)
    actor.start()

    if params.WANDB:
        set_wandb(agent)

    loss_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    actor_objective_dequeue = deque(maxlen=params.AVG_STEP_SIZE_FOR_TRAIN_LOSS)
    step_idx = 0
    episode = 0

    while actor.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        solved = False
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if isinstance(exp, dict):
                episode += 1
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
                    evaluation_msg=train_info_dict["evaluation_msg"]
                )

                if train_info_dict["solved"]:
                    solved = True

                if params.WANDB:
                    train_info_dict["train mean (critic) loss"] = mean_loss
                    train_info_dict["train mean actor objective"] = mean_actor_objective
                    del train_info_dict["evaluation_msg"]
                    del train_info_dict["solved"]
                    wandb.log(train_info_dict)
            else:
                if exp is None:
                    solved = True
                    actor.join()
                    break
                else:
                    if parent is multiprocessing.Process:
                        agent.buffer._add(exp)

        if solved:
            print("Solved in {0} steps and {1} episodes!".format(step_idx, episode))
            break
        else:
            if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                continue

            _, critic_loss, actor_objective = agent.train(step_idx=step_idx)

            loss_dequeue.append(critic_loss)

            if actor_objective:
                actor_objective_dequeue.append(actor_objective)

            if hasattr(params, "PER_RANK_BASED") and getattr(params, "PER_RANK_BASED"):
                if step_idx % 100 < params.TRAIN_STEP_FREQ:
                    agent.buffer.rebalance()


if __name__ == "__main__":
    advance_check()
    print_params(params)

    assert params.RL_ALGORITHM in OFF_POLICY_RL_ALGORITHMS

    main()

