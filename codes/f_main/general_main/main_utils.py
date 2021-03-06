# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
import glob
import torch
import os, sys
import numpy as np
import wandb
from termcolor import colored

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params
if params.VERBOSE_TO_LOG:
    files = glob.glob(os.path.join(PROJECT_HOME, "out", "logs", "*.log*"))
    for f in files:
        os.remove(f)

from codes.b_environments.quanser_rotary_inverted_pendulum.quanser_rip import get_quanser_rip_observation_space, \
    get_quanser_rip_action_info
from codes.b_environments.rotary_inverted_pendulum.rip import get_rip_observation_space, get_rip_action_info
from codes.c_models.continuous_action.continuous_action_model import ContinuousActionModel
from codes.e_utils.reward_changer import RewardChanger
from codes.e_utils.rl_utils import get_environment_input_output_info, MODEL_ZOO_SAVE_DIR, MODEL_SAVE_FILE_PREFIX
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model, print_environment_info, remove_models, \
    print_agent_info, load_model, map_range
from codes.e_utils.train_tracker import EarlyStopping
from codes.e_utils.logger import get_logger
from codes.e_utils.names import EnvironmentName, AgentMode, RLAlgorithmName

WANDB_DIR = os.path.join(PROJECT_HOME, "out", "wandb")
if not os.path.exists(WANDB_DIR):
    os.makedirs(WANDB_DIR)

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

print("DEVICE: {0}".format(device))

my_logger = get_logger("main")


def get_agent(env):
    if env is None:
        if params.ENVIRONMENT_ID in [
            EnvironmentName.PENDULUM_MATLAB_V0,
            EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0,
            EnvironmentName.REAL_DEVICE_RIP,
            EnvironmentName.REAL_DEVICE_DOUBLE_RIP,
        ]:
            observation_space, _ = get_rip_observation_space(params.ENVIRONMENT_ID, params)
            action_space, num_outputs, action_n, action_min, action_max, _ = get_rip_action_info(
                params, pendulum_type=params.ENVIRONMENT_ID
            )

            observation_shape = observation_space.shape
            action_shape = action_space.shape
        elif params.ENVIRONMENT_ID == EnvironmentName.QUANSER_SERVO_2:
            observation_space, _ = get_quanser_rip_observation_space()
            action_space, num_outputs, action_n, action_min, action_max, _ = get_quanser_rip_action_info(params)

            observation_shape = observation_space.shape
            action_shape = action_space.shape
        else:
            raise ValueError()
    else:
        observation_shape, action_shape, num_outputs, action_n, action_min, action_max = get_environment_input_output_info(env)

    agent = rl_utils.get_rl_agent(
        observation_shape, action_shape, num_outputs, action_n=action_n,
        action_min=action_min, action_max=action_max, worker_id=0, params=params, device=device
    )

    load_model(MODEL_ZOO_SAVE_DIR, MODEL_SAVE_FILE_PREFIX, agent, inquery=True)
    print_agent_info(agent, params)

    return agent


def set_wandb(agent):
    configuration = {key: getattr(params, key) for key in dir(params) if not key.startswith("__")}
    wandb.init(
        project=params.WANDB_PROJECT,
        entity=params.WANDB_ENTITY,
        dir=WANDB_DIR,
        config=configuration
    )

    run_name = wandb.run.name
    run_number = run_name.split("-")[-1]
    wandb.run.name = "{0}_{1}_{2}_{3}_{4}".format(
        run_number, params.ENVIRONMENT_ID.value, agent.__name__, agent.model.__name__,
        params.MY_PLATFORM if params.MY_PLATFORM is not None else "COMMON"
    )
    wandb.run.save()
    wandb.watch(agent.model.base, log="all")


def get_train_and_test_envs():
    train_env = rl_utils.get_environment(params=params)
    print_environment_info(train_env, params)

    if params.ENVIRONMENT_ID in [
        EnvironmentName.PENDULUM_MATLAB_V0,
        EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0,
        EnvironmentName.REAL_DEVICE_RIP,
        EnvironmentName.REAL_DEVICE_DOUBLE_RIP,
        EnvironmentName.QUANSER_SERVO_2
    ]:
        test_env = train_env.envs[0]
    else:
        test_env = rl_utils.get_single_environment(params=params, mode=AgentMode.TEST)

    return train_env, test_env


class EpisodeProcessor:
    def __init__(self, test_env, agent, params):
        self.test_env = test_env
        self.agent = agent
        self.params = params
        self.test_mean_episode_reward = None
        self.test_std_episode_reward = None
        self.evaluation_msg = None

        self.early_stopping = EarlyStopping(
            patience=params.STOP_PATIENCE_COUNT,
            evaluation_value_min_threshold=params.TRAIN_STOP_EPISODE_REWARD,
            evaluation_std_max_threshold=params.TRAIN_STOP_EPISODE_REWARD_STD,
            delta=0.001,
            model_save_dir=MODEL_SAVE_DIR,
            model_save_file_prefix=params.ENVIRONMENT_ID.value,
            agent=agent,
            params=params
        )

        self.test_mean_episode_reward, self.test_std_episode_reward = self.agent_model_test(
            num_tests=1
        )

    def process(
            self,
            train_episode_reward_lst_for_test,
            train_episode_reward_lst_for_stat,
            current_episode_reward,
            speed_tracker,
            step_idx,
            episode,
            current_episode_step,
            exp
    ):
        train_episode_reward_lst_for_test.append(current_episode_reward)
        train_episode_reward_lst_for_stat.append(current_episode_reward)

        epsilon = self.agent.train_action_selector.epsilon \
            if hasattr(self.agent.train_action_selector, 'epsilon') else None

        speed, elapsed_time = speed_tracker.get_speed_and_elapsed_time(
            episode_done_step=step_idx
        )

        evaluation_msg = None

        train_info_dict = {
            "### EVERY TRAIN EPISODE REWARDS ###": current_episode_reward,
            "train mean ({0} episode rewards)".format(params.AVG_EPISODE_SIZE_FOR_STAT):
                np.mean(train_episode_reward_lst_for_stat),
            '*** TEST MEAN ({0} episode rewards) ***'.format(params.TEST_NUM_EPISODES): self.test_mean_episode_reward,
            '*** TEST STD ({0} episode rewards) ***'.format(params.TEST_NUM_EPISODES): self.test_std_episode_reward,
            "steps/episode": current_episode_step,
            "speed": speed,
            "step_idx": step_idx,
            "episode": episode,
            'last_actions': exp.action,
            "elapsed_time": elapsed_time,
            "last_info": exp.info,
            "evaluation_msg": evaluation_msg,
        }

        if epsilon:
            train_info_dict["epsilon"] = epsilon

        if hasattr(self.agent, "last_noise"):
            train_info_dict["last_noise"] = self.agent.last_noise

        if "global_uncertainty" in exp.info and hasattr(self.agent, "global_uncertainty"):
            self.agent.global_uncertainty = exp.info["global_uncertainty"]

        return train_info_dict

    def test(self, step_idx):
        self.test_mean_episode_reward, self.test_std_episode_reward = self.agent_model_test(
            num_tests=params.TEST_NUM_EPISODES
        )
        test_env_str = colored("TEST ENV", "yellow")

        mean_std_str = colored(
            "{0:7.2f}\u00B1{1:.2f}".format(self.test_mean_episode_reward, self.test_std_episode_reward), "yellow"
        )

        model_save_msg = "* MODEL SAVE & TRAIN STOP TEST for {0} *, EPISODE REWARD ({1} EPISODES): {2}".format(
            test_env_str, params.TEST_NUM_EPISODES, mean_std_str
        )

        solved, good_model_saved, early_stopping_evaluation_msg = self.early_stopping.evaluate(
            evaluation_value=self.test_mean_episode_reward,
            evaluation_value_std=self.test_std_episode_reward,
            episode_done_step=step_idx
        )

        evaluation_msg = model_save_msg + " ---> " + early_stopping_evaluation_msg

        return solved, good_model_saved, evaluation_msg

    def agent_model_test(self, num_tests):
        self.agent.agent_mode = AgentMode.TEST
        self.agent.model.eval()

        self.agent.test_model.sync(self.agent.model)
        self.agent.test_model.eval()

        num_step = 0

        episode_rewards = np.zeros(num_tests)

        tests_done = 0
        print("#######################################################################################################")
        for test_episode in range(num_tests):
            done = False
            episode_reward = 0

            state = self.test_env.reset()

            num_episode_step = 0

            agent_state = rl_utils.initial_agent_state()

            while not done:
                num_step += 1
                num_episode_step += 1

                state = np.expand_dims(state, axis=0)

                action, agent_state, = self.agent(state, agent_state)

                action = action[0]

                if isinstance(self.agent.model, ContinuousActionModel):
                    action = map_range(
                        np.asarray(action),
                        np.ones_like(self.agent.action_min) * -1.0, np.ones_like(self.agent.action_max),
                        self.agent.action_min, self.agent.action_max
                    )

                next_state, reward, done, info = self.test_env.step(action)

                if isinstance(self.test_env, RewardChanger):
                    reward = self.test_env.reverse_reward(reward)

                episode_reward += reward

                state = next_state

            episode_rewards[test_episode] = episode_reward
            tests_done += 1
            print("TEST {0}: EPISODE REWARD: {1:7.2f}".format(tests_done, float(np.mean(episode_reward).item())))
        self.agent.agent_mode = AgentMode.TRAIN
        self.agent.model.train()

        return np.mean(episode_rewards), np.std(episode_rewards)


def last_model_save(agent, step_idx, train_episode_reward_lst_for_stat):
    remove_models(
        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent
    )
    model_save_filename = save_model(
        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, np.mean(train_episode_reward_lst_for_stat)
    )
    print("MODEL SAVE @ LAST STEP {0} - {1}".format(step_idx, model_save_filename))


def print_performance(
        params, episode_done_step, done_episode, episode_reward, mean_episode_reward, epsilon,
        elapsed_time, last_info, speed, mean_loss, mean_actor_objective, worker_id=None, last_action=None,
        evaluation_msg=None, last_done_reason=None
):
    if worker_id is not None:
        prefix = "[Worker ID: {0}]".format(worker_id)
    else:
        prefix = ""

    if isinstance(epsilon, tuple) or isinstance(epsilon, list):
        epsilon_str = ", eps.: {0:5.3f}, {1:5.3f},".format(
            epsilon[0] if epsilon[0] else 0.0,
            epsilon[1] if epsilon[1] else 0.0
        )
    elif isinstance(epsilon, float):
        epsilon_str = ", EPSILON: {0:5.3f},".format(
            epsilon if epsilon else 0.0,
        )
    else:
        epsilon_str = ""

    mean_episode_reward_str = "{0:9.3f}".format(mean_episode_reward)

    if isinstance(episode_reward, np.ndarray):
        episode_reward = episode_reward[0]

    if evaluation_msg:
        print(evaluation_msg)
        print("#######################################################################################################")

    print("{0}[{1:6}/{2}] Ep. {3}, EPISODE REWARD: {4:9.3f}, MEAN_{5} EPSIODE REWARD: {6}{7} SPEED: {8:7.2f}steps/sec., {9}".format(
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

    if last_done_reason is not None:
        print(", REASON: {0}".format(last_done_reason), flush=True)
    else:
        print("", flush=True)


def advance_check():
    if params.TRAIN_ONLY_AFTER_EPISODE:
        assert params.NUM_ENVIRONMENTS == 1

    if hasattr(params, "DISTRIBUTIONAL") and params.DISTRIBUTIONAL:
        assert params.NOISY_NET
        assert hasattr(params, "NUM_SUPPORTS")
