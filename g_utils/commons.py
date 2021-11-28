import time
from datetime import date

import torch
import os
import torch.multiprocessing as mp
import wandb

from a_configuration.config.config import Config
from g_utils.types import AgentType

if torch.cuda.is_available():
    import nvidia_smi
    nvidia_smi.nvmlInit()

    import pynvml
    pynvml.nvmlInit()
else:
    nvidia_smi = None
    pynvml = None



def model_save(model, env_name, agent_type_name, test_episode_reward_avg, test_episode_reward_std):
    env_model_home = os.path.join(Config.MODEL_HOME, env_name)
    if not os.path.exists(env_model_home):
        os.mkdir(env_model_home)

    agent_model_home = os.path.join(Config.MODEL_HOME, env_name, agent_type_name)
    if not os.path.exists(agent_model_home):
        os.mkdir(agent_model_home)

    today_date = date.today()

    file_name = "{0:4.1f}_{1:3.1f}_{2}_{3}_{4}.pth".format(
        test_episode_reward_avg, test_episode_reward_std,
        today_date.year, today_date.month, today_date.day
    )

    torch.save(model.state_dict(), os.path.join(agent_model_home, file_name))


def model_load(model, env_name, agent_type_name, file_name):
    agent_model_home = os.path.join(Config.MODEL_HOME, env_name, agent_type_name)
    model_params = torch.load(os.path.join(agent_model_home, file_name))
    model.load_state_dict(model_params)


def print_basic_info(device, params):
    if device:
        n_cpu_cores = mp.cpu_count()
        print('\n' + '#' * 72 + " Base Parameters " + '#' * 73)
        print("{0:55} {1:55} {2:55}".format(
            "DEVICE: {0}".format(device),
            "CPU CORES: {0}".format(n_cpu_cores),
            "MAX TOTAL TIME STEPS: {0:,}".format(params.MAX_TRAINING_STEPS)
        ), end="\n")
        print("{0:55} {1:55} {2:55}".format(
            "N_ACTORS: {0}".format(params.N_ACTORS),
            "ENVS PER ACTOR: {0}".format(params.N_VECTORIZED_ENVS),
            "TOTAL NUMBERS OF ENVS: {0}".format(
                params.N_ACTORS * params.N_VECTORIZED_ENVS
            )
        ))

    print('-' * 75 + " Parameters " + '-' * 75)
    items = []
    for param in dir(params):
        if not param.startswith("__"):
            if param in [
                "BATCH_SIZE", "BUFFER_CAPACITY", "CONSOLE_LOG_INTERVAL_TOTAL_TIME_STEPS",
                "EPISODE_REWARD_AVG_SOLVED", "MAX_TRAINING_STEPS",
                "MIN_BUFFER_SIZE_FOR_TRAIN", "N_EPISODES_FOR_MEAN_CALCULATION",
                "TEST_INTERVAL_TOTAL_TIME_STEPS"
            ]:
                item = "{0}: {1:,}".format(param, getattr(params, param))
            else:
                item = "{0}: {1:}".format(param, getattr(params, param))
            items.append(item)
        if len(items) == 3:
            print("{0:55} {1:55} {2:55}".format(items[0], items[1], items[2]), end="\n")
            items.clear()
    print('#' * 162)
    print()


def console_log(
        total_train_start_time, total_episodes_v, total_time_steps_v,
        last_mean_episode_reward_v, n_rollout_transitions_v, train_steps_v,
        agent, params
):
    total_training_time = time.time() - total_train_start_time
    formatted_total_training_time = time.strftime(
        '%H:%M:%S', time.gmtime(total_training_time)
    )

    console_log = "[Total Episodes: {0:5,}, Total Time Steps {1:7,}] " \
                  "Mean Episode Reward: {2:5.1f}, Rolling Transitions: {3:6,} ({4:.3f}/sec.), " \
                  "Training Steps: {5:5,} ({6:.3f}/sec.), " \
        .format(
            total_episodes_v,
            total_time_steps_v,
            last_mean_episode_reward_v,
            n_rollout_transitions_v,
            n_rollout_transitions_v / total_training_time,
            train_steps_v,
            train_steps_v / total_training_time
        )

    if params.AGENT_TYPE == AgentType.Dqn:
        console_log += "Q_net_loss: {0:5.1f}, Epsilon: {0:4.2f}, ".format(
            agent.last_q_net_loss.value, agent.epsilon.value
        )
    elif params.AGENT_TYPE == AgentType.Reinforce:
        console_log += "log_policy_objective: {0:5.1f}, ".format(
            agent.last_log_policy_objective.value
        )
    elif params.AGENT_TYPE == AgentType.A2c:
        console_log += "critic_loss: {0:5.1f}, log_actor_objective: {1:5.1f}, ".format(
            agent.last_critic_loss.value, agent.last_log_actor_objective.value
        )
    else:
        pass

    # if torch.cuda.is_available():
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # gpu_resource = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # console_log += f'gpu: {0}%, gpu-mem: {1}%, '.format(gpu_resource.gpu, gpu_resource.memory)
        # import GPUtil
        # gpu = GPUtil.getGPUs()[0]
        # console_log += f'gpu: {0}%, gpu-mem: {1}%, '.format(gpu.load * 100, gpu.memoryUtil * 100)

    console_log += "Total Elapsed Time {}".format(formatted_total_training_time)

    print(console_log)


def get_wandb_obj(params):
    wandb_obj = wandb.init(
        entity=params.WANDB_ENTITY,
        project="{0}_{1}".format(params.ENV_NAME, params.AGENT_TYPE.name),
        config={
            key: getattr(params, key) for key in dir(params) if not key.startswith("__")
        }
    )
    return wandb_obj


def wandb_log(learner, wandb_obj, params):
    log_dict = {
        "[TEST] Average Episode Reward": learner.test_episode_reward_avg.value,
        "[TEST] Std. Episode Reward": learner.test_episode_reward_std.value,
        "Mean Episode Reward": learner.last_mean_episode_reward.value,
        "Episode": learner.total_episodes.value,
        "Buffer Size": learner.n_rollout_transitions.value,
        "Training Steps": learner.training_steps.value,
        "Total Time Steps": learner.total_time_steps.value
    }

    if params.AGENT_TYPE == AgentType.Dqn:
        log_dict["Q_net_loss"] = learner.agent.last_q_net_loss.value
        log_dict["Epsilon"] = learner.agent.epsilon.value
    elif params.AGENT_TYPE == AgentType.Reinforce:
        log_dict["log_policy_objective"] = learner.agent.last_log_policy_objective.value
    elif params.AGENT_TYPE == AgentType.A2c:
        log_dict["critic_loss"] = learner.agent.last_critic_loss.value
        log_dict["log_actor_objective"] = learner.agent.last_log_actor_objective.value
    else:
        pass

    wandb_obj.log(log_dict)


class EpsilonTracker:
    def __init__(self, epsilon_init, epsilon_final, epsilon_final_time_step_percent, max_training_steps):
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_final_time_step = max_training_steps * epsilon_final_time_step_percent

    def epsilon(self, training_step):
        epsilon = max(
            self.epsilon_init - training_step / self.epsilon_final_time_step,
            self.epsilon_final
        )
        return epsilon