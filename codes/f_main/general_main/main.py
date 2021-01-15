# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
import torch
import torch.multiprocessing as mp
import os, sys

print("PyTorch Version", torch.__version__)

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir, os.pardir))
if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.a_config.parameters import PARAMETERS as params

from codes.e_utils import rl_utils
from codes.e_utils.common_utils import save_model
from codes.e_utils.experience_single import ExperienceSourceSingleEnvFirstLast
from codes.e_utils.experience_tracker import RewardTracker
from codes.e_utils.logger import get_logger
from codes.e_utils.names import DeepLearningModelName, RLAlgorithmName, EnvironmentName

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

my_logger = get_logger("openai_pendulum_ddpg")

env = rl_utils.get_environment(owner="actual_worker", params=params)
print("env:", params.ENVIRONMENT_ID)
print("observation_space:", env.observation_space)
print("action_space:", env.action_space)

def play_func(exp_queue, agent, epsilon_tracker):
    if params.DEEP_LEARNING_MODEL in [
        DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU,
        DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_GRU_ATTENTION
    ]:
        step_length = params.RNN_STEP_LENGTH
    else:
        step_length = -1

    experience_source = ExperienceSourceSingleEnvFirstLast(
        env=env, agent=agent, gamma=params.GAMMA, steps_count=params.N_STEP, step_length=step_length
    )

    exp_source_iter = iter(experience_source)
    stat = None
    step_idx = 0

    with RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEP:
            # 1 스텝 진행하고 exp를 exp_queue에 넣음
            step_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            if epsilon_tracker:
                epsilon_tracker.udpate(step_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()
            if episode_rewards:
                current_episode_reward = episode_rewards[0]

                epsilon = agent.action_selector.epsilon if hasattr(agent.action_selector, 'epsilon') else None

                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    current_episode_reward, step_idx, epsilon=epsilon, last_info=exp.info, model=agent.model
                )

                if solved:
                    save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, mean_episode_reward
                    )
                    if solved:
                        break

        if params.SAVE_AT_MAX_GLOBAL_STEPS:
            save_model(
                MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, agent, step_idx, mean_episode_reward
            )

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    agent, epsilon_tracker = rl_utils.get_rl_agent(env=env, worker_id=0, params=params, device=device)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(exp_queue, agent, epsilon_tracker))
    play_proc.start()

    time.sleep(0.5)

    step_idx = 0
    trajectory = []
    exp = None

    while play_proc.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            agent.buffer._add(exp)

        if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0]:
            # print(last_experience[0])
            assert params.TRAIN_STEP_FREQ == 1 and exp is not None
            trajectory.append(exp)
            if len(trajectory) < params.PPO_TRAJECTORY_SIZE:
                continue
        elif params.RL_ALGORITHM in [RLAlgorithmName.DDPG_V0, RLAlgorithmName.DQN_V0]:
            if len(agent.buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
                continue
        else:
            if len(agent.buffer) < params.BATCH_SIZE:
                continue

        if params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_PPO_V0]:
            _, last_loss, _ = agent.train_net(trajectory=trajectory)
            trajectory.clear()
        elif params.RL_ALGORITHM in [
            RLAlgorithmName.DDPG_V0,
            RLAlgorithmName.DISCRETE_A2C_V0,
            RLAlgorithmName.CONTINUOUS_A2C_V0
        ]:
            _, last_loss, _ = agent.train_net(step_idx=step_idx)
        elif params.RL_ALGORITHM == RLAlgorithmName.DQN_V0:
            _, last_loss = agent.train_net(step_idx=step_idx)
        else:
            raise ValueError()

        if params.PER_RANK_BASED:
            if step_idx % 100 < params.TRAIN_STEP_FREQ:
                agent.buffer.rebalance()


if __name__ == "__main__":
    main()