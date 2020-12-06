# https://github.com/AdrianHsu/breakout-Deep-Q-Network
#!/usr/bin/env python3
import time
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import os
import warnings

from common import common_utils
from common.common_utils import make_atari_env
from common.fast_rl import experience, rl_agent, value_based_model, actions
from common.fast_rl.common import utils
from common.fast_rl.common import statistics, wrappers
from common.fast_rl.value_based_model import insert_experience_into_buffer

##### NOTE #####
from config.parameters import PARAMETERS as params
##### NOTE #####

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

MODEL_SAVE_DIR = os.path.join(".", "saved_models")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)


def play_func(env, net, exp_queue):
    action_selector = actions.EpsilonGreedyActionSelector(epsilon=params.EPSILON_INIT)
    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )
    agent = rl_agent.DQNAgent(net, action_selector, device=device)
    experience_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params.GAMMA, steps_count=1)
    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForValueBasedRL(method="nature_dqn")
    else:
        stat = None

    action_count = []
    for _ in env.unwrapped.get_action_meanings():
        action_count.append(0)

    frame_idx = 0

    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.RewardTracker(params=params, frame=True, stat=stat) as reward_tracker:
        while frame_idx < params.MAX_GLOBAL_STEPS:
            frame_idx += 1
            exp = next(exp_source_iter)
            action_count[exp.action] += 1
            exp_queue.put(exp)

            epsilon_tracker.udpate(frame_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()

            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.set_episode_reward(episode_rewards[0], frame_idx, action_selector.epsilon, action_count)

                if frame_idx >= next_save_frame_idx:
                    rl_agent.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, frame_idx, mean_episode_reward
                    )
                    next_save_frame_idx += params.MODEL_SAVE_STEP_PERIOD

                if solved:
                    rl_agent.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, frame_idx, mean_episode_reward
                    )
                    break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')
    common_utils.print_fast_rl_params(params)

    params.BATCH_SIZE *= params.TRAIN_STEP_FREQ

    env = make_atari_env(params.ENVIRONMENT_ID.value, seed=params.SEED)
    if params.SEED is not None:
        env.seed(params.SEED)

    suffix = "" if params.SEED is None else "_seed=%s" % params.SEED

    net = value_based_model.DQN(
        input_shape=env.observation_space.shape,
        n_actions=env.action_space.n
    ).to(device)
    print(net)
    print("ACTION MEANING: {0}".format(env.unwrapped.get_action_meanings()))

    tgt_net = rl_agent.TargetNet(net)

    buffer = experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE)
    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)
    # optimizer = optim.RMSprop(net.parameters(), lr=params.LEARNING_RATE, momentum=0.95, eps=0.01)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(env, net, exp_queue))
    play_proc.start()

    time.sleep(0.5)
    if params.DRAW_VIZ:
        stat_for_model_loss = statistics.StatisticsForValueBasedOptimization()
    else:
        stat_for_model_loss = None

    frame_idx = 0

    while play_proc.is_alive():
        frame_idx += params.TRAIN_STEP_FREQ
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            insert_experience_into_buffer(exp, buffer)

        if len(buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
            if params.DRAW_VIZ and frame_idx % 1000 == 0:
                stat_for_model_loss.draw_optimization_performance(frame_idx, 0.0)
            continue

        optimizer.zero_grad()
        batch = buffer.sample(params.BATCH_SIZE)
        loss_v = value_based_model.calc_loss_dqn(batch, net, tgt_net, gamma=params.GAMMA, cuda=params.CUDA, cuda_async=True)
        loss_v.backward()
        optimizer.step()

        if params.DRAW_VIZ and frame_idx % 1000 == 0:
            stat_for_model_loss.draw_optimization_performance(frame_idx, loss_v.item())

        if frame_idx % params.TARGET_NET_SYNC_STEP_PERIOD < params.TRAIN_STEP_FREQ:
            tgt_net.sync()


# python atari_dqn.py --env=pong --draw_viz=1 --cuda
# python atari_dqn.py --env=breakout --draw_viz=1 --cuda
if __name__ == "__main__":
    main()