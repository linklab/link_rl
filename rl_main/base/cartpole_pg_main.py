#!/usr/bin/env python3
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim
import os
import numpy as np

from common.common_utils import make_gym_env, smooth
from common.fast_rl.rl_agent import float32_preprocessor

print(torch.__version__)

from common.fast_rl import actions, experience, dqn_model, rl_agent
from common.fast_rl.common import statistics, utils

from config.parameters import PARAMETERS as params

MODEL_SAVE_DIR = os.path.join(".", "saved_models")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if params.CUDA else "cpu")


def play_func(exp_queue, env, net):
    agent = rl_agent.PolicyAgent(net, preprocessor=float32_preprocessor, apply_softmax=True)

    experience_source = experience.ExperienceSourceFirstLast(
        env, agent, gamma=params.GAMMA, steps_count=params.N_STEP
    )

    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.Statistics(method="nature_dqn")
    else:
        stat = None

    step_idx = 0
    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.RewardTracker(params.STOP_MEAN_EPISODE_REWARD, params.AVG_EPISODE_SIZE_FOR_STAT, params.DRAW_VIZ, stat) as reward_tracker:
        while True:
            # 1 스텝 진행하고 exp를 exp_queue에 넣음
            step_idx += 1
            exp = next(exp_source_iter)
            exp_queue.put(exp)

            episode_rewards = experience_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    episode_rewards[0], step_idx, epsilon=0.0
                )

                if step_idx >= next_save_frame_idx:
                    dqn_model.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, step_idx, mean_episode_reward
                    )
                    next_save_frame_idx += params.MODEL_SAVE_STEP_PERIOD

                if solved:
                    dqn_model.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, step_idx, mean_episode_reward
                    )
                    break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    env = make_gym_env(params.ENVIRONMENT_ID.value, seed=params.SEED)

    net = dqn_model.DuelingDQNMLP(
        obs_size=4,
        hidden_size_1=128, hidden_size_2=128,
        n_actions=2
    ).to(device)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(exp_queue, env, net))
    play_proc.start()

    time.sleep(0.5)

    if params.DRAW_VIZ:
        stat_for_policy_based_rl = statistics.StatisticsForPolicyBasedRL()
    else:
        stat_for_policy_based_rl = 0.0

    step_idx = 0
    reward_sum = 0.0

    mean_batch_scale = 0.0
    entropy = 0.0
    loss_entropy = 0.0
    loss_policy = 0.0
    loss_total = 0.0

    batch_states, batch_actions, batch_scales = [], [], []

    while play_proc.is_alive():
        exp = exp_queue.get()
        if exp is None:
            play_proc.join()
            break
        step_idx += 1

        reward_sum += exp.reward
        baseline = reward_sum / step_idx

        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        batch_scales.append(exp.reward - baseline)

        if len(batch_states) < params.BATCH_SIZE:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        optimizer.zero_grad()
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_scale_v * log_prob_v[range(params.BATCH_SIZE), batch_actions_t]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
        entropy_loss_v = -params.ENTROPY_BETA * entropy_v

        # loss_policy_v를 작아지도록 만듦 --> log_prob_actions_v.mean()가 커지도록 만듦
        # entropy_loss_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦
        loss_v = loss_policy_v + entropy_loss_v

        loss_v.backward()
        optimizer.step()

        # calc KL-div
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()

        grad_max = 0.0
        grad_means = 0.0
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1
        grad_means = grad_means / grad_count

        mean_batch_scale = smooth(mean_batch_scale, float(np.mean(batch_scales)))
        entropy = smooth(entropy, entropy_v.item())
        loss_entropy = smooth(loss_entropy, entropy_loss_v.item())
        loss_policy = smooth(loss_policy, loss_policy_v.item())
        loss_total = smooth(loss_total, loss_v.item())

        if params.DRAW_VIZ:
            stat_for_policy_based_rl.draw_optimization_performance(
                step_idx,
                kl_div_v.item(),
                baseline,
                mean_batch_scale,
                entropy,
                loss_entropy, loss_policy, loss_total,
                grad_means, grad_max
            )

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()


if __name__ == "__main__":
    main()