#!/usr/bin/env python3
import math
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.nn.utils as nn_utils
from torch import optim
import os
import numpy as np

from common.common_utils import make_gym_env, smooth
from common.fast_rl.policy_based_model import unpack_batch_for_policy_gradient
from common.fast_rl.rl_agent import float32_preprocessor

print(torch.__version__)

from common.fast_rl import actions, experience, policy_based_model, rl_agent
from common.fast_rl.common import statistics, utils

from config.parameters import PARAMETERS as params

MODEL_SAVE_DIR = os.path.join(".", "saved_models")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if params.CUDA else "cpu")


def calc_log_prob_actions(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


def play_func(exp_queue, env, net):
    action_min = -env.max_torque
    action_max = env.max_torque

    agent = rl_agent.ContinuousActorCriticAgent(
        net, action_min=action_min, action_max=action_max, device=device, preprocessor=float32_preprocessor
    )

    experience_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params.GAMMA, steps_count=params.N_STEP)

    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForPolicyBasedRL(method="policy_gradient")
    else:
        stat = None

    step_idx = 0
    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.RewardTracker(
            params.STOP_MEAN_EPISODE_REWARD, params.AVG_EPISODE_SIZE_FOR_STAT,
            frame=True, draw_viz=params.DRAW_VIZ, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEPS:
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
                    rl_agent.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, step_idx, mean_episode_reward
                    )
                    next_save_frame_idx += params.MODEL_SAVE_STEP_PERIOD

                if solved:
                    rl_agent.save_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID.value, net.__name__, net, step_idx, mean_episode_reward
                    )
                    break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    env = make_gym_env(params.ENVIRONMENT_ID.value, seed=params.SEED)
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", env.observation_space)
    print("action_space:", env.action_space)
    net = policy_based_model.ContinuousA2CMLP(
        obs_size=3,
        hidden_size_1=128, hidden_size_2=128,
        n_actions=1
    ).to(device)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=params.LEARNING_RATE)

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(exp_queue, env, net))
    play_proc.start()

    time.sleep(0.5)

    if params.DRAW_VIZ:
        stat_for_a2c = statistics.StatisticsForA2COptimization()
    else:
        stat_for_a2c = 0.0

    step_idx = 0

    grad_l2 = 0.0
    grad_max = 0.0
    grad_variance = 0.0

    mean_advantage = 0.0
    entropy = 0.0
    loss_actor = 0.0
    loss_critic = 0.0
    loss_entropy = 0.0
    loss_total = 0.0

    batch = []

    while play_proc.is_alive():
        exp = exp_queue.get()
        if exp is None:
            play_proc.join()
            break
        step_idx += 1

        batch.append(exp)

        if len(batch) < params.BATCH_SIZE:
            continue

        batch_states_v, batch_actions_v, batch_target_values_v = unpack_batch_for_policy_gradient(
            batch, net, params, device=device
        )
        batch.clear()

        optimizer.zero_grad()
        batch_mu_v, batch_var_v, batch_value_v = net(batch_states_v)
        loss_critic_v = F.mse_loss(batch_value_v.squeeze(-1), batch_target_values_v)

        batch_advantage_v = batch_target_values_v - batch_value_v.squeeze(-1).detach()
        batch_log_prob_actions_v = batch_advantage_v * calc_log_prob_actions(batch_mu_v, batch_var_v, batch_actions_v)
        loss_actor_v = -batch_log_prob_actions_v.mean()

        entropy_v = (torch.log(2 * math.pi * batch_var_v) + 1) / 2
        entropy_v = entropy_v.mean()
        loss_entropy_v = -params.ENTROPY_BETA * entropy_v

        # loss_policy_v를 작아지도록 만듦 --> log_prob_actions_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦
        loss_v = loss_actor_v + loss_critic_v + loss_entropy_v
        loss_v.backward()

        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in net.parameters()
                                if p.grad is not None])

        nn_utils.clip_grad_norm_(net.parameters(), params.CLIP_GRAD)
        optimizer.step()

        # calc KL-div
        batch_new_mu_v, batch_new_var_v, _ = net(batch_states_v)
        batch_new_mu_v = F.softmax(batch_new_mu_v, dim=1)
        kl_div_v = -((batch_new_mu_v / batch_mu_v).log() * batch_mu_v).sum(dim=1).mean()

        grad_l2 = smooth(grad_l2, np.sqrt(np.mean(np.square(grads))))
        grad_max = smooth(grad_max, np.max(np.abs(grads)))
        grad_variance = smooth(grad_variance, float(np.var(grads)))

        mean_advantage = smooth(mean_advantage, float(np.mean(batch_advantage_v.numpy())))
        entropy = smooth(entropy, entropy_v.item())
        loss_actor = smooth(loss_actor, loss_actor_v.item())
        loss_critic = smooth(loss_critic, loss_critic_v.item())
        loss_entropy = smooth(loss_entropy, loss_entropy_v.item())
        loss_total = smooth(loss_total, loss_v.item())

        if params.DRAW_VIZ:
            stat_for_a2c.draw_optimization_performance(
                step_idx,
                kl_div_v.item(),
                mean_advantage,
                entropy,
                loss_actor, loss_critic, loss_entropy, loss_total,
                grad_l2, grad_variance, grad_max
            )


if __name__ == "__main__":
    main()