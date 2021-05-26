#!/usr/bin/env python3
import os
import math
import gym

import argparse

from codes.e_utils.experience import ExperienceSourceFirstLast
from codes.e_utils.train_tracker import RewardTracker
from codes.e_utils.names import EnvironmentName
from z_externals.handson_second_edition.Chapter17.lib import model, common

import torch
import torch.optim as optim
import torch.nn.functional as F

from codes.a_config.parameters import PARAMETERS as parameters

params = parameters

ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4

TEST_ITERS = 1000


def calc_logprob(mu_v, var_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable CUDA')
    parser.add_argument("-n", "--name", default="a2c", help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    save_path = os.path.join("saves", "a2c-" + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(EnvironmentName.PENDULUM_V0.value)

    net = model.ModelA2C(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net)

    agent = model.AgentA2C(net, device=device)
    experience_source = ExperienceSourceFirstLast(env, agent, GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    batch = []
    episode = 0
    best_reward = None
    with RewardTracker(params=params) as tracker:
        for step_idx, exp in enumerate(experience_source):
            episode_rewards, episode_steps = experience_source.pop_episode_reward_and_done_step_lst()
            if episode_rewards and episode_steps:
                for current_episode_reward, current_episode_step in zip(episode_rewards, episode_steps):
                    episode += 1
                    epsilon = 0.0
                    mean_loss = 0.0

                    solved, mean_episode_reward = tracker.set_episode_reward(
                        episode_reward=current_episode_reward, episode_done_step=step_idx, epsilon=epsilon,
                        last_info=exp.info, current_episode_step=current_episode_step,
                        mean_loss=mean_loss, model=agent.net, wandb=False
                    )

            batch.append(exp)
            if len(batch) < BATCH_SIZE:
                continue

            states_v, actions_v, vals_ref_v = common.unpack_batch_a2c(
                batch, net, device=device, last_val_gamma=GAMMA ** REWARD_STEPS
            )
            batch.clear()

            optimizer.zero_grad()
            mu_v, var_v, value_v = net(states_v)

            loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

            adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
            log_prob_v = adv_v * calc_logprob(mu_v, var_v, actions_v)
            loss_policy_v = -log_prob_v.mean()

            ent_v = -(torch.log(2*math.pi*var_v) + 1)/2
            entropy_loss_v = ENTROPY_BETA * ent_v.mean()

            loss_v = loss_policy_v + entropy_loss_v + loss_value_v
            loss_v.backward()
            optimizer.step()

