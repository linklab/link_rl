#!/usr/bin/env python3
import gym
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common.fast_rl import experience, rl_agent

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 32

REWARD_STEPS = 10


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=False, action='store_true', help="Enable mean baseline")
    args = parser.parse_args()

    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-pg" + "-baseline=%s" % args.baseline)

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    agent = rl_agent.PolicyAgent(net, preprocessor=rl_agent.float32_preprocessor, apply_softmax=True)
    experience_source = experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_episode_rewards = []
    step_rewards = []
    step_idx = 0
    done_episodes = 0
    reward_sum = 0.0

    batch_states, batch_actions, batch_scales = [], [], []

    for step_idx, exp in enumerate(experience_source):
        reward_sum += exp.reward
        baseline = reward_sum / (step_idx + 1)
        writer.add_scalar("baseline", baseline, step_idx)
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        if args.baseline:
            batch_scales.append(exp.reward - baseline)
        else:
            batch_scales.append(exp.reward)

        # handle new rewards
        new_episode_rewards = experience_source.pop_episode_reward_lst()
        if new_episode_rewards:
            done_episodes += 1
            episode_reward = new_episode_rewards[0]
            total_episode_rewards.append(episode_reward)
            mean_episode_reward = float(np.mean(total_episode_rewards[-100:]))
            print(
                f"step: {step_idx}, episode_reward: {episode_reward:6.2f}, "
                f"mean_100_episode_reward: {mean_episode_reward:6.2f}, episodes: {done_episodes:d}"
            )
            writer.add_scalar("episode_reward", episode_reward, step_idx)
            writer.add_scalar("mean_100_episode_reward", mean_episode_reward, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_episode_reward > 195:
                print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
                break

        if len(batch_states) < BATCH_SIZE:
            continue

        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scale_v = torch.FloatTensor(batch_scales)

        optimizer.zero_grad()
        logits_v = net(states_v)                      # logits_v: (32, 2)
        log_prob_v = F.log_softmax(logits_v, dim=1)   # Applies a softmax followed by a logarithm --> lo_prob_v: (32, 2)
        log_p_a_v = log_prob_v[range(BATCH_SIZE), batch_actions_t]  # (32,)
        log_prob_actions_v = batch_scale_v * log_p_a_v  # (32,)
        loss_policy_v = -log_prob_actions_v.mean()      # (1,)

        # https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
        # What does the parameter retain_graph mean in the Variable's backward() method?
        loss_policy_v.backward(retain_graph=True)
        grads = np.concatenate([p.grad.data.numpy().flatten() for p in net.parameters() if p.grad is not None])

        prob_v = F.softmax(logits_v, dim=1)           # prob_v: (32, 2)
        entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()  # (1,) 엔트로피가 높을 수록 각 행동이 다양하게 샘플
        entropy_loss_v = -ENTROPY_BETA * entropy_v
        entropy_loss_v.backward()

        optimizer.step()

        loss_v = loss_policy_v + entropy_loss_v

        # calc KL-div
        new_logits_v = net(states_v)
        new_prob_v = F.softmax(new_logits_v, dim=1)
        kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
        writer.add_scalar("kl", kl_div_v.item(), step_idx)

        writer.add_scalar("baseline", baseline, step_idx)
        writer.add_scalar("entropy", entropy_v.item(), step_idx)
        writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
        writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
        writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
        writer.add_scalar("loss_total", loss_v.item(), step_idx)

        g_l2 = np.sqrt(np.mean(np.square(grads)))
        g_max = np.max(np.abs(grads))
        writer.add_scalar("grad_l2", g_l2, step_idx)
        writer.add_scalar("grad_max", g_max, step_idx)
        writer.add_scalar("grad_var", np.var(grads), step_idx)

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()

    writer.close()
