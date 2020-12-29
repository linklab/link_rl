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
from config.names import PROJECT_HOME

print(torch.__version__)

from common.fast_rl import experience, value_based_model, rl_agent
from common.fast_rl.common import statistics, utils

from config.parameters import PARAMETERS as params

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


def calc_discounted_q_values_baseline(rewards):
    discounted_q_values = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= params.GAMMA
        sum_r += r
        discounted_q_values.append(sum_r)
    discounted_q_values = list(reversed(discounted_q_values))
    baseline = np.mean(discounted_q_values)
    return [q - baseline for q in discounted_q_values], baseline


def play_func(exp_queue, env, net):
    agent = rl_agent.PolicyAgent(net, preprocessor=float32_preprocessor, apply_softmax=True, device=device)

    experience_source = experience.ExperienceSourceFirstLast(env, agent, gamma=params.GAMMA, steps_count=1)

    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForPolicyBasedRL(method="policy_gradient")
    else:
        stat = None

    step_idx = 0
    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEP:
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

    net = value_based_model.DuelingDQNMLP(
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
        stat_for_policy_based_rl = statistics.StatisticsForPolicyBasedRLOptimization()
    else:
        stat_for_policy_based_rl = 0.0

    step_idx = 0

    grad_l2 = 0.0
    grad_max = 0.0
    grad_variance = 0.0

    mean_batch_scale = 0.0
    baseline = 0.0
    loss_total = 0.0

    batch_episodes = 0
    batch_states, batch_actions, batch_scales = [], [], []
    single_episode_states, single_episode_actions, single_episode_rewards = [], [], []

    while play_proc.is_alive():
        exp = exp_queue.get()
        if exp is None:
            play_proc.join()
            break
        step_idx += 1

        single_episode_states.append(exp.state)
        single_episode_actions.append(int(exp.action))
        single_episode_rewards.append(exp.reward)

        if exp.last_state is None:
            batch_states.extend(single_episode_states)
            batch_actions.extend(single_episode_actions)

            discounted_q_values_baseline, baseline = calc_discounted_q_values_baseline(single_episode_rewards)
            batch_scales.extend(discounted_q_values_baseline)

            single_episode_states.clear()
            single_episode_actions.clear()
            single_episode_rewards.clear()

            batch_episodes += 1

        if batch_episodes < params.EPISODES_TO_TRAIN:
            continue

        batch_states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_scales_v = torch.FloatTensor(batch_scales)

        optimizer.zero_grad()
        batch_logits_v = net(batch_states_v)
        batch_log_prob_v = F.log_softmax(batch_logits_v, dim=1)
        batch_log_prov_actions_v = batch_log_prob_v[range(len(batch_states)), batch_actions_t]
        batch_log_prob_actions_v = batch_scales_v * batch_log_prov_actions_v
        loss_v = -batch_log_prob_actions_v.mean()

        batch_prob_v = F.softmax(batch_logits_v, dim=1)

        loss_v.backward()

        grads = np.concatenate([p.grad.data.numpy().flatten()
                                for p in net.parameters()
                                if p.grad is not None])

        optimizer.step()

        # calc KL-div
        batch_new_logits_v = net(batch_states_v)
        batch_new_prob_v = F.softmax(batch_new_logits_v, dim=1)
        kl_div_v = -((batch_new_prob_v / batch_prob_v).log() * batch_prob_v).sum(dim=1).mean()

        grad_l2 = smooth(grad_l2, np.sqrt(np.mean(np.square(grads))))
        grad_max = smooth(grad_max, np.max(np.abs(grads)))
        grad_variance = smooth(grad_variance, float(np.var(grads)))

        mean_batch_scale = smooth(mean_batch_scale, float(np.mean(batch_scales)))
        entropy = 0.0
        loss_policy = smooth(loss_total, loss_v.item())
        loss_entropy = 0.0
        loss_total = smooth(loss_total, loss_v.item())

        if params.DRAW_VIZ:
            stat_for_policy_based_rl.draw_optimization_performance(
                step_idx,
                kl_div_v.item(),
                baseline,
                mean_batch_scale,
                entropy,
                loss_policy, loss_entropy, loss_total,
                grad_l2, grad_variance, grad_max
            )

        batch_episodes = 0

        batch_states.clear()
        batch_actions.clear()
        batch_scales.clear()


if __name__ == "__main__":
    main()