# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim
import os, sys
import numpy as np

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from codes.f_utils.common_utils import make_gym_env, smooth
from common.fast_rl.policy_based_model import unpack_batch_for_ddpg
from common.fast_rl.rl_agent import float32_preprocessor

print(torch.__version__)

from common.fast_rl import actions, experience, policy_based_model, rl_agent, experience_single
from common.fast_rl.common import statistics, utils

from config.parameters import PARAMETERS as params
from collections import deque

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")

target_update_period = 100
state_deque = deque(maxlen=30)
step_length = 1

def play_func(exp_queue, env, net):
    print(env.action_space.low[0], env.action_space.high[0])
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]

    #action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=params.EPSILON_INIT)

    action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=2.0)

    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )

    agent = rl_agent.AgentDDPG(
        net, n_actions=1, action_selector=action_selector,
        action_min=action_min, action_max=action_max, device=device, preprocessor=float32_preprocessor
    )

    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(
        env, agent, gamma=params.GAMMA, steps_count=params.N_STEP
    )

    # experience_source = experience.ExperienceSourceFirstLast(
    #     env, agent, gamma=params.GAMMA, steps_count=params.N_STEP
    # )

    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForPolicyBasedRL(method="policy_gradient")
    else:
        stat = None

    step_idx = 0
    next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    with utils.RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEP:
            # 1 ?????? ???????????? exp??? exp_queue??? ??????
            step_idx += 1
            exp = next(exp_source_iter)


            # #######################################################################################################
            # state_deque.append(exp[0])
            #
            # if step_length == -1:
            #     next_state = np.array(state_deque[-1])
            # elif step_length >= 1:
            #     if len(state_deque) < step_length:
            #         next_state = list(state_deque)
            #
            #         for _ in range(step_length - len(state_deque)):
            #             next_state.insert(0, [0.0] * 3)
            #         next_state = np.array(next_state)
            #         print()
            #     else:
            #         next_state = np.array(
            #             [
            #                 state_deque[-step_length + offset] for offset in range(step_length)
            #             ]
            #         )
            #     # print(next_state.shape)
            #     # print(next_state)
            # else:
            #     raise ValueError()
            #
            # print("%%%%%%%%%%%%%",next_state)
            # print(exp)
            # ############################################################################################################################


            exp_queue.put(exp)
            epsilon_tracker.udpate(step_idx)

            episode_rewards = experience_source.pop_episode_reward_lst()
            if episode_rewards:
                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    episode_rewards[0], step_idx, epsilon=action_selector.epsilon
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

    actor_net = policy_based_model.DDPGGruActor(
        obs_size=3,
        hidden_size_1=128, hidden_size_2=64,
        n_actions=1,
        bidirectional=False,
        scale=2.0
    ).to(device)

    critic_net = policy_based_model.DDPGGruCritic(
        obs_size=3,
        hidden_size_1=128, hidden_size_2=64,
        n_actions=1,
        bidirectional=False
    ).to(device)

    print(actor_net)
    print(critic_net)

    target_actor_net = rl_agent.TargetNet(actor_net)
    target_critic_net = rl_agent.TargetNet(critic_net)

    actor_optimizer = optim.Adam(actor_net.parameters(), lr=params.ACTOR_LEARNING_RATE)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=params.LEARNING_RATE)

    buffer = experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE)
    # buffer = experience.PrioritizedReplayBuffer(
    #     experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.N_STEP
    # )

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    play_proc = mp.Process(target=play_func, args=(exp_queue, env, actor_net))
    play_proc.start()

    time.sleep(0.5)

    if params.DRAW_VIZ:

        #stat_for_ddpg = statistics.StatisticsForDDPGOptimization(n_actions=1)
        stat_for_ddpg = statistics.StatisticsForSimpleDDPGOptimization(n_actions=1)
    else:
        stat_for_ddpg = 0.0

    step_idx = 0

    actor_grad_l2 = 0.0
    actor_grad_max = 0.0
    actor_grad_variance = 0.0

    critic_grad_l2 = 0.0
    critic_grad_max = 0.0
    critic_grad_variance = 0.0

    loss_actor = 0.0
    loss_critic = 0.0
    loss_total = 0.0

    #$ pip install line_profiler
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(model_update)

    while play_proc.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        exp = None
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)

        if len(buffer) < params.MIN_REPLAY_SIZE_FOR_TRAIN:
            continue

        if exp is not None and exp.last_state is None:
            for _ in range(3):
                # actor_grad_l2, actor_grad_max, actor_grad_variance, critic_grad_l2, critic_grad_max, critic_grad_variance, loss_actor, loss_critic, loss_total = lp_wrapper(
                # buffer, actor_net, critic_net, target_actor_net, target_critic_net, actor_optimizer, critic_optimizer,
                #     stat_for_ddpg, step_idx, exp,
                #     actor_grad_l2, actor_grad_max, actor_grad_variance,
                #     critic_grad_l2, critic_grad_max, critic_grad_variance,
                #     loss_actor, loss_critic, loss_total, len(buffer.buffer)
                # )
                #
                # lp.print_stats()

                actor_grad_l2, actor_grad_max, actor_grad_variance, critic_grad_l2, critic_grad_max, critic_grad_variance, loss_actor, loss_critic, loss_total = model_update(
                    buffer, actor_net, critic_net, target_actor_net, target_critic_net, actor_optimizer, critic_optimizer,
                    step_idx, actor_grad_l2, actor_grad_max, actor_grad_variance,
                    critic_grad_l2, critic_grad_max, critic_grad_variance,
                    loss_actor, loss_critic, loss_total, per=False
                )

            if params.DRAW_VIZ:
                # stat_for_ddpg.draw_optimization_performance(
                #     step_idx,
                #     loss_actor, loss_critic, loss_total,
                #     actor_grad_l2, actor_grad_variance, actor_grad_max,
                #     critic_grad_l2, critic_grad_variance, critic_grad_max,
                #     buffer_length, exp.noise, exp.action
                # )

                stat_for_ddpg.draw_optimization_performance(
                    step_idx, exp.noise, exp.action
                )


def model_update(buffer, actor_net, critic_net, target_actor_net, target_critic_net, actor_optimizer, critic_optimizer,
                 step_idx, actor_grad_l2, actor_grad_max, actor_grad_variance,
                 critic_grad_l2, critic_grad_max, critic_grad_variance,
                 loss_actor, loss_critic, loss_total, per):
    global target_update_period
    if per:
        batch, batch_indices, batch_weights = buffer.sample(params.BATCH_SIZE)
    else:
        batch = buffer.sample(params.BATCH_SIZE)
        batch_indices, batch_weights = None, None

    #print(batch)

    batch_states_v, batch_actions_v, batch_rewards_v, batch_dones_mask, batch_last_states_v = unpack_batch_for_ddpg(
        batch, device
    )

    # train critic
    critic_optimizer.zero_grad()
    batch_q_v = critic_net(batch_states_v, batch_actions_v)
    batch_last_act_v = target_actor_net.target_model(batch_last_states_v)
    batch_q_last_v = target_critic_net.target_model(batch_last_states_v, batch_last_act_v)
    batch_q_last_v[batch_dones_mask] = 0.0
    batch_target_q_v = batch_rewards_v.unsqueeze(dim=-1) + batch_q_last_v * params.GAMMA ** params.N_STEP

    if per:
        batch_l1_loss = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach(), reduction='none') # for PER
        batch_weights_v = torch.tensor(batch_weights)
        loss_critic_v = batch_weights_v * batch_l1_loss

        buffer.update_priorities(batch_indices, batch_l1_loss.detach().cpu().numpy() + 1e-5)
        buffer.update_beta(step_idx)
    else:
        loss_critic_v = F.smooth_l1_loss(batch_q_v, batch_target_q_v.detach())

    loss_critic_v.mean().backward()

    critic_grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                   for p in critic_net.parameters()
                                   if p.grad is not None])
    critic_optimizer.step()

    # train actor
    actor_optimizer.zero_grad()
    batch_current_actions_v = actor_net(batch_states_v)
    actor_loss_v = -critic_net(batch_states_v, batch_current_actions_v)
    loss_actor_v = actor_loss_v.mean()
    loss_actor_v.backward()

    actor_grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                  for p in actor_net.parameters()
                                  if p.grad is not None])
    actor_optimizer.step()

    target_actor_net.alpha_sync(alpha=1.0 - 0.001)
    target_critic_net.alpha_sync(alpha=1.0 - 0.001)
    # if target_update_period >= 100:
    #     target_update_period = 0
    #     target_actor_net.target_model.load_state_dict(target_actor_net.model.state_dict())
    #     target_critic_net.target_model.load_state_dict(target_critic_net.model.state_dict())
    # target_update_period += 1

    actor_grad_l2 = smooth(actor_grad_l2, np.sqrt(np.mean(np.square(actor_grads))))
    actor_grad_max = smooth(actor_grad_max, np.max(np.abs(actor_grads)))
    actor_grad_variance = smooth(actor_grad_variance, float(np.var(actor_grads)))

    critic_grad_l2 = smooth(critic_grad_l2, np.sqrt(np.mean(np.square(critic_grads))))
    critic_grad_max = smooth(critic_grad_max, np.max(np.abs(critic_grads)))
    critic_grad_variance = smooth(critic_grad_variance, float(np.var(critic_grads)))

    loss_actor = smooth(loss_actor, loss_actor_v.item())
    loss_critic = smooth(loss_critic, loss_critic_v.mean().item())
    loss_total = smooth(loss_total, loss_actor_v.item() + loss_critic_v.mean().item())

    return actor_grad_l2, actor_grad_max, actor_grad_variance, critic_grad_l2, critic_grad_max, critic_grad_variance, loss_actor, loss_critic, loss_total


if __name__ == "__main__":
    main()