# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import math
import profile
import time
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim
import os, sys
import numpy as np
import math
idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
sys.path.append(PROJECT_HOME)

from common.environments.matlab.matlabenv import MatlabRotaryInvertedPendulumEnv

from common.common_utils import make_gym_env, smooth
from common.fast_rl.policy_based_model import unpack_batch_for_ddpg
from common.fast_rl.rl_agent import float32_preprocessor

print(torch.__version__)

from common.fast_rl import actions, experience, policy_based_model, rl_agent
from common.fast_rl.common import statistics, utils

from config.parameters import PARAMETERS as params

MODEL_SAVE_DIR = os.path.join(".", "saved_models")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda" if params.CUDA else "cpu")
else:
    device = torch.device("cpu")


SWING_UP_SCALE_FACTOR = 0.025
BALANCING_SCALE_FACTOR = 0.001


def play_func(exp_queue, exp_queue_balance, env, actor_net, critic_net, actor_balance_net, critic_balance_net):
    # print(env.action_space.low[0], env.action_space.high[0])
    env.start()
    swing_up_action_min = -SWING_UP_SCALE_FACTOR
    swing_up_action_max = SWING_UP_SCALE_FACTOR
    balancing_action_min = -BALANCING_SCALE_FACTOR
    balancing_action_max = BALANCING_SCALE_FACTOR
    count_bal = 0

    # action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=params.EPSILON_INIT)

    action_selector = actions.DDPGActionSelector(
        epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=SWING_UP_SCALE_FACTOR
    )

    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )

    agent = rl_agent.AgentDDPG(
        actor_net, n_actions=1, action_selector=action_selector,
        action_min=swing_up_action_min, action_max=swing_up_action_max, device=device, preprocessor=float32_preprocessor
    )

    experience_source = experience.ExperienceSourceSingleEnvFirstLast(
        env, agent, gamma=params.GAMMA, steps_count=params.N_STEP
    )

    exp_source_iter = iter(experience_source)

##########################################################################################################################
    action_selector_bal = actions.DDPGActionSelector(
        epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=BALANCING_SCALE_FACTOR
    )

    epsilon_tracker_balance = actions.EpsilonTracker(
        action_selector=action_selector_bal,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )

    agent_bal = rl_agent.AgentDDPG(
        actor_balance_net, n_actions=1, action_selector=action_selector_bal,
        action_min=balancing_action_min, action_max=balancing_action_max, device=device,
        preprocessor=float32_preprocessor
    )

    experience_source_balance = experience.ExperienceSourceSingleEnvFirstLast(
        env, agent_bal, gamma=params.GAMMA, steps_count=params.N_STEP
    )

    exp_source_iter_balance = iter(experience_source_balance)
#########################################################################################################################

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForPolicyBasedRL(method="policy_gradient")
    else:
        stat = None

    step_idx = 0
    #next_save_frame_idx = params.MODEL_SAVE_STEP_PERIOD

    best_episode_reward = 0
    #current_episode_reward = 0

    with utils.RewardTracker(
            params.STOP_MEAN_EPISODE_REWARD, params.AVG_EPISODE_SIZE_FOR_STAT,
            frame=True, draw_viz=params.DRAW_VIZ, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEPS:
            # 1 스텝 진행하고 exp를 exp_queue에 넣음
            step_idx += 1

            if step_idx == 1:
                exp = next(exp_source_iter)

            if math.cos(3.0892) < exp[0][0] < math.cos(3.19395): #cos(177) < exp[0][0] < cos(183)
                count_bal += 1
            else:
                count_bal = 0

            if count_bal < 10:  # Balance 제어로 넘어가는 조건: 177 ~ 183 각도 사이에 연속적으로 10번 이상
                exp = next(exp_source_iter)
                exp_queue.put(exp)
                exp_queue_balance.put(0)

                epsilon_tracker.udpate(step_idx)
                episode_rewards = experience_source.pop_episode_reward_lst()
            else:
                exp = next(exp_source_iter_balance)
                exp_queue.put(0)
                exp_queue_balance.put(exp)

                epsilon_tracker_balance.udpate(step_idx)
                episode_rewards = experience_source_balance.pop_episode_reward_lst()

            if episode_rewards:  # 에피소드가 종료될 때만 True
                current_episode_reward = episode_rewards[0]

                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    current_episode_reward, step_idx, epsilon=(action_selector.epsilon, action_selector_bal.epsilon)
                )

                model_save_condition = [
                    current_episode_reward > best_episode_reward,
                    step_idx > params.MAX_GLOBAL_STEPS / 4
                ]

                if all(model_save_condition):
                    rl_agent.save_actor_critic_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID,
                        actor_balance_net.__name__, actor_balance_net, critic_balance_net.__name__, critic_balance_net,
                        step_idx, current_episode_reward
                    )

                if current_episode_reward > best_episode_reward:
                    best_episode_reward = current_episode_reward

                if solved:
                    break

    exp_queue.put(None)
    exp_queue_balance.put(None)


def main():
    mp.set_start_method('spawn')
    env = MatlabRotaryInvertedPendulumEnv()
    # env = make_gym_env(params.ENVIRONMENT_ID.value, seed=params.SEED)
    print("env:", params.ENVIRONMENT_ID)
    print("observation_space:", 3)
    print("action_space:", 1)

    actor_net = policy_based_model.DDPGActor(
        obs_size=4,
        hidden_size_1=512, hidden_size_2=256,
        n_actions=1,
        scale=SWING_UP_SCALE_FACTOR
    ).to(device)

    critic_net = policy_based_model.DDPGCritic(
        obs_size=4,
        hidden_size_1=512, hidden_size_2=256,
        n_actions=1
    ).to(device)

    print(actor_net)
    print(critic_net)

    target_actor_net = rl_agent.TargetNet(actor_net)
    target_critic_net = rl_agent.TargetNet(critic_net)

    actor_optimizer = optim.Adam(actor_net.parameters(), lr=params.ACTOR_LEARNING_RATE)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=params.LEARNING_RATE)
    
###########################################################################################
    actor_balance_net = policy_based_model.DDPGActor(
        obs_size=4,
        hidden_size_1=512, hidden_size_2=256,
        n_actions=1,
        scale=BALANCING_SCALE_FACTOR
    ).to(device)
    critic_balance_net = policy_based_model.DDPGCritic(
        obs_size=4,
        hidden_size_1=512, hidden_size_2=256,
        n_actions=1
    ).to(device)

    target_actor_balance_net = rl_agent.TargetNet(actor_balance_net)
    target_critic_balance_net = rl_agent.TargetNet(critic_balance_net)

    actor_balance_optimizer = optim.Adam(actor_balance_net.parameters(), lr=params.ACTOR_LEARNING_RATE)
    critic_balance_optimizer = optim.Adam(critic_balance_net.parameters(), lr=params.LEARNING_RATE)
##########################################################################################

    buffer = experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE)
    buffer_balance = experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE)

    # buffer = experience.PrioritizedReplayBuffer(
    #     experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.N_STEP
    # )

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    exp_queue_balance = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)

    play_proc = mp.Process(target=play_func, args=(
        exp_queue, exp_queue_balance, env, actor_net, critic_net, actor_balance_net, critic_balance_net
    ))
    play_proc.start()

    time.sleep(0.5)

    if params.DRAW_VIZ:
        #stat_for_ddpg = statistics.StatisticsForDDPGOptimization(n_actions=1)
        stat_for_ddpg = statistics.StatisticsForSimpleDDPGOptimization(n_actions=1)
    else:
        stat_for_ddpg = None

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

########################################################################################
    actor_balance_grad_l2 = 0.0
    actor_balance_grad_max = 0.0
    actor_balance_grad_variance = 0.0

    critic_balance_grad_l2 = 0.0
    critic_balance_grad_max = 0.0
    critic_balance_grad_variance = 0.0

    loss_balance_actor = 0.0
    loss_balance_critic = 0.0
    loss_balance_total = 0.0

    count_bal = 0
########################################################################################

    #$ pip install line_profiler
    # from line_profiler import LineProfiler
    # lp = LineProfiler()
    # lp_wrapper = lp(model_update)

    while play_proc.is_alive():
        step_idx += params.N_STEP # 4, 8, 12, 16

        if step_idx % params.TRAIN_STEP_FREQ:
            continue

        exp = exp_queue.get()
        exp_balance = exp_queue_balance.get()

        if exp is None and exp_balance is None:
            play_proc.join()
            break

        if exp != 0:
            buffer._add(exp)

        if exp_balance != 0:
            buffer_balance._add(exp_balance)

        if step_idx % params.DRAW_VIZ_PERIOD_STEPS == 0:
            if params.DRAW_VIZ:
                # stat_for_ddpg.draw_optimization_performance(
                #     step_idx,
                #     loss_actor, loss_critic, loss_total,
                #     actor_grad_l2, actor_grad_variance, actor_grad_max,
                #     critic_grad_l2, critic_grad_variance, critic_grad_max,
                #     buffer_length, exp.noise, exp.action
                # )
                # TODO: exp_balance.noise, exp_balance.action 정보 처리
                stat_for_ddpg.draw_optimization_performance(
                    step_idx, exp.noise, exp.action
                )
            else:
                # print("[{0:6}] noise: {1:7.4f}, action: {2:7.4f}, reward: {3:8}, loss_actor: {4:7.4f}, loss_critic: {5:7.4f}".format(
                #     step_idx, exp.noise[0], exp.action[0], exp.reward, loss_actor, loss_critic
                # ), end="\n")
                pass

        ## buffer를 통하여 경험 정보 가져와 모델 업데이트
        if exp and len(buffer) >= params.MIN_REPLAY_SIZE_FOR_TRAIN:
            actor_grad_l2, actor_grad_max, actor_grad_variance, critic_grad_l2, critic_grad_max, critic_grad_variance, \
            loss_actor, loss_critic, loss_total = model_update(
                buffer, actor_net, critic_net, target_actor_net, target_critic_net, actor_optimizer, critic_optimizer,
                step_idx, actor_grad_l2, actor_grad_max, actor_grad_variance,
                critic_grad_l2, critic_grad_max, critic_grad_variance,
                loss_actor, loss_critic, loss_total, per=False
            )

        ## buffer_balance를 통하여 경험 정보 가져와 모델 업데이트
        if exp_balance and len(buffer_balance) >= params.MIN_REPLAY_SIZE_FOR_TRAIN:
            print("Update Balance!!!")
            actor_balance_grad_l2, actor_balance_grad_max, actor_balance_grad_variance, critic_balance_grad_l2, \
            critic_balance_grad_max, critic_balance_grad_variance, loss_balance_actor, loss_balance_critic, \
            loss_balance_total = model_update(
                buffer_balance, actor_balance_net, critic_balance_net, target_actor_balance_net, target_critic_balance_net,
                actor_balance_optimizer, critic_balance_optimizer,
                step_idx, actor_balance_grad_l2, actor_balance_grad_max, actor_balance_grad_variance,
                critic_balance_grad_l2, critic_balance_grad_max, critic_balance_grad_variance,
                loss_balance_actor, loss_balance_critic, loss_balance_total, per=False
            )


def model_update(buffer, actor_net, critic_net, target_actor_net, target_critic_net, actor_optimizer, critic_optimizer,
                 step_idx, actor_grad_l2, actor_grad_max, actor_grad_variance,
                 critic_grad_l2, critic_grad_max, critic_grad_variance,
                 loss_actor, loss_critic, loss_total, per):
    if per:
        batch, batch_indices, batch_weights = buffer.sample(params.BATCH_SIZE)
    else:
        batch = buffer.sample(params.BATCH_SIZE)
        batch_indices, batch_weights = None, None


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
        loss_critic_v = batch_weights_v.detach() * batch_l1_loss

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

    target_actor_net.alpha_sync(alpha=1 - 0.001)
    target_critic_net.alpha_sync(alpha=1 - 0.001)

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