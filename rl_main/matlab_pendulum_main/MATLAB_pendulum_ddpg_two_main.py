# https://github.com/openai/gym/blob/master/gym/envs/classic_control/pendulum.py
# https://mspries.github.io/jimmy_pendulum.html
#!/usr/bin/env python3
import math
import profile
import time
from enum import Enum

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim
import os, sys
import numpy as np
import copy

from config.names import DeepLearningModelName
from rl_main.matlab_pendulum_main.experience_pendulum_ddpg_two import ExperienceSourceSingleEnvFirstLastDdpgTwo

idx = os.getcwd().index("link_rl")
PROJECT_HOME = os.getcwd()[:idx] + "link_rl"
sys.path.append(PROJECT_HOME)

from common.environments.matlab.matlabenv import MatlabRotaryInvertedPendulumEnv, Status

from common.common_utils import smooth
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


if params.TEAMVIEWER:
    SWING_UP_SCALE_FACTOR = 0.05
    BALANCING_SCALE_FACTOR = 0.0005

elif params.CH:
    SWING_UP_SCALE_FACTOR = 0.05
    BALANCING_SCALE_FACTOR = 0.0005
else:
    SWING_UP_SCALE_FACTOR = 0.035
    BALANCING_SCALE_FACTOR = 0.002
CLIP = 1

env = MatlabRotaryInvertedPendulumEnv(
    action_min=SWING_UP_SCALE_FACTOR * -1.0, action_max=SWING_UP_SCALE_FACTOR
)
print("env:", params.ENVIRONMENT_ID)
print("observation_space:", env.observation_space)
print("action_space:", env.action_space)

OBS_SIZE = env.observation_space.shape[0]


def play_func(exp_queue_swing_up, exp_queue_balancing, actor_net, critic_net, actor_balance_net, critic_balance_net):
    env.start()
    swing_up_action_min = -SWING_UP_SCALE_FACTOR
    swing_up_action_max = SWING_UP_SCALE_FACTOR
    balancing_action_min = -BALANCING_SCALE_FACTOR
    balancing_action_max = BALANCING_SCALE_FACTOR

    # action_selector = actions.EpsilonGreedyDDPGActionSelector(epsilon=params.EPSILON_INIT)

    action_selector_swing_up = actions.DDPGActionSelector(
        epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=SWING_UP_SCALE_FACTOR
    )

    epsilon_tracker_swing_up = actions.EpsilonTracker(
        action_selector=action_selector_swing_up,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )

    agent_swing_up = rl_agent.AgentDDPG(
        actor_net, n_actions=1, action_selector=action_selector_swing_up,
        action_min=swing_up_action_min, action_max=swing_up_action_max,
        device=device, preprocessor=float32_preprocessor,
        name="SwingUp_AgentDDPG"
    )


    action_selector_balancing = actions.DDPGActionSelector(
        epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=BALANCING_SCALE_FACTOR
    )

    epsilon_tracker_balancing = actions.EpsilonTracker(
        action_selector=action_selector_balancing,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )

    agent_balancing = rl_agent.AgentDDPG(
        actor_balance_net, n_actions=1, action_selector=action_selector_balancing,
        action_min=balancing_action_min, action_max=balancing_action_max, device=device,
        preprocessor=float32_preprocessor,
        name="Balance_AgentDDPG"
    )

    if params.DEEP_LEARNING_MODEL in [DeepLearningModelName.DDPG_GRU, DeepLearningModelName.DDPG_GRU_ATTENTION]:
        step_length = params.RNN_STEP_LENGTH
    else:
        step_length = -1

    experience_source = ExperienceSourceSingleEnvFirstLastDdpgTwo(
        params, env, agent_swing_up, agent_balancing, gamma=params.GAMMA, steps_count=params.N_STEP, step_length=step_length
    )

    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForPolicyBasedRL(method="policy_gradient")
    else:
        stat = None

    step_idx = 0
    swing_up_step_idx = 0
    balancing_step_idx = 0

    best_episode_reward = 0

    balancing_step_reward_list = []

    recent_swing_up_to_balance_exp = None

    with utils.RewardTracker(
            params.STOP_MEAN_EPISODE_REWARD, params.AVG_EPISODE_SIZE_FOR_STAT,
            frame=True, draw_viz=params.DRAW_VIZ, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEPS:
            # 1 스텝 진행하고 exp를 exp_queue에 넣음
            step_idx += 1

            exp = next(exp_source_iter)

            if params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_MLP:
                status_value = exp[0][-1]
            elif params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_GRU:
                status_value = exp[0][-1][-1]
            else:
                raise ValueError()

            if status_value == Status.SWING_UP.value: # SWING_UP: -1.0
                swing_up_step_idx += 1
                epsilon_tracker_swing_up.udpate(swing_up_step_idx)

                exp_queue_swing_up.put(exp)
                exp_queue_balancing.put(0)

            elif status_value == Status.SWING_UP_TO_BALANCING.value:  # SWING_UP_TO_BALANCING: 0.5
                swing_up_step_idx += 1
                epsilon_tracker_swing_up.udpate(swing_up_step_idx)

                # NOTE: exp 잠시 대기
                recent_swing_up_to_balance_exp = copy.deepcopy(exp)

            elif status_value == Status.BALANCING.value:  # BALANCING:1.0, BALANCING_TO_SWING_UP:-0.5
                balancing_step_idx += 1
                epsilon_tracker_balancing.udpate(balancing_step_idx)

                exp_queue_swing_up.put(0)
                exp_queue_balancing.put(exp)

                balancing_step_reward_list.append(exp.reward)

            elif status_value == Status.BALANCING_TO_SWING_UP.value:
                balancing_step_idx += 1
                epsilon_tracker_balancing.udpate(balancing_step_idx)

                exp_queue_balancing.put(exp)

                # NOTE: 대기 중인 exp의 reward를 수정하고 exp_queue_swing_up에 넣기
                recent_swing_up_to_balance_exp._replace(reward=sum(balancing_step_reward_list))
                exp_queue_swing_up.put(recent_swing_up_to_balance_exp)

                balancing_step_reward_list.clear()

            else:
                raise ValueError()

            episode_rewards = experience_source.pop_episode_reward_lst()

            if episode_rewards:  # 에피소드가 종료될 때만 True

                current_episode_reward = episode_rewards[0]

                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    current_episode_reward, step_idx,
                    epsilon=(action_selector_swing_up.epsilon, action_selector_balancing.epsilon)
                )

                model_save_condition = [
                    current_episode_reward > best_episode_reward,
                    step_idx > params.MAX_GLOBAL_STEPS / 4
                ]

                if all(model_save_condition):
                    rl_agent.save_actor_critic_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID,
                        actor_net.__name__, actor_net, critic_net.__name__, critic_net,
                        step_idx, current_episode_reward
                    )

                    rl_agent.save_actor_critic_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID,
                        actor_balance_net.__name__, actor_balance_net, critic_balance_net.__name__, critic_balance_net,
                        step_idx, current_episode_reward
                    )

                if current_episode_reward > best_episode_reward:
                    best_episode_reward = current_episode_reward

                if solved:
                    break

    exp_queue_swing_up.put(None)
    exp_queue_balancing.put(None)


def main():
    mp.set_start_method('spawn')

    ###########################
    ### SWING_UP Controller ###
    ###########################
    if params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_MLP:
        actor_net = policy_based_model.DDPGActor(
            obs_size=OBS_SIZE,
            hidden_size_1=512, hidden_size_2=512,
            n_actions=1,
            scale=SWING_UP_SCALE_FACTOR
        ).to(device)

        critic_net = policy_based_model.DDPGCritic(
            obs_size=OBS_SIZE,
            hidden_size_1=512, hidden_size_2=512,
            n_actions=1
        ).to(device)
    elif params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_GRU:
        actor_net = policy_based_model.DDPGGruActor(
            obs_size=OBS_SIZE,
            hidden_size_1=256, hidden_size_2=256,
            n_actions=1,
            bidirectional=False,
            scale=BALANCING_SCALE_FACTOR
        ).to(device)

        critic_net = policy_based_model.DDPGGruCritic(
            obs_size=OBS_SIZE,
            hidden_size_1=256, hidden_size_2=256,
            n_actions=1,
            bidirectional=False
        ).to(device)
    else:
        raise ValueError()

    print(actor_net)
    print(critic_net)

    target_actor_net = rl_agent.TargetNet(actor_net)
    target_critic_net = rl_agent.TargetNet(critic_net)

    actor_optimizer = optim.Adam(actor_net.parameters(), lr=params.ACTOR_LEARNING_RATE)
    critic_optimizer = optim.Adam(critic_net.parameters(), lr=params.LEARNING_RATE)

    ############################
    ### BALANCING Controller ###
    ############################
    if params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_MLP:
        actor_balance_net = policy_based_model.DDPGActor(
            obs_size=OBS_SIZE,
            hidden_size_1=512, hidden_size_2=512,
            n_actions=1,
            scale=BALANCING_SCALE_FACTOR
        ).to(device)

        critic_balance_net = policy_based_model.DDPGCritic(
            obs_size=OBS_SIZE,
            hidden_size_1=512, hidden_size_2=512,
            n_actions=1
        ).to(device)
    elif params.DEEP_LEARNING_MODEL is DeepLearningModelName.DDPG_GRU:
        actor_balance_net = policy_based_model.DDPGGruActor(
            obs_size=OBS_SIZE,
            hidden_size_1=256, hidden_size_2=256,
            n_actions=1,
            bidirectional=False,
            scale=BALANCING_SCALE_FACTOR
        ).to(device)

        critic_balance_net = policy_based_model.DDPGGruCritic(
            obs_size=OBS_SIZE,
            hidden_size_1=256, hidden_size_2=256,
            n_actions=1,
            bidirectional=False
        ).to(device)
    else:
        raise ValueError()

    target_actor_balance_net = rl_agent.TargetNet(actor_balance_net)
    target_critic_balance_net = rl_agent.TargetNet(critic_balance_net)

    actor_balance_optimizer = optim.Adam(actor_balance_net.parameters(), lr=params.ACTOR_LEARNING_RATE)
    critic_balance_optimizer = optim.Adam(critic_balance_net.parameters(), lr=params.LEARNING_RATE)
##########################################################################################

    buffer_swing_up = experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE)
    buffer_balancing = experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE)

    # buffer = experience.PrioritizedReplayBuffer(
    #     experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.N_STEP
    # )

    exp_queue_swing_up = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)
    exp_queue_balancing = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)

    play_proc = mp.Process(target=play_func, args=(
        exp_queue_swing_up, exp_queue_balancing, actor_net, critic_net, actor_balance_net, critic_balance_net
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

        exp_swing_up = exp_queue_swing_up.get()
        exp_balancing = exp_queue_balancing.get()

        if exp_swing_up is None and exp_balancing is None:
            play_proc.join()
            break

        if exp_swing_up != 0:
            buffer_swing_up._add(exp_swing_up)

        if exp_balancing != 0:
            buffer_balancing._add(exp_balancing)

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
                    step_idx, exp_swing_up.noise, exp_swing_up.action
                )
            else:
                # print("[{0:6}] noise: {1:7.4f}, action: {2:7.4f}, reward: {3:8}, loss_actor: {4:7.4f}, loss_critic: {5:7.4f}".format(
                #     step_idx, exp.noise[0], exp.action[0], exp.reward, loss_actor, loss_critic
                # ), end="\n")
                pass

        ## buffer를 통하여 경험 정보 가져와 모델 업데이트
        if exp_swing_up and len(buffer_swing_up) >= params.MIN_REPLAY_SIZE_FOR_TRAIN:
            actor_grad_l2, actor_grad_max, actor_grad_variance, critic_grad_l2, critic_grad_max, critic_grad_variance, \
            loss_actor, loss_critic, loss_total = model_update(
                buffer_swing_up, actor_net, critic_net, target_actor_net, target_critic_net, actor_optimizer, critic_optimizer,
                step_idx, actor_grad_l2, actor_grad_max, actor_grad_variance,
                critic_grad_l2, critic_grad_max, critic_grad_variance,
                loss_actor, loss_critic, loss_total, per=False
            )

        ## buffer_balance를 통하여 경험 정보 가져와 모델 업데이트
        if exp_balancing and len(buffer_balancing) >= params.MIN_REPLAY_SIZE_FOR_TRAIN:
            #print("Update Balance!!!")
            actor_balance_grad_l2, actor_balance_grad_max, actor_balance_grad_variance, critic_balance_grad_l2, \
            critic_balance_grad_max, critic_balance_grad_variance, loss_balance_actor, loss_balance_critic, \
            loss_balance_total = model_update(
                buffer_balancing, actor_balance_net, critic_balance_net, target_actor_balance_net, target_critic_balance_net,
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

    # clip the gradients to prevent the model from exploding gradient
    torch.nn.utils.clip_grad_norm_(critic_net.parameters(), CLIP)

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

    # clip the gradients to prevent the model from exploding gradient
    torch.nn.utils.clip_grad_norm_(actor_net.parameters(), CLIP)

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