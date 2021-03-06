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

from codes.e_utils import rl_utils

current_path = os.path.dirname(os.path.realpath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(current_path, os.pardir, os.pardir))

if PROJECT_HOME not in sys.path:
    sys.path.append(PROJECT_HOME)

from common.environments import MatlabRotaryInvertedPendulumEnv
from common.fast_rl.common.utils import RewardTracker
from codes.e_utils.names import DeepLearningModelName
from codes.f_utils.common_utils import smooth
from common.fast_rl.policy_based_model import unpack_batch_for_ddpg
from common.fast_rl.rl_agent import float32_preprocessor

print("PyTorch Version", torch.__version__)

from common.fast_rl import actions, experience, policy_based_model, rl_agent, experience_single
from common.fast_rl.common import statistics

from codes.a_config.parameters import PARAMETERS as params

MODEL_SAVE_DIR = os.path.join(PROJECT_HOME, "out", "model_save_files")
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if params.TEAMVIEWER:
    ACTION_SCALE_FACTOR = 0.05
elif params.CH:
    ACTION_SCALE_FACTOR = 0.025
else:
    ACTION_SCALE_FACTOR = 0.035
CLIP = 1

env = rl_utils.get_environment(owner="worker", params=params)
print("env:", params.ENVIRONMENT_ID)
print("observation_space:", env.observation_space)
print("action_space:", env.action_space)

OBS_SIZE = env.observation_space.shape[0]


def play_func(exp_queue, actor_net, critic_net):
    action_selector = actions.EpsilonGreedyDDPGActionSelector(
        epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=ACTION_SCALE_FACTOR
    )

    epsilon_tracker = actions.EpsilonTracker(
        action_selector=action_selector,
        eps_start=params.EPSILON_INIT,
        eps_final=params.EPSILON_MIN,
        eps_frames=params.EPSILON_MIN_STEP
    )

    agent = rl_agent.AgentDDPG(
        actor_net, n_actions=1, action_selector=action_selector,
        action_min=ACTION_SCALE_FACTOR * -1.0, action_max=ACTION_SCALE_FACTOR,
        device=device, preprocessor=float32_preprocessor,
        name="One_AgentDDPG"
    )

    if params.DEEP_LEARNING_MODEL in [DeepLearningModelName.DETERMINISTIC_ACTOR_CRITIC_GRU, DeepLearningModelName.DETERMINISTIC_ACTOR_CRITIC_GRU_ATTENTION]:
        step_length = params.RNN_STEP_LENGTH
    else:
        step_length = -1

    experience_source = experience_single.ExperienceSourceSingleEnvFirstLast(
        env, agent, gamma=params.GAMMA, steps_count=params.N_STEP, step_length=step_length
    )

    exp_source_iter = iter(experience_source)

    if params.DRAW_VIZ:
        stat = statistics.StatisticsForPolicyBasedRL(method="policy_gradient")
    else:
        stat = None

    step_idx = 0

    best_mean_episode_reward = 0

    with RewardTracker(params=params, frame=False, stat=stat) as reward_tracker:
        while step_idx < params.MAX_GLOBAL_STEP:
            # 1 ?????? ???????????? exp??? exp_queue??? ??????
            step_idx += 1

            exp = next(exp_source_iter)

            exp_queue.put(exp)

            epsilon_tracker.udpate(step_idx)  #step_idx ??? epsilon ????????????

            episode_rewards = experience_source.pop_episode_reward_lst()

            if episode_rewards:  # ??????????????? ????????? ?????? True
                current_episode_reward = episode_rewards[0]

                solved, mean_episode_reward = reward_tracker.set_episode_reward(
                    current_episode_reward, step_idx, epsilon=action_selector.epsilon
                )

                model_save_condition = [
                    reward_tracker.mean_episode_reward > best_mean_episode_reward,
                    step_idx > params.EPSILON_MIN_STEP
                ]

                if reward_tracker.mean_episode_reward > best_mean_episode_reward:
                    best_mean_episode_reward = reward_tracker.mean_episode_reward

                if all(model_save_condition) or solved:
                    rl_agent.save_actor_critic_model(
                        MODEL_SAVE_DIR, params.ENVIRONMENT_ID,
                        actor_net.__name__, actor_net, critic_net.__name__, critic_net,
                        step_idx, reward_tracker.mean_episode_reward
                    )
                    if solved:
                        break

    exp_queue.put(None)


def main():
    mp.set_start_method('spawn')

    if params.DEEP_LEARNING_MODEL is DeepLearningModelName.DETERMINISTIC_ACTOR_CRITIC_MLP:
        actor_net = policy_based_model.DDPGActor(
            obs_size=OBS_SIZE,
            hidden_size_1=512, hidden_size_2=512,
            n_actions=1,
            scale=ACTION_SCALE_FACTOR
        ).to(device)

        critic_net = policy_based_model.DDPGCritic(
            obs_size=OBS_SIZE,
            hidden_size_1=512, hidden_size_2=512,
            n_actions=1
        ).to(device)
    elif params.DEEP_LEARNING_MODEL is DeepLearningModelName.DETERMINISTIC_ACTOR_CRITIC_GRU:
        actor_net = policy_based_model.DDPGGruActor(
            obs_size=OBS_SIZE,
            hidden_size_1=256, hidden_size_2=256,
            n_actions=1,
            bidirectional=False,
            scale=ACTION_SCALE_FACTOR
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

    buffer = experience.ExperienceReplayBuffer(experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE)

    # buffer = experience.PrioritizedReplayBuffer(
    #     experience_source=None, buffer_size=params.REPLAY_BUFFER_SIZE, n_step=params.N_STEP
    # )

    exp_queue = mp.Queue(maxsize=params.TRAIN_STEP_FREQ * 2)

    play_proc = mp.Process(target=play_func, args=(exp_queue, actor_net, critic_net))
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

    while play_proc.is_alive():
        step_idx += params.TRAIN_STEP_FREQ
        for _ in range(params.TRAIN_STEP_FREQ):
            exp = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            if exp != 0:
                buffer._add(exp)

        if step_idx % params.DRAW_VIZ_PERIOD_STEPS == 0:
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
            else:
                # print("[{0:6}] noise: {1:7.4f}, action: {2:7.4f}, reward: {3:8}, loss_actor: {4:7.4f}, loss_critic: {5:7.4f}".format(
                #     step_idx, exp.noise[0], exp.action[0], exp.reward, loss_actor, loss_critic
                # ), end="\n")
                pass

        ## buffer??? ????????? ?????? ?????? ????????? ?????? ????????????
        if exp and len(buffer) >= params.MIN_REPLAY_SIZE_FOR_TRAIN:
            actor_grad_l2, actor_grad_max, actor_grad_variance, critic_grad_l2, critic_grad_max, critic_grad_variance, \
            loss_actor, loss_critic, loss_total = model_update(
                buffer, actor_net, critic_net, target_actor_net, target_critic_net, actor_optimizer, critic_optimizer,
                step_idx, actor_grad_l2, actor_grad_max, actor_grad_variance,
                critic_grad_l2, critic_grad_max, critic_grad_variance,
                loss_actor, loss_critic, loss_total, per=False
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

    #########################
    # train critic: start   #
    #########################
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
    #########################
    # train critic: end   #
    #########################

    #########################
    # train actor: start    #
    #########################
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

    #########################
    # train actor: end      #
    #########################

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