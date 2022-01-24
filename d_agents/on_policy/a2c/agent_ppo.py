import copy

import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp
import numpy as np

from d_agents.on_policy.a2c.agent_a2c import AgentA2c


class AgentPpo(AgentA2c):
    def __init__(self, observation_space, action_space, parameter):
        super(AgentPpo, self).__init__(observation_space, action_space, parameter)

        self.actor_old_model = copy.deepcopy(self.actor_critic_model.actor_model)

        self.actor_old_model.share_memory()

        self.last_actor_objective = mp.Value('d', 0.0)
        self.last_ratio = mp.Value('d', 0.0)

    def train_ppo(self):
        count_training_steps = 0

        #####################################
        # Trajectory 처리: BEGIN
        #####################################
        trajectory_next_values = self.critic_model.v(self.next_observations)
        trajectory_next_values[self.dones] = 0.0

        # td_target_values.shape: (32, 1)
        trajectory_td_target_values = self.rewards + self.parameter.GAMMA ** self.parameter.N_STEP * trajectory_next_values
        trajectory_values = self.critic_model.v(self.observations)
        #
        trajectory_advantages = (trajectory_td_target_values - trajectory_values).detach()

        if isinstance(self.action_space, Discrete):
            trajectory_action_probs = self.actor_old_model.pi(self.observations)
            trajectory_dist = Categorical(probs=trajectory_action_probs)
            trajectory_old_log_pi_action_v = trajectory_dist.log_prob(value=self.actions.squeeze(dim=-1))
            trajectory_advantages = trajectory_advantages.squeeze(dim=-1)  # NOTE
        elif isinstance(self.action_space, Box):
            trajectory_mu_v, trajectory_var_v = self.actor_old_model.pi(self.observations)
            trajectory_old_log_pi_action_v = self.calc_log_prob(trajectory_mu_v, trajectory_var_v, self.actions)
        else:
            raise ValueError()

        #####################################
        # Trajectory 처리: END
        #####################################

        sum_critic_loss = 0.0
        sum_actor_objective = 0.0
        sum_ratio = 0.0
        sum_entropy = 0.0

        for _ in range(self.parameter.PPO_K_EPOCH):
            for batch_offset in range(0, self.parameter.PPO_TRAJECTORY_SIZE, self.parameter.BATCH_SIZE):
                batch_l = batch_offset + self.parameter.BATCH_SIZE

                batch_observations = self.observations[batch_offset:batch_l]
                batch_actions = self.actions[batch_offset:batch_l]
                batch_td_target_values = trajectory_td_target_values[batch_offset:batch_l]
                batch_old_log_pi_action_v = trajectory_old_log_pi_action_v[batch_offset:batch_l]
                batch_advantages = trajectory_advantages[batch_offset:batch_l]

                if torch.any(torch.isnan(batch_advantages)):
                    print(trajectory_advantages, batch_advantages)
                    raise ValueError()

                batch_values = self.critic_model.v(batch_observations)

                assert batch_values.shape == batch_td_target_values.shape
                batch_critic_loss = self.parameter.LOSS_FUNCTION(batch_values, batch_td_target_values.detach())

                self.critic_optimizer.zero_grad()
                batch_critic_loss.backward()
                self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params)
                self.critic_optimizer.step()

                if isinstance(self.action_space, Discrete):
                    # actions.shape: (32, 1)
                    # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
                    # criticized_log_pi_action_v.shape: (32,)
                    batch_action_probs = self.actor_model.pi(batch_observations)
                    batch_dist = Categorical(probs=batch_action_probs)
                    batch_log_pi_action_v = batch_dist.log_prob(value=batch_actions.squeeze(dim=-1))
                    batch_entropy = batch_dist.entropy().mean()
                elif isinstance(self.action_space, Box):
                    batch_mu_v, batch_var_v = self.actor_model.pi(batch_observations)
                    batch_log_pi_action_v = self.calc_log_prob(batch_mu_v, batch_var_v, batch_actions)
                    batch_entropy = 0.5 * (torch.log(2.0 * np.pi * batch_var_v) + 1.0).sum(dim=-1)
                    batch_entropy = batch_entropy.mean()
                else:
                    raise ValueError()

                batch_ratio = torch.exp(batch_log_pi_action_v - batch_old_log_pi_action_v.detach())

                assert batch_ratio.shape == batch_advantages.shape, "{0}, {1}".format(
                    batch_ratio.shape, batch_advantages.shape
                )
                batch_surrogate_loss_pre_clip = batch_ratio * batch_advantages
                batch_surrogate_loss_clip = torch.clamp(
                    batch_ratio, 1.0 - self.parameter.PPO_EPSILON_CLIP, 1.0 + self.parameter.PPO_EPSILON_CLIP
                ) * batch_advantages

                assert batch_surrogate_loss_clip.shape == batch_surrogate_loss_pre_clip.shape, "".format(
                    batch_surrogate_loss_clip.shape, batch_surrogate_loss_pre_clip.shape
                )
                batch_actor_objective = torch.mean(torch.min(batch_surrogate_loss_pre_clip, batch_surrogate_loss_clip))
                batch_actor_loss = -1.0 * batch_actor_objective
                batch_entropy_loss = -1.0 * batch_entropy
                batch_actor_loss = batch_actor_loss + batch_entropy_loss * self.parameter.ENTROPY_BETA

                self.actor_optimizer.zero_grad()
                batch_actor_loss.backward()
                self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params)
                self.actor_optimizer.step()

                sum_critic_loss += batch_critic_loss.item()
                sum_actor_objective += batch_actor_objective.item()
                sum_entropy += batch_entropy.item()
                sum_ratio += batch_ratio.mean().item()
                count_training_steps += 1

        ##############################
        #  Actor Objective 산출 - END #
        ##############################

        self.synchronize_models(source_model=self.actor_model, target_model=self.actor_old_model)

        self.last_critic_loss.value = sum_critic_loss / count_training_steps
        self.last_actor_objective.value = sum_actor_objective / count_training_steps
        self.last_entropy.value = sum_entropy / count_training_steps
        self.last_ratio.value = sum_ratio / count_training_steps

        return count_training_steps
