import copy
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp

from d_agents.on_policy.a2c.agent_a2c import AgentA2c


class AgentPpo(AgentA2c):
    def __init__(self, observation_space, action_space, config):
        super(AgentPpo, self).__init__(observation_space, action_space, config)

        self.actor_old_model = copy.deepcopy(self.actor_model)

        self.actor_old_model.share_memory()

        self.last_actor_objective = mp.Value('d', 0.0)
        self.last_ratio = mp.Value('d', 0.0)

    def train_ppo(self):
        count_training_steps = 0

        #####################################
        # OLD_LOG_PI 처리: BEGIN
        #####################################
        if isinstance(self.action_space, Discrete):
            action_probs = self.actor_old_model.pi(self.observations)
            dist = Categorical(probs=action_probs)
            old_log_pi_action_v = dist.log_prob(value=self.actions.squeeze(dim=-1))
        elif isinstance(self.action_space, Box):
            mu_v, sigma_v = self.actor_old_model.pi(self.observations)
            dist = Normal(loc=mu_v, scale=sigma_v)
            old_log_pi_action_v = dist.log_prob(value=self.actions).sum(dim=-1, keepdim=True)
        else:
            raise ValueError()

        #####################################
        # OLD_LOG_PI 처리: END
        #####################################

        sum_critic_loss = 0.0
        sum_actor_objective = 0.0
        sum_ratio = 0.0
        sum_entropy = 0.0

        for _ in range(self.config.PPO_K_EPOCH):
            #############################################
            #  Critic (Value) Loss 산출 & Update - BEGIN #
            #############################################

            batch_td_target_values = self.get_normalized_td_target_values(self.next_observations, self.rewards, self.dones)

            # batch_next_values = self.critic_model.v(self.next_observations)
            # batch_next_values[self.dones] = 0.0
            #
            # # td_target_values.shape: (32, 1)
            # batch_td_target_values = self.rewards + self.config.GAMMA ** self.config.N_STEP * batch_next_values
            # # normalize td_target_value
            # batch_td_target_values = (batch_td_target_values - torch.mean(batch_td_target_values)) / (torch.std(batch_td_target_values) + 1e-7)

            batch_values = self.critic_model.v(self.observations)

            assert batch_values.shape == batch_td_target_values.shape
            batch_critic_loss = self.config.LOSS_FUNCTION(batch_values, batch_td_target_values.detach())

            self.critic_optimizer.zero_grad()
            batch_critic_loss.backward()
            self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params_list)
            self.critic_optimizer.step()
            ##########################################
            #  Critic (Value) Loss 산출 & Update- END #
            ##########################################

            #########################################
            #  Actor Objective 산출 & Update - BEGIN #
            #########################################
            batch_advantages = (batch_td_target_values - batch_values).detach()

            if isinstance(self.action_space, Discrete):
                # actions.shape: (32, 1)
                # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
                # criticized_log_pi_action_v.shape: (32,)
                batch_action_probs = self.actor_model.pi(self.observations)
                batch_dist = Categorical(probs=batch_action_probs)
                batch_log_pi_action_v = batch_dist.log_prob(value=self.actions.squeeze(dim=-1))
                batch_entropy = batch_dist.entropy().mean()

                batch_advantages = batch_advantages.squeeze(dim=-1)  # NOTE

            elif isinstance(self.action_space, Box):
                batch_mu_v, batch_sigma_v = self.actor_model.pi(self.observations)

                # batch_log_pi_action_v = self.calc_log_prob(batch_mu_v, batch_var_v, batch_actions)
                # batch_entropy = 0.5 * (torch.log(2.0 * np.pi * batch_var_v) + 1.0).sum(dim=-1)
                # batch_entropy = batch_entropy.mean()

                batch_dist = Normal(loc=batch_mu_v, scale=batch_sigma_v)
                batch_log_pi_action_v = batch_dist.log_prob(value=self.actions).sum(dim=-1, keepdim=True)
                batch_entropy = batch_dist.entropy().mean()

            else:
                raise ValueError()

            batch_ratio = torch.exp(batch_log_pi_action_v - old_log_pi_action_v.detach())

            assert batch_ratio.shape == batch_advantages.shape, "{0}, {1}".format(
                batch_ratio.shape, batch_advantages.shape
            )
            batch_surrogate_loss_pre_clip = batch_ratio * batch_advantages
            batch_surrogate_loss_clip = torch.clamp(
                batch_ratio, 1.0 - self.config.PPO_EPSILON_CLIP, 1.0 + self.config.PPO_EPSILON_CLIP
            ) * batch_advantages

            assert batch_surrogate_loss_clip.shape == batch_surrogate_loss_pre_clip.shape, "".format(
                batch_surrogate_loss_clip.shape, batch_surrogate_loss_pre_clip.shape
            )
            batch_actor_objective = torch.mean(torch.min(batch_surrogate_loss_pre_clip, batch_surrogate_loss_clip))
            batch_actor_loss = -1.0 * batch_actor_objective
            batch_entropy_loss = -1.0 * batch_entropy
            batch_actor_loss = batch_actor_loss + batch_entropy_loss * self.config.ENTROPY_BETA

            self.actor_optimizer.zero_grad()
            batch_actor_loss.backward()
            self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params_list)
            self.actor_optimizer.step()

            sum_critic_loss += batch_critic_loss.item()
            sum_actor_objective += batch_actor_objective.item()
            sum_entropy += batch_entropy.item()
            sum_ratio += batch_ratio.mean().item()
            count_training_steps += 1

        #######################################
        #  Actor Objective 산출 & Update - END #
        #######################################

        self.synchronize_models(source_model=self.actor_model, target_model=self.actor_old_model)

        self.last_critic_loss.value = sum_critic_loss / count_training_steps
        self.last_actor_objective.value = sum_actor_objective / count_training_steps
        self.last_entropy.value = sum_entropy / count_training_steps
        self.last_ratio.value = sum_ratio / count_training_steps

        return count_training_steps
