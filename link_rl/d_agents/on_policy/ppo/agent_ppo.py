# https://github.com/ericyangyu/PPO-for-Beginners/blob/master/ppo.py
# https://github.com/medipixel/rl_algorithms
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp

from link_rl.d_agents.on_policy.a2c.agent_a2c import AgentA2c


class AgentPpo(AgentA2c):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentPpo, self).__init__(observation_space, action_space, config, need_train)

        self.last_actor_objective = mp.Value('d', 0.0)
        self.last_ratio = mp.Value('d', 0.0)

    def process_with_old_log_pi(self):
        #####################################
        # OLD_LOG_PI 처리: BEGIN
        #####################################
        if isinstance(self.action_space, Discrete):
            action_probs = self.actor_forward(self.observations)
            dist = Categorical(probs=action_probs)
            old_log_pi_action_v = dist.log_prob(value=self.actions.squeeze(dim=-1))
        elif isinstance(self.action_space, Box):
            mu_v, var_v = self.actor_forward(self.observations)
            dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
            old_log_pi_action_v = dist.log_prob(value=self.actions).sum(dim=-1)
        else:
            raise ValueError()

        return old_log_pi_action_v
        #####################################
        # OLD_LOG_PI 처리: END
        #####################################

    def get_ppo_actor_loss(self, old_log_pi_action_v, detached_advantages):
        if isinstance(self.action_space, Discrete):
            # actions.shape: (32, 1)
            # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
            # criticized_log_pi_action_v.shape: (32,)
            action_probs = self.actor_forward(self.observations)
            dist = Categorical(probs=action_probs)
            log_pi_action_v = dist.log_prob(value=self.actions.squeeze(dim=-1))
        elif isinstance(self.action_space, Box):
            mu_v, var_v = self.actor_forward(self.observations)

            # log_pi_action_v = self.calc_log_prob(mu_v, var_v, actions)
            # entropy = 0.5 * (torch.log(2.0 * np.pi * var_v) + 1.0).sum(dim=-1)
            # entropy = entropy.mean()

            dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
            log_pi_action_v = dist.log_prob(value=self.actions).sum(dim=-1)
        else:
            raise ValueError()

        entropy = dist.entropy().mean()
        ratio = torch.exp(log_pi_action_v - old_log_pi_action_v.detach())

        assert ratio.shape == detached_advantages.shape, "{0}, {1}".format(
            ratio.shape, detached_advantages.shape
        )
        surrogate_objective_1 = ratio * detached_advantages
        surrogate_objective_2 = torch.clamp(
            ratio, 1.0 - self.config.PPO_EPSILON_CLIP, 1.0 + self.config.PPO_EPSILON_CLIP
        ) * detached_advantages

        assert surrogate_objective_1.shape == surrogate_objective_2.shape, "{0} {1}".format(
            surrogate_objective_1.shape, surrogate_objective_2.shape
        )
        actor_objective = torch.mean(torch.min(surrogate_objective_1, surrogate_objective_2))
        actor_loss = -1.0 * actor_objective
        entropy_loss = -1.0 * entropy
        actor_loss = actor_loss + entropy_loss * self.config.ENTROPY_BETA
        return actor_loss, actor_objective, entropy, ratio

    def train_ppo(self):
        count_training_steps = 0

        old_log_pi_action_v = self.process_with_old_log_pi()

        sum_critic_loss = 0.0
        sum_actor_objective = 0.0
        sum_ratio = 0.0
        sum_entropy = 0.0

        _, detached_target_values, detached_advantages = self.get_target_values_and_advantages()
        detached_advantages = detached_advantages.squeeze(dim=-1)  # NOTE

        for _ in range(self.config.PPO_K_EPOCH):
            #############################################
            #  Critic (Value) Loss 산출 & Update - BEGIN #
            #############################################
            values = self.critic_forward(self.observations)
            critic_loss = self.get_critic_loss(values=values, detached_target_values=detached_target_values)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.clip_critic_model_parameter_grad_value(self.critic_model.parameters())
            self.critic_optimizer.step()

            if self.encoder_is_not_identity:
                self.last_loss_for_encoder = critic_loss

            ##########################################
            #  Critic (Value) Loss 산출 & Update- END #
            ##########################################

            #########################################
            #  Actor Objective 산출 & Update - BEGIN #
            #########################################
            actor_loss, actor_objective, entropy, ratio = self.get_ppo_actor_loss(
                old_log_pi_action_v=old_log_pi_action_v, detached_advantages=detached_advantages
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.clip_actor_model_parameter_grad_value(self.actor_model.parameters())
            self.actor_optimizer.step()

            sum_critic_loss += critic_loss.item()
            sum_actor_objective += actor_objective.item()
            sum_entropy += entropy.item()
            sum_ratio += ratio.mean().item()

            count_training_steps += 1
            #######################################
            #  Actor Objective 산출 & Update - END #
            #######################################

        self.last_critic_loss.value = sum_critic_loss / count_training_steps
        self.last_actor_objective.value = sum_actor_objective / count_training_steps
        self.last_entropy.value = sum_entropy / count_training_steps
        self.last_ratio.value = sum_ratio / count_training_steps

        return count_training_steps
