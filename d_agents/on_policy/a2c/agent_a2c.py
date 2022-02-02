import copy

import torch.optim as optim
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp
import numpy as np

from c_models.e_a2c_models import ContinuousActorCriticModel, DiscreteActorCriticModel
from d_agents.agent import Agent
from g_utils.types import AgentMode


class AgentA2c(Agent):
    def __init__(self, observation_space, action_space, config):
        super(AgentA2c, self).__init__(observation_space, action_space, config)

        if isinstance(self.action_space, Discrete):
            self.actor_critic_model = DiscreteActorCriticModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, config=config
            )
        elif isinstance(self.action_space, Box):
            self.actor_critic_model = ContinuousActorCriticModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, config=config
            )
        else:
            raise ValueError()

        self.actor_model = self.actor_critic_model.actor_model
        self.critic_model = self.actor_critic_model.critic_model

        self.model = self.actor_model  # 에이전트 밖에서는 model이라는 이름으로 제어 모델 접근

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.config.LEARNING_RATE)

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)
        self.last_entropy = mp.Value('d', 0.0)

        self.step = 0

    def get_action(self, obs, mode=AgentMode.TRAIN):
        self.step += 1
        if isinstance(self.action_space, Discrete):
            action_prob = self.actor_model.pi(obs, save_hidden=True)

            if mode == AgentMode.TRAIN:
                dist = Categorical(probs=action_prob)
                action = dist.sample().detach().cpu().numpy()
            else:
                action = np.argmax(a=action_prob.detach().cpu().numpy(), axis=-1)
            return action

        elif isinstance(self.action_space, Box):
            mu_v, sigma_v = self.actor_model.pi(obs)

            if mode == AgentMode.TRAIN:
                # actions = np.random.normal(
                #     loc=mu_v.detach().cpu().numpy(), scale=torch.sqrt(var_v).detach().cpu().numpy()
                # )

                dist = Normal(loc=mu_v, scale=sigma_v)
                actions = dist.sample().detach().cpu().numpy()
            else:
                actions = mu_v.detach().cpu().numpy()

            actions = np.clip(a=actions, a_min=self.np_minus_ones, a_max=self.np_plus_ones)

            return actions
        else:
            raise ValueError()

    def get_td_target_values(self, next_observations, rewards, dones):
        with torch.no_grad():
            # values.shape: (32, 1), next_values.shape: (32, 1)
            next_values = self.critic_model.v(next_observations)
            next_values[dones] = 0.0

            # td_target_values.shape: (32, 1)
            td_target_values = rewards + (self.config.GAMMA ** self.config.N_STEP) * next_values
            # normalize td_target
            if self.config.TARGET_VALUE_NORMALIZE:
                td_target_values = (td_target_values - torch.mean(td_target_values)) / (torch.std(td_target_values) + 1e-7)

        return td_target_values.detach()

    def train_a2c(self):
        count_training_steps = 0

        #############################################
        #  Critic (Value) Loss 산출 & Update - BEGIN #
        #############################################

        td_target_values = self.get_td_target_values(self.next_observations, self.rewards, self.dones)

        # # next_values.shape: (32, 1)
        # next_values = self.critic_model.v(self.next_observations)
        # next_values[self.dones] = 0.0
        #
        # # td_target_values.shape: (32, 1)
        # td_target_values = self.rewards + (self.config.GAMMA ** self.config.N_STEP) * next_values
        # # normalize td_target
        # if self.config.TARGET_VALUE_NORMALIZE:
        #     td_target_values = (td_target_values - torch.mean(td_target_values)) / (torch.std(td_target_values) + 1e-7)

        # values.shape: (32, 1)
        values = self.critic_model.v(self.observations)
        # # loss_critic.shape: (,) <--  값 1개

        critic_loss = self.config.LOSS_FUNCTION(values, td_target_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params_list)
        self.critic_optimizer.step()
        ##########################################
        #  Critic (Value) Loss 산출 & Update- END #
        ##########################################

        #########################################
        #  Actor Objective 산출 & Update - BEGIN #
        #########################################
        advantages = (td_target_values - values).detach()

        if isinstance(self.action_space, Discrete):
            action_probs = self.actor_model.pi(self.observations)
            dist = Categorical(probs=action_probs)

            # action_probs.shape: (32, 2)
            # actions.shape: (32, 1)
            # advantage.shape: (32, 1)
            # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
            # criticized_log_pi_action_v.shape: (32,)
            criticized_log_pi_action_v = dist.log_prob(value=self.actions.squeeze(dim=-1)) * advantages.squeeze(dim=-1)

            entropy = dist.entropy().mean()
        elif isinstance(self.action_space, Box):
            mu_v, sigma_v = self.actor_model.pi(self.observations)

            # criticized_log_pi_action_v = self.calc_log_prob(mu_v, var_v, self.actions) * advantages
            # entropy = 0.5 * (torch.log(2.0 * np.pi * var_v) + 1.0).sum(dim=-1)
            # entropy = entropy.mean()

            dist = Normal(loc=mu_v, scale=sigma_v)
            criticized_log_pi_action_v = dist.log_prob(value=self.actions).sum(dim=-1, keepdim=True) * advantages
            entropy = dist.entropy().mean()
        else:
            raise ValueError()

        # actor_objective.shape: (,) <--  값 1개
        actor_objective = torch.mean(criticized_log_pi_action_v)

        # if self.step % 1000 == 0:
        #     print("mu_v:", mu_v, "var_v:", var_v)
        #     print("actor_objective:", actor_objective)
        #     print("td_target_values:", td_target_values)
        #     print("values:", values)
        #     print("advantages:", advantages)
        #     print("self.actions:", self.actions)
        #     print("dist.log_prob(value=self.actions).sum(dim=-1, keepdim=True):", dist.log_prob(value=self.actions).sum(dim=-1, keepdim=True))

        actor_loss = -1.0 * actor_objective
        entropy_loss = -1.0 * entropy
        actor_loss = actor_loss + entropy_loss * self.config.ENTROPY_BETA

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params_list)
        self.actor_optimizer.step()
        #######################################
        #  Actor Objective 산출 & Update - END #
        #######################################

        self.last_critic_loss.value = critic_loss.item()
        self.last_actor_objective.value = actor_objective.item()
        self.last_entropy.value = entropy.item()

        count_training_steps += 1

        return count_training_steps
