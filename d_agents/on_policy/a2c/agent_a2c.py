import math

import torch.optim as optim
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

from c_models.d_actor_critic_models import ContinuousActorCriticModel, DiscreteActorCriticModel
from d_agents.agent import Agent
from g_utils.types import AgentMode


class AgentA2c(Agent):
    def __init__(self, observation_space, action_space, device, parameter):
        super(AgentA2c, self).__init__(observation_space, action_space, device, parameter)

        if isinstance(self.action_space, Discrete):
            self.actor_critic_model = DiscreteActorCriticModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
            )
        elif isinstance(self.action_space, Box):
            self.actor_critic_model = ContinuousActorCriticModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            )
        else:
            raise ValueError()

        self.actor_model = self.actor_critic_model.actor_model
        self.critic_model = self.actor_critic_model.critic_model

        self.model = self.actor_model  # 에이전트 밖에서는 model이라는 이름으로 제어 모델 접근

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.actor_optimizer = optim.Adam(self.actor_model.actor_params, lr=self.parameter.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.critic_params, lr=self.parameter.LEARNING_RATE)

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_log_actor_objective = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        if isinstance(self.action_space, Discrete):
            action_prob = self.actor_model.pi(obs)
            # m = Categorical(probs=action_prob)
            if mode == AgentMode.TRAIN:
                action = np.random.choice(a=self.n_discrete_actions, p=action_prob.detach().cpu().numpy())
                # action = m.sample()
            else:
                action = np.argmax(input=action_prob.detach().cpu().numpy(), dim=-1)
            return action
        elif isinstance(self.action_space, Box):
            mu_v, var_v = self.actor_model.pi(obs)

            if mode == AgentMode.TRAIN:
                actions = np.random.normal(
                    loc=mu_v.detach().cpu().numpy(), scale=torch.sqrt(var_v).detach().cpu().numpy()
                )
                # dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
                # actions = dist.sample().detach().cpu().numpy()
            else:
                actions = mu_v.detach().cpu().numpy()

            actions = np.clip(a=actions, a_min=self.np_minus_ones, a_max=self.np_plus_ones)

            return actions
        else:
            raise ValueError()

    def train_a2c(self):
        # observations.shape: torch.Size([32, 4, 84, 84]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4, 84, 84]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])

        observations, actions, next_observations, rewards, dones = self.buffer.sample(
            batch_size=self.parameter.BATCH_SIZE, device=self.device
        )

        ###################################
        #  Critic (Value) 손실 산출 - BEGIN #
        ###################################
        # next_values.shape: (32, 1)
        next_values = self.critic_model.v(next_observations)
        next_values[dones] = 0.0

        # td_target_values.shape: (32, 1)
        td_target_values = rewards + self.parameter.GAMMA ** self.parameter.N_STEP * next_values

        # values.shape: (32, 1)
        values = self.critic_model.v(observations)
        # loss_critic.shape: (,) <--  값 1개
        critic_loss = F.mse_loss(td_target_values.detach(), values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params)
        self.critic_optimizer.step()
        ###################################
        #  Critic (Value)  Loss 산출 - END #
        ###################################

        ################################
        #  Actor Objective 산출 - BEGIN #
        ################################
        q_values = td_target_values
        advantages = (q_values - values).detach()

        if isinstance(self.action_space, Discrete):
            action_probs = self.actor_model.pi(observations)
            dist = Categorical(probs=action_probs)

            # actions.shape: (32, 1)
            # advantage.shape: (32, 1)
            # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
            # criticized_log_pi_action_v.shape: (32,)
            criticized_log_pi_action_v = dist.log_prob(value=actions.squeeze(-1)) * advantages.squeeze(-1)
            entropy = None
        elif isinstance(self.action_space, Box):
            mu_v, var_v = self.actor_model.pi(observations)

            criticized_log_pi_action_v = self.calc_log_prob(mu_v, var_v, actions) * advantages
            entropy = 0.5 * (torch.log(2.0 * np.pi * var_v) + 1.0)

            # dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
            # criticized_log_pi_action_v = dist.log_prob(value=actions) * advantages
            # entropy = dist.entropy()

            # print(criticized_log_pi_action_v.shape, entropy.shape, "!!")
            # print(dist.log_prob(value=actions).shape, advantages.shape, "!!!!!!")
        else:
            raise ValueError()

        # actor_objective.shape: (,) <--  값 1개
        log_actor_objective = torch.mean(criticized_log_pi_action_v)
        actor_loss = -1.0 * log_actor_objective

        entropy_loss = -1.0 * torch.mean(entropy)

        actor_loss = actor_loss + entropy_loss * self.parameter.ENTROPY_BETA

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params)
        self.actor_optimizer.step()
        ##############################
        #  Actor Objective 산출 - END #
        ##############################

        self.last_critic_loss.value = critic_loss.item()
        self.last_log_actor_objective.value = log_actor_objective.item()