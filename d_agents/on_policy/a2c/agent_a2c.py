import torch.optim as optim
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

from c_models.d_actor_critic_models import ContinuousActorCritic, DiscreteActorCritic
from d_agents.agent import Agent
from g_utils.types import AgentMode


class AgentA2c(Agent):
    def __init__(self, observation_space, action_space, device, parameter):
        super(AgentA2c, self).__init__(observation_space, action_space, device, parameter)

        if isinstance(self.action_space, Discrete):
            self.actor_critic_model = DiscreteActorCritic(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            ).to(device)
        elif isinstance(self.action_space, Box):
            self.action_bound_low = np.expand_dims(self.action_space.low, axis=0)
            self.action_bound_high = np.expand_dims(self.action_space.high, axis=0)

            self.action_scale_factor = np.max(np.maximum(
                np.absolute(self.action_bound_low), np.absolute(self.action_bound_high)
            ), axis=-1)[0]

            self.actor_critic_model = ContinuousActorCritic(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            ).to(device)
        else:
            raise ValueError()

        self.actor_critic_model.share_memory()

        self.actor_optimizer = optim.Adam(
            self.actor_critic_model.actor_params, lr=self.parameter.LEARNING_RATE
        )
        self.critic_optimizer = optim.Adam(
            self.actor_critic_model.critic_params, lr=self.parameter.LEARNING_RATE
        )

        self.model = self.actor_critic_model  # 에이전트 밖에서는 model이라는 이름으로 제어 모델 접근

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_log_actor_objective = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        if isinstance(self.action_space, Discrete):
            action_prob = self.actor_critic_model.pi(obs)
            m = Categorical(probs=action_prob)
            if mode == AgentMode.TRAIN:
                action = m.sample()
            else:
                action = torch.argmax(m.probs, dim=-1)
            return action.cpu().numpy()
        elif isinstance(self.action_space, Box):
            mu_v, std_v = self.actor_critic_model.pi(obs)
            mu_v = mu_v * self.action_scale_factor

            if mode == AgentMode.TRAIN:
                dist = Normal(loc=mu_v, scale=std_v)
                actions = dist.sample()
            else:
                actions = mu_v.detach()

            actions = np.clip(actions.cpu().numpy(), self.action_bound_low, self.action_bound_high)
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
        next_values = self.actor_critic_model.v(next_observations)
        td_target_value_lst = []

        for reward, next_value, done in zip(rewards, next_values, dones):
            td_target = reward + self.parameter.GAMMA ** self.parameter.N_STEP * next_value * (0.0 if done else 1.0)
            td_target_value_lst.append(td_target)

        # td_target_values.shape: (32, 1)
        td_target_values = torch.tensor(td_target_value_lst, dtype=torch.float32, device=self.device).unsqueeze(dim=-1)

        # values.shape: (32, 1)
        values = self.actor_critic_model.v(observations)
        # loss_critic.shape: (,) <--  값 1개
        critic_loss = F.mse_loss(td_target_values.detach(), values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor_critic_model.critic_params, self.parameter.CLIP_GRADIENT_VALUE)
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
            action_probs = self.actor_critic_model.pi(observations)
            dist = Categorical(probs=action_probs)

            # actions.shape: (32, 1)
            # advantage.shape: (32, 1)
            # dist.log_prob(value=actions.squeeze(-1)).shape: (32,)
            # criticized_log_pi_action_v.shape: (32,)
            criticized_log_pi_action_v = dist.log_prob(value=actions.squeeze(-1)) * advantages.squeeze(-1)
        elif isinstance(self.action_space, Box):
            mu_v, std_v = self.actor_critic_model.pi(observations)
            dist = Normal(loc=mu_v, scale=std_v)

            # actions.shape: (32, 8)
            # dist.log_prob(value=actions).shape: (32, 8)
            # advantages.shape: (32, 1)
            # criticized_log_pi_action_v.shape: (32, 8)
            # print(dist.log_prob(value=actions).shape, advantages.shape, "!!!!!!")
            criticized_log_pi_action_v = dist.log_prob(value=actions) * advantages
        else:
            raise ValueError()

        # actor_objective.shape: (,) <--  값 1개
        log_actor_objective = torch.mean(criticized_log_pi_action_v)
        actor_loss = -1.0 * log_actor_objective

        entropy_loss = -1.0 * torch.mean(dist.entropy())

        actor_loss = actor_loss + entropy_loss * self.parameter.ENTROPY_BETA

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor_critic_model.actor_params, self.parameter.CLIP_GRADIENT_VALUE)
        self.actor_optimizer.step()
        ##############################
        #  Actor Objective 산출 - END #
        ##############################

        self.last_critic_loss.value = critic_loss.item()
        self.last_log_actor_objective.value = log_actor_objective.item()
