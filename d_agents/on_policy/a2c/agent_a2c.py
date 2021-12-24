import torch.optim as optim
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np

from c_models.d_actor_critic_models import ActorCritic, ContinuousActorCritic
from d_agents.agent import Agent
from g_utils.types import AgentMode


class AgentA2c(Agent):
    def __init__(self, observation_space, action_space, device, parameter):
        super(AgentA2c, self).__init__(observation_space, action_space, device, parameter)

        if isinstance(self.action_space, Discrete):
            self.actor_critic_model = ActorCritic(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            ).to(device)
        elif isinstance(self.action_space, Box):
            self.actor_critic_model = ContinuousActorCritic(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            ).to(device)
        else:
            raise ValueError()

        self.actor_critic_model.share_memory()

        self.optimizer = optim.Adam(
            self.actor_critic_model.parameters(), lr=self.parameter.LEARNING_RATE
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
            mu_v, logstd_v = self.actor_critic_model.pi(obs)
            if mode == AgentMode.TRAIN:
                dist = Normal(loc=mu_v, scale=torch.exp(logstd_v) + 1.0e-7)
                actions = dist.sample()
            else:
                actions = mu_v.detach()

            actions = np.clip(actions.cpu().numpy(), -1.0, 1.0)
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

        self.optimizer.zero_grad()

        ###################################
        #  Critic (Value) 손실 산출 - BEGIN #
        ###################################
        # next_values.shape: (32, 1)
        next_values = self.actor_critic_model.v(next_observations)
        td_target_value_lst = []

        for reward, next_value, done in zip(rewards, next_values, dones):
            td_target = reward + self.parameter.GAMMA * next_value * (0.0 if done else 1.0)
            td_target_value_lst.append(td_target)

        # td_target_values.shape: (32, 1)
        td_target_values = torch.tensor(td_target_value_lst, dtype=torch.float32, device=self.device).unsqueeze(dim=-1)

        # values.shape: (32, 1)
        values = self.actor_critic_model.v(observations)
        # loss_critic.shape: (,) <--  값 1개
        critic_loss = F.mse_loss(td_target_values.detach(), values)
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
            criticized_log_pi_action_v = torch.multiply(dist.log_prob(value=actions.squeeze(-1)), advantages.squeeze(-1))
        elif isinstance(self.action_space, Box):
            mu_v, logstd_v = self.actor_critic_model.pi(observations)
            dist = Normal(loc=mu_v, scale=torch.exp(logstd_v))

            # actions.shape: (32, 8)
            # dist.log_prob(value=actions).shape: (32, 8)
            # advantages.shape: (32, 1)
            # criticized_log_pi_action_v.shape: (32, 8)
            criticized_log_pi_action_v = torch.multiply(dist.log_prob(value=actions), advantages)
        else:
            raise ValueError()

        # actor_objective.shape: (,) <--  값 1개
        log_actor_objective = torch.sum(criticized_log_pi_action_v)
        actor_loss = torch.multiply(log_actor_objective, -1.0)
        ##############################
        #  Actor Objective 산출 - END #
        ##############################

        entropy_v = dist.entropy()
        entropy_loss = -1.0 * entropy_v.mean()

        loss = actor_loss + critic_loss * 0.5 + entropy_loss * 0.01

        loss.backward()
        self.optimizer.step()

        self.last_critic_loss.value = critic_loss.item()
        self.last_log_actor_objective.value = log_actor_objective.item()
