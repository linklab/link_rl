from abc import abstractmethod

import torch
from gym.spaces import Discrete, Box
import numpy as np

from link_rl.d_agents.agent import Agent
from link_rl.g_utils.buffers.buffer import Buffer
from link_rl.g_utils.types import AgentMode, AgentType, OnPolicyAgentTypes, ActorCriticAgentTypes


class OnPolicyAgent(Agent):
    def __init__(self, observation_space, action_space, config):
        super(OnPolicyAgent, self).__init__(observation_space, action_space, config)
        assert self.config.AGENT_TYPE in OnPolicyAgentTypes
        assert self.config.USE_PER is False

        self.buffer = Buffer(observation_space=observation_space, action_space=action_space, config=self.config)

    def _before_train(self):
        if self.config.AGENT_TYPE in ActorCriticAgentTypes:
            assert self.actor_model
            assert self.critic_model
            assert self.model is self.actor_model

        transition = self.buffer.sample(batch_size=None)
        self.observations, self.actions, self.next_observations, self.rewards, self.dones, self.infos = transition

    def train(self, training_steps_v=None):
        count_training_steps = 0

        if self.config.AGENT_TYPE == AgentType.REINFORCE:
            if len(self.buffer) > 0:
                self._before_train()  # sample all in order as it is
                count_training_steps = self.train_reinforce()
                self.buffer.clear()  # ON_POLICY!
                self._after_train()

        elif self.config.AGENT_TYPE == AgentType.A2C:
            if len(self.buffer) >= self.config.BATCH_SIZE:
                self._before_train()
                assert len(self.observations) == self.config.BATCH_SIZE
                count_training_steps = self.train_a2c()
                self.buffer.clear()  # ON_POLICY!
                self._after_train()

        elif self.config.AGENT_TYPE == AgentType.A3C:
            if len(self.buffer) >= self.config.BATCH_SIZE:
                self._before_train()
                assert len(self.observations) == self.config.BATCH_SIZE
                count_training_steps = self.train_a3c()
                self.buffer.clear()  # ON_POLICY!
                self._after_train()

        elif self.config.AGENT_TYPE == AgentType.PPO:
            if len(self.buffer) >= self.config.BATCH_SIZE:
                self._before_train()
                assert len(self.observations) == self.config.BATCH_SIZE
                count_training_steps = self.train_ppo()
                self.buffer.clear()  # ON_POLICY!
                self._after_train()

        elif self.config.AGENT_TYPE == AgentType.PPO_TRAJECTORY:
            if len(self.buffer) >= self.config.PPO_TRAJECTORY_SIZE:
                self._before_train()
                assert len(self.observations) == self.config.PPO_TRAJECTORY_SIZE
                count_training_steps = self.train_ppo()
                self.buffer.clear()  # ON_POLICY!
                self._after_train()

        else:
            raise ValueError()

        return count_training_steps

    def _after_train(self, loss_each=None):
        del self.observations
        del self.actions
        del self.next_observations
        del self.rewards
        del self.dones

    # ON_POLICY
    @abstractmethod
    def train_reinforce(self):
        raise NotImplementedError()

    @abstractmethod
    def train_a2c(self):
        raise NotImplementedError()

    @abstractmethod
    def train_a3c(self):
        raise NotImplementedError()

    @abstractmethod
    def train_ppo(self):
        raise NotImplementedError()

    @abstractmethod
    def train_asynchronous_ppo(self):
        raise NotImplementedError()

    def get_action(self, obs, mode=AgentMode.TRAIN):
        self.step += 1
        if isinstance(self.action_space, Discrete):
            action_prob = self.actor_model.pi(obs, save_hidden=True)

            if mode == AgentMode.TRAIN:
                action = np.random.choice(
                    a=self.n_discrete_actions, size=self.n_out_actions, p=action_prob[0].detach().cpu().numpy()
                )

                # dist = Categorical(probs=action_prob)
                # action = dist.sample().detach().cpu().numpy()
            else:
                action = np.argmax(a=action_prob.detach().cpu().numpy(), axis=-1)
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

    def get_returns(self):
        G = 0
        return_lst = []
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = reward + (self.config.GAMMA ** self.config.N_STEP) * G
            return_lst.append(G)

        returns = torch.tensor(return_lst[::-1], dtype=torch.float32, device=self.config.DEVICE).detach()
        return returns

    def get_target_values_and_advantages(self):
        combined_observations = torch.vstack([self.observations, self.next_observations[-1:]])
        combined_values = self.critic_model.v(combined_observations)

        # values.shape: (32, 1), next_values.shape: (32, 1)
        values = combined_values[:-1]
        next_values = combined_values[1:]
        next_values[self.dones] = 0.0

        if self.config.USE_GAE:
            assert self.config.N_STEP == 1

            target_values = self.rewards + self.config.GAMMA * next_values

            # generalized advantage estimator (gae): smoothed version of the advantage
            # by trajectory calculate advantage and 1-step target action value
            assert target_values.shape == values.shape, "{0} {1}".format(target_values.shape, values.shape)
            deltas = target_values - values

            last_gae = 0.0
            advantages = []
            for delta in reversed(deltas):
                last_gae = delta + self.config.GAMMA * self.config.GAE_LAMBDA * last_gae
                advantages.append(last_gae)

            advantages = torch.tensor(advantages[::-1], dtype=torch.float32, device=self.config.DEVICE).unsqueeze(dim=-1)

            if self.config.USE_GAE_RECALCULATE_TARGET_VALUE:
                target_values = advantages + values

            # target_values.shape: (256, 1)
            if self.config.TARGET_VALUE_NORMALIZE:
                target_values = (target_values - torch.mean(target_values)) / (torch.std(target_values) + 1e-7)
        else:
            if self.config.USE_BOOTSTRAP_FOR_TARGET_VALUE:
                target_values = self.rewards + (self.config.GAMMA ** self.config.N_STEP) * next_values
            else:
                target_values = self.get_returns().unsqueeze(dim=-1)

            # target_values.shape: (32, 1)
            # normalize td_target
            if self.config.TARGET_VALUE_NORMALIZE:
                target_values = (target_values - torch.mean(target_values)) / (torch.std(target_values) + 1e-7)

            assert target_values.shape == values.shape, "{0} {1}".format(target_values.shape, values.shape)
            advantages = (target_values - values).detach()

        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-7)

        return values, target_values.detach(), advantages.detach()

    def calc_log_prob(self, mu_v, var_v, actions_v):
        p1 = -0.5 * ((actions_v - mu_v) ** 2) / (var_v.clamp(min=1e-03))
        # p1 = -1.0 * ((mu_v - actions_v) ** 2) / (2.0 * var_v.clamp(min=1e-3))
        p2 = -0.5 * torch.log(2 * np.pi * var_v)

        log_prob = (p1 + p2).sum(dim=-1, keepdim=True)
        return log_prob