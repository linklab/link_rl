import math
from abc import abstractmethod, ABC

import torch
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box, MultiDiscrete

import numpy as np
from g_utils.buffers import Buffer
from g_utils.commons import get_continuous_action_info
from g_utils.prioritized_buffer import PrioritizedBuffer
from g_utils.types import AgentMode, AgentType, OnPolicyAgentTypes, ActorCriticAgentTypes, OffPolicyAgentTypes


class Agent:
    def __init__(self, observation_space, action_space, config):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

        # Box
        # Dict
        # Tuple
        # Discrete
        # MultiBinary
        # MultiDiscrete
        self.observation_shape = observation_space.shape
        self.action_shape = action_space.shape

        if isinstance(action_space, Discrete):
            self.n_discrete_actions = action_space.n
            self.n_out_actions = 1

            self.np_minus_ones = None
            self.np_plus_ones = None
            self.torch_minus_ones = None
            self.torch_plus_ones = None

            self.action_scale = None
            self.action_bias = None
        elif isinstance(action_space, Box):
            self.n_discrete_actions = None
            self.n_out_actions = action_space.shape[0]

            self.np_minus_ones = np.full(shape=action_space.shape, fill_value=-1.0)
            self.np_plus_ones = np.full(shape=action_space.shape, fill_value=1.0)
            self.torch_minus_ones = torch.full(size=action_space.shape, fill_value=-1.0).to(self.config.DEVICE)
            self.torch_plus_ones = torch.full(size=action_space.shape, fill_value=1.0).to(self.config.DEVICE)

            _, _, self.action_scale, self.action_bias = get_continuous_action_info(action_space)
        else:
            raise ValueError()

        if self.config.USE_PER:
            assert self.config.AGENT_TYPE in OffPolicyAgentTypes
            self.buffer = PrioritizedBuffer(action_space=action_space, config=self.config)
        else:
            self.buffer = Buffer(action_space=action_space, config=self.config)

        self.model = None
        if self.config.AGENT_TYPE in ActorCriticAgentTypes:
            self.actor_model = None
            self.critic_model = None

        self.last_model_grad_max = mp.Value('d', 0.0)
        self.last_model_grad_l2 = mp.Value('d', 0.0)

        self.last_actor_model_grad_max = mp.Value('d', 0.0)
        self.last_critic_model_grad_max = mp.Value('d', 0.0)

        self.last_actor_model_grad_l2 = mp.Value('d', 0.0)
        self.last_critic_model_grad_l2 = mp.Value('d', 0.0)

    @abstractmethod
    def get_action(self, obs, mode=AgentMode.TRAIN):
        pass

    def _before_train(self, sample_length):
        if self.config.AGENT_TYPE in ActorCriticAgentTypes:
            assert self.actor_model
            assert self.critic_model
            assert self.model is self.actor_model

        # [MLP]
        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        #
        # [CNN]
        # observations.shape: torch.Size([32, 4, 84, 84]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4, 84, 84]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        self.observations, self.actions, self.next_observations, self.rewards, self.dones = self.buffer.sample(
            batch_size=sample_length
        )

    def train(self, training_steps_v=None):
        count_training_steps = 0

        if self.config.AGENT_TYPE in OffPolicyAgentTypes:
            if self.config.AGENT_TYPE in (AgentType.DQN, AgentType.DUELING_DQN):
                if len(self.buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                    self._before_train(sample_length=self.config.BATCH_SIZE)
                    count_training_steps, q_net_loss_each = self.train_dqn(training_steps_v=training_steps_v)
                    self._after_train(q_net_loss_each)

            elif self.config.AGENT_TYPE in (AgentType.DOUBLE_DQN, AgentType.DOUBLE_DUELING_DQN):
                if len(self.buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                    self._before_train(sample_length=self.config.BATCH_SIZE)
                    count_training_steps, q_net_loss_each = self.train_double_dqn(training_steps_v=training_steps_v)
                    self._after_train(q_net_loss_each)

            elif self.config.AGENT_TYPE == AgentType.DDPG:
                if len(self.buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                    self._before_train(sample_length=self.config.BATCH_SIZE)
                    count_training_steps, critic_loss_each = self.train_ddpg()
                    self._after_train(critic_loss_each)

            elif self.config.AGENT_TYPE == AgentType.TD3:
                if len(self.buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                    self._before_train(sample_length=self.config.BATCH_SIZE)
                    count_training_steps, critic_loss_each = self.train_td3(training_steps_v=training_steps_v)
                    self._after_train(critic_loss_each)

            elif self.config.AGENT_TYPE == AgentType.SAC:
                if len(self.buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                    self._before_train(sample_length=self.config.BATCH_SIZE)
                    count_training_steps, critic_loss_each = self.train_sac(training_steps_v=training_steps_v)
                    self._after_train(critic_loss_each)

            else:
                raise ValueError()

        elif self.config.AGENT_TYPE in OnPolicyAgentTypes:
            if self.config.AGENT_TYPE == AgentType.REINFORCE:
                if len(self.buffer) > 0:
                    self._before_train(sample_length=None)  # sample all in order as it is
                    count_training_steps = self.train_reinforce()
                    self.buffer.clear()     # ON_POLICY!
                    self._after_train()

            elif self.config.AGENT_TYPE == AgentType.A2C:
                if len(self.buffer) >= self.config.BATCH_SIZE:
                    self._before_train(sample_length=self.config.BATCH_SIZE)
                    count_training_steps = self.train_a2c()
                    self.buffer.clear()                 # ON_POLICY!
                    self._after_train()

            elif self.config.AGENT_TYPE == AgentType.PPO:
                if len(self.buffer) >= self.config.BATCH_SIZE:
                    self._before_train(sample_length=self.config.BATCH_SIZE)
                    count_training_steps = self.train_ppo()
                    self.buffer.clear()                 # ON_POLICY!
                    self._after_train()

            elif self.config.AGENT_TYPE == AgentType.PPO_TRAJECTORY:
                if len(self.buffer) >= self.config.PPO_TRAJECTORY_SIZE:
                    self._before_train(sample_length=self.config.PPO_TRAJECTORY_SIZE)
                    count_training_steps = self.train_ppo()
                    self.buffer.clear()                 # ON_POLICY!
                    self._after_train()

            else:
                raise ValueError()

        else:
            raise ValueError()

        return count_training_steps

    def _after_train(self, loss_each=None):
        if loss_each is not None and self.config.USE_PER:
            self.buffer.update_priorities(loss_each.detach().cpu().numpy())

        del self.observations
        del self.actions
        del self.next_observations
        del self.rewards
        del self.dones

    def clip_model_config_grad_value(self, model_parameters):
        torch.nn.utils.clip_grad_norm_(model_parameters, self.config.CLIP_GRADIENT_VALUE)

        grads_list = [p.grad.data.cpu().numpy().flatten() for p in model_parameters if p.grad is not None]

        if grads_list:
            grads = np.concatenate(grads_list)
            self.last_model_grad_l2.value = np.sqrt(np.mean(np.square(grads)))
            self.last_model_grad_max.value = np.max(grads)

    def clip_actor_model_parameter_grad_value(self, actor_model_parameters):
        torch.nn.utils.clip_grad_norm_(actor_model_parameters, self.config.CLIP_GRADIENT_VALUE)

        actor_grads_list = [p.grad.data.cpu().numpy().flatten() for p in actor_model_parameters if p.grad is not None]

        if actor_grads_list:
            actor_grads = np.concatenate(actor_grads_list)
            self.last_actor_model_grad_l2.value = np.sqrt(np.mean(np.square(actor_grads)))
            self.last_actor_model_grad_max.value = np.max(actor_grads)

    def clip_critic_model_parameter_grad_value(self, critic_model_parameters):
        torch.nn.utils.clip_grad_norm_(critic_model_parameters, self.config.CLIP_GRADIENT_VALUE)

        critic_grads_list = [p.grad.data.cpu().numpy().flatten() for p in self.critic_model.parameters() if p.grad is not None]

        if critic_grads_list:
            critic_grads = np.concatenate(critic_grads_list)
            self.last_critic_model_grad_l2.value = np.sqrt(np.mean(np.square(critic_grads)))
            self.last_critic_model_grad_max.value = np.max(critic_grads)

    def synchronize_models(self, source_model, target_model):
        target_model.load_state_dict(source_model.state_dict())

    def soft_synchronize_models(self, source_model, target_model, tau):
        assert isinstance(tau, float)
        assert 0.0 < tau <= 1.0

        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = (1.0 - tau) * target_model_state[k] + tau * v
        target_model.load_state_dict(target_model_state)

    def calc_log_prob(self, mu_v, var_v, actions_v):
        p1 = -0.5 * ((actions_v - mu_v) ** 2) / (var_v.clamp(min=1e-03))
        # p1 = -1.0 * ((mu_v - actions_v) ** 2) / (2.0 * var_v.clamp(min=1e-3))
        p2 = -0.5 * torch.log(2 * np.pi * var_v)

        log_prob = (p1 + p2).sum(dim=-1, keepdim=True)
        return log_prob

    # OFF POLICY
    @abstractmethod
    def train_dqn(self, training_steps_v):
        return None, None

    @abstractmethod
    def train_double_dqn(self, training_steps_v):
        raise NotImplementedError()

    @abstractmethod
    def train_ddpg(self):
        raise NotImplementedError()

    @abstractmethod
    def train_td3(self, training_steps_v):
        raise NotImplementedError()

    @abstractmethod
    def train_sac(self, training_steps_v):
        raise NotImplementedError()

    # ON_POLICY
    @abstractmethod
    def train_reinforce(self):
        raise NotImplementedError()

    @abstractmethod
    def train_a2c(self):
        raise NotImplementedError()

    @abstractmethod
    def train_ppo(self):
        raise NotImplementedError()
