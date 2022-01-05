import math
from abc import abstractmethod, ABC

import torch
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box, MultiDiscrete

import numpy as np
from g_utils.buffers import Buffer
from g_utils.commons import get_continuous_action_info
from g_utils.types import AgentMode, AgentType, OnPolicyAgentTypes, ActorCriticAgentTypes


class Agent:
    def __init__(self, observation_space, action_space, parameter):
        self.observation_space = observation_space
        self.action_space = action_space
        self.parameter = parameter

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
            self.action_scale_factor = None
            self.np_minus_ones = None
            self.np_plus_ones = None

            self.action_scale = None
            self.action_bias = None
        elif isinstance(action_space, Box):
            self.n_discrete_actions = None
            self.n_out_actions = action_space.shape[0]
            _, _, self.action_scale_factor = get_continuous_action_info(action_space)
            self.np_minus_ones = np.full(shape=action_space.shape, fill_value=-1.0)
            self.np_plus_ones = np.full(shape=action_space.shape, fill_value=1.0)
            self.torch_minus_ones = torch.full(size=action_space.shape, fill_value=-1.0).to(self.parameter.DEVICE)
            self.torch_plus_ones = torch.full(size=action_space.shape, fill_value=1.0).to(self.parameter.DEVICE)

            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)
        else:
            raise ValueError()

        self.buffer = Buffer(capacity=parameter.BUFFER_CAPACITY, device=self.parameter.DEVICE)

        self.model = None
        if self.parameter.AGENT_TYPE in ActorCriticAgentTypes:
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

    def before_train(self):
        if self.parameter.AGENT_TYPE in ActorCriticAgentTypes:
            assert self.actor_model
            assert self.critic_model
            assert self.model is self.actor_model

        # observations.shape: torch.Size([32, 4, 84, 84]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4, 84, 84]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        self.observations, self.actions, self.next_observations, self.rewards, self.dones = self.buffer.sample(
            batch_size=self.parameter.BATCH_SIZE
        )

    def train(self, training_steps_v=None):
        self.before_train()

        is_train_success_done = False
        if self.parameter.AGENT_TYPE in [AgentType.DQN, AgentType.DUELING_DQN]:
            if len(self.buffer) >= self.parameter.MIN_BUFFER_SIZE_FOR_TRAIN:
                self.train_dqn(training_steps_v=training_steps_v)
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.DOUBLE_DQN:
            if len(self.buffer) >= self.parameter.MIN_BUFFER_SIZE_FOR_TRAIN:
                self.train_double_dqn(training_steps_v=training_steps_v)
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.REINFORCE:
            if len(self.buffer) > 0:
                self.train_reinforce()
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.A2C:
            if len(self.buffer) >= self.parameter.BATCH_SIZE:
                self.train_a2c()
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.DDPG:
            if len(self.buffer) >= self.parameter.BATCH_SIZE:
                self.train_ddpg()
                is_train_success_done = True
        elif self.parameter.AGENT_TYPE == AgentType.SAC:
            if len(self.buffer) >= self.parameter.BATCH_SIZE:
                self.train_sac(training_steps_v=training_steps_v)
                is_train_success_done = True
        else:
            raise ValueError()

        # NOTE !!!

        if is_train_success_done:
            if self.parameter.AGENT_TYPE in OnPolicyAgentTypes:
                self.buffer.clear()

        if self.parameter.AGENT_TYPE in ActorCriticAgentTypes:
            self.after_actor_critic_train()
        else:
            self.after_train()

        return is_train_success_done

    def after_actor_critic_train(self):
        del self.observations
        del self.actions
        del self.next_observations
        del self.rewards
        del self.dones

    def after_train(self):
        del self.observations
        del self.actions
        del self.next_observations
        del self.rewards
        del self.dones

    def clip_model_parameter_grad_value(self, model_parameters):
        torch.nn.utils.clip_grad_norm_(model_parameters, self.parameter.CLIP_GRADIENT_VALUE)

        grads_list = [p.grad.data.cpu().numpy().flatten() for p in model_parameters if p.grad is not None]
        if grads_list:
            grads = np.concatenate(grads_list)
            self.last_model_grad_l2.value = np.sqrt(np.mean(np.square(grads)))
            self.last_model_grad_max.value = np.max(grads)

    def clip_actor_model_parameter_grad_value(self, actor_model_parameters):
        torch.nn.utils.clip_grad_norm_(actor_model_parameters, self.parameter.CLIP_GRADIENT_VALUE)
        actor_grads_list = [p.grad.data.cpu().numpy().flatten() for p in actor_model_parameters if p.grad is not None]
        if actor_grads_list:
            actor_grads = np.concatenate(actor_grads_list)
            self.last_actor_model_grad_l2.value = np.sqrt(np.mean(np.square(actor_grads)))
            self.last_actor_model_grad_max.value = np.max(actor_grads)

    def clip_critic_model_parameter_grad_value(self, critic_model_parameters):
        torch.nn.utils.clip_grad_norm_(critic_model_parameters, self.parameter.CLIP_GRADIENT_VALUE)
        critic_grads_list = [p.grad.data.cpu().numpy().flatten() for p in self.critic_model.parameters() if p.grad is not None]
        if critic_grads_list:
            critic_grads = np.concatenate(critic_grads_list)
            self.last_critic_model_grad_l2.value = np.sqrt(np.mean(np.square(critic_grads)))
            self.last_critic_model_grad_max.value = np.max(critic_grads)

    @abstractmethod
    def train_dqn(self, training_steps_v):
        return 0.0

    @abstractmethod
    def train_double_dqn(self, training_steps_v):
        return 0.0

    @abstractmethod
    def train_reinforce(self):
        return 0.0

    @abstractmethod
    def train_a2c(self):
        return 0.0

    @abstractmethod
    def train_ddpg(self):
        return 0.0

    @abstractmethod
    def train_sac(self, training_steps_v):
        return 0.0

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


