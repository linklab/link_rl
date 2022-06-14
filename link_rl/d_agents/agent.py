from abc import abstractmethod

import torch
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box

import numpy as np

from link_rl.g_utils.commons import get_continuous_action_info
from link_rl.g_utils.types import AgentMode, ActorCriticAgentTypes


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

        self.model = None
        if self.config.AGENT_TYPE in ActorCriticAgentTypes:
            self.actor_model = None
            self.critic_model = None

        self.last_model_grad_max = mp.Value('d', 0.0)
        self.last_model_grad_l2 = mp.Value('d', 0.0)

        self.last_actor_model_grad_l2 = mp.Value('d', 0.0)
        self.last_actor_model_grad_max = mp.Value('d', 0.0)

        self.last_critic_model_grad_l2 = mp.Value('d', 0.0)
        self.last_critic_model_grad_max = mp.Value('d', 0.0)

        self.step = 0

    @abstractmethod
    def get_action(self, obs, unavailable_actions=None, mode=AgentMode.TRAIN):
        raise NotImplementedError()

    @abstractmethod
    def train(self, training_steps_v=None):
        raise NotImplementedError()

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

        critic_grads_list = [p.grad.data.cpu().numpy().flatten() for p in self.critic_model.parameters() if
                             p.grad is not None]

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

