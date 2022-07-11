from abc import abstractmethod

import torch
import torch.multiprocessing as mp
from torch import optim
from gym.spaces import Discrete, Box

import numpy as np

from link_rl.a_configuration.a_base_config.c_models.config_recurrent_convolutional_models import \
    ConfigRecurrent1DConvolutionalModel, ConfigRecurrent2DConvolutionalModel
from link_rl.a_configuration.a_base_config.c_models.config_recurrent_linear_models import ConfigRecurrentLinearModel
from link_rl.c_encoders.a_encoder import BaseEncoder
from link_rl.d_models import model_creators
from link_rl.c_encoders import encoder_creators
from link_rl.d_models.a_model import BaseModel
from link_rl.h_utils.commons import get_continuous_action_info
from link_rl.h_utils.types import AgentMode, ActorCriticAgentTypes, AgentType, OffPolicyAgentTypes


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

        if not hasattr(self, "_encoder_creator"):
            print(self.config.ENCODER_TYPE, "##########")
            encoder_creators_class = encoder_creators.get(self.config.ENCODER_TYPE)
            self._encoder_creator: BaseEncoder = encoder_creators_class(
                observation_shape=self.observation_shape
            )

        if not hasattr(self, "_model_creator"):
            print(self.config.MODEL_TYPE, "##########")
            model_creator_class = model_creators.get(self.config.MODEL_TYPE)
            self._model_creator: BaseModel = model_creator_class(
                n_input=self._encoder_creator.encoder_out,
                n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions
            )

        self.model = None
        if self.config.AGENT_TYPE in ActorCriticAgentTypes:
            self.actor_model = None
            self.critic_model = None

        self.last_model_grad_max = mp.Value('d', 0.0)
        self.last_model_grad_l1 = mp.Value('d', 0.0)

        self.last_actor_model_grad_l1 = mp.Value('d', 0.0)
        self.last_actor_model_grad_max = mp.Value('d', 0.0)

        self.last_critic_model_grad_l1 = mp.Value('d', 0.0)
        self.last_critic_model_grad_max = mp.Value('d', 0.0)

        self.is_recurrent_model = any([
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrentLinearModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent1DConvolutionalModel),
            isinstance(self.config.MODEL_PARAMETER, ConfigRecurrent2DConvolutionalModel)
        ])

        self.step = 0

        ### ENCODER - START###
        self.encoder = self._encoder_creator.create_encoder()
        self.target_encoder = self._encoder_creator.create_encoder()

        self.encoder.to(self.config.DEVICE)
        self.target_encoder.to(self.config.DEVICE)

        self.synchronize_models(source_model=self.encoder, target_model=self.target_encoder)

        self.encoder.share_memory()
        self.encoder.eval()

        self.encoder_is_not_identity = type(self.encoder).__name__ != "Identity"
        if self.encoder_is_not_identity:
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.config.LEARNING_RATE)
        ### ENCODER - END ###

    @property
    def enc_out(self):
        return self._encoder_creator.encoder_out

    @abstractmethod
    @torch.no_grad()
    def get_action(self, obs, unavailable_actions=None, mode=AgentMode.TRAIN, t0=False, step=None):
        raise NotImplementedError()

    def _before_train(self):
        if self.config.AGENT_TYPE in ActorCriticAgentTypes:
            self.actor_model.train()
            self.critic_model.train()
        else:
            self.model.train()

        if self.encoder_is_not_identity:
            self.encoder.train()
            self.encoder_optimizer.zero_grad()

    @abstractmethod
    def train(self, training_steps_v=None):
        raise NotImplementedError()

    def train_encoder(self):
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.config.CLIP_GRADIENT_VALUE)
        self.encoder_optimizer.step()

        if self.config.AGENT_TYPE in OffPolicyAgentTypes:
            if self.config.AGENT_TYPE in [AgentType.DQN]:
                self.synchronize_models(source_model=self.encoder, target_model=self.target_encoder)
            else:
                assert hasattr(self.config, "TAU")
                self.soft_synchronize_models(
                    source_model=self.encoder, target_model=self.target_encoder, tau=self.config.TAU
                )

        self.encoder.eval()

    def _after_train(self):
        if self.config.AGENT_TYPE in ActorCriticAgentTypes:
            self.actor_model.eval()
            self.critic_model.eval()
        else:
            self.model.eval()

    def clip_model_config_grad_value(self, model_parameters):
        total_norm = torch.nn.utils.clip_grad_norm_(model_parameters, self.config.CLIP_GRADIENT_VALUE)

        grads_list = [p.grad.data.cpu().numpy().flatten() for p in self.model.parameters() if p.grad is not None]

        if grads_list:
            grads = np.concatenate(grads_list)
            self.last_model_grad_l1.value = total_norm
            self.last_model_grad_max.value = np.max(grads)

    def clip_actor_model_parameter_grad_value(self, actor_model_parameters):
        total_norm = torch.nn.utils.clip_grad_norm_(actor_model_parameters, self.config.CLIP_GRADIENT_VALUE)

        actor_grads_list = [p.grad.data.cpu().numpy().flatten() for p in self.actor_model.parameters() if
                            p.grad is not None]

        if actor_grads_list:
            actor_grads = np.concatenate(actor_grads_list)
            self.last_actor_model_grad_l1.value = total_norm
            self.last_actor_model_grad_max.value = np.max(actor_grads)

    def clip_critic_model_parameter_grad_value(self, critic_model_parameters):
        torch.nn.utils.clip_grad_norm_(critic_model_parameters, self.config.CLIP_GRADIENT_VALUE)

        critic_grads_list = [p.grad.data.cpu().numpy().flatten() for p in self.critic_model.parameters() if
                             p.grad is not None]

        if critic_grads_list:
            critic_grads = np.concatenate(critic_grads_list)
            self.last_critic_model_grad_l1.value = np.sqrt(np.mean(np.square(critic_grads)))
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

