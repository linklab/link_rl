import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box

from c_models.e_ddpg_models import DiscreteDdpgModel, ContinuousDdpgModel
from d_agents.agent import Agent
from g_utils.commons import EpsilonTracker
from g_utils.types import AgentMode, ModelType


class AgentDdpg(Agent):
    def __init__(self, observation_space, action_space, device, parameter):
        super(AgentDdpg, self).__init__(observation_space, action_space, device, parameter)

        if isinstance(self.action_space, Discrete):
            self.n_actions = self.n_discrete_actions
            self.ddpg_model = DiscreteDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
            ).to(device)

            self.target_ddpg_model = DiscreteDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
            ).to(device)
        elif isinstance(self.action_space, Box):
            self.n_actions = self.n_out_actions
            self.action_bound_low = np.expand_dims(self.action_space.low, axis=0)
            self.action_bound_high = np.expand_dims(self.action_space.high, axis=0)

            self.action_scale_factor = np.max(np.maximum(
                np.absolute(self.action_bound_low), np.absolute(self.action_bound_high)
            ), axis=-1)[0]

            self.ddpg_model = ContinuousDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            ).to(device)

            self.target_ddpg_model = ContinuousDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            ).to(device)
        else:
            raise ValueError()

        self.ddpg_model.share_memory()
        self.synchronize_models(source_model=self.ddpg_model, target_model=self.target_ddpg_model)

        self.actor_optimizer = optim.Adam(self.ddpg_model.actor_params, lr=self.parameter.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.ddpg_model.critic_params, lr=self.parameter.LEARNING_RATE)

        self.model = self.ddpg_model
        self.training_steps = 0

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_loss = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        mu = self.ddpg_model.pi(obs)
        mu = mu.detach()
        if mode == AgentMode.TRAIN:
            noises = np.random.normal(size=self.n_actions, loc=0, scale=1.0)
            action = mu + noises
        else:
            action = mu

        action = np.clip(action.cpu().numpy(), self.action_bound_low, self.action_bound_high)
        return action

    def train_ddpg(self, training_steps_v):
        batch = self.buffer.sample(self.parameter.BATCH_SIZE, device=self.device)

        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, next_observations, rewards, dones = batch

        #######################
        # train actor - BEGIN #
        #######################
        mu_v = self.ddpg_model.pi(observations)
        q_v = self.ddpg_model.q(observations, mu_v)
        actor_loss = -1.0 * q_v.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.ddpg_model.actor_params, self.parameter.CLIP_GRADIENT_VALUE)
        self.actor_optimizer.step()
        #####################
        # train actor - END #
        #####################

        ########################
        # train critic - BEGIN #
        ########################
        next_mu_v = self.target_ddpg_model.pi(next_observations)
        next_q_v = self.target_ddpg_model.q(next_observations, next_mu_v)
        target_q_v = rewards + self.parameter.GAMMA ** self.parameter.N_STEP * next_q_v

        q_v = self.ddpg_model.q(observations, actions)

        critic_loss_v = F.mse_loss(q_v, target_q_v.detach(), reduction='none')

        critic_loss = critic_loss_v.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.ddpg_model.critic_params, self.parameter.CLIP_GRADIENT_VALUE)
        self.critic_optimizer.step()
        ######################
        # train critic - end #
        ######################

        self.soft_synchronize_models(
            source_model=self.ddpg_model, target_model=self.target_ddpg_model, tau=self.parameter.TAU
        ) # TAU: 0.0001

        self.last_critic_loss.value = critic_loss.item()
        self.last_actor_loss.value = actor_loss.item()
