import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box

from c_models.e_ddpg_models import DiscreteDdpgModel, ContinuousDdpgModel
from d_agents.agent import Agent
from g_utils.commons import EpsilonTracker, get_continuous_action_info
from g_utils.types import AgentMode, ModelType


class AgentDdpg(Agent):
    def __init__(self, observation_space, action_space, parameter):
        super(AgentDdpg, self).__init__(observation_space, action_space, parameter)

        if isinstance(self.action_space, Discrete):
            self.n_actions = self.n_discrete_actions
            self.ddpg_model = DiscreteDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, parameter=parameter
            )

            self.target_ddpg_model = DiscreteDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, parameter=parameter
            )
        elif isinstance(self.action_space, Box):
            self.n_actions = self.n_out_actions

            self.ddpg_model = ContinuousDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, parameter=parameter
            )

            self.target_ddpg_model = ContinuousDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, parameter=parameter
            )
        else:
            raise ValueError()

        self.model = self.ddpg_model.actor_model

        self.actor_model = self.ddpg_model.actor_model
        self.critic_model = self.ddpg_model.critic_model

        self.target_actor_model = self.target_ddpg_model.actor_model
        self.target_critic_model = self.target_ddpg_model.critic_model

        self.synchronize_models(source_model=self.actor_model, target_model=self.target_actor_model)
        self.synchronize_models(source_model=self.critic_model, target_model=self.target_critic_model)

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.actor_optimizer = optim.Adam(self.actor_model.actor_params, lr=self.parameter.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.critic_params, lr=self.parameter.LEARNING_RATE)

        self.training_steps = 0

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_loss = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        if isinstance(self.action_space, Discrete):
            pass
        elif isinstance(self.action_space, Box):
            mu = self.actor_model.pi(obs)
            mu = mu.detach().cpu().numpy()

            if mode == AgentMode.TRAIN:
                noises = np.random.normal(size=self.n_actions, loc=0, scale=1.0)
                action = mu + noises
            else:
                action = mu

            action = np.clip(a=action, a_min=self.np_minus_ones, a_max=self.np_plus_ones)
            return action
        else:
            raise ValueError()

    def train_ddpg(self):
        #######################
        # train actor - BEGIN #
        #######################
        mu_v = self.actor_model.pi(self.observations)
        q_v = self.critic_model.q(self.observations, mu_v)
        actor_loss = -1.0 * q_v.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params)
        self.actor_optimizer.step()
        #####################
        # train actor - END #
        #####################

        ########################
        # train critic - BEGIN #
        ########################
        with torch.no_grad():
            next_mu_v = self.target_actor_model.pi(self.next_observations)
            next_q_v = self.target_critic_model.q(self.next_observations, next_mu_v)
            next_q_v[self.dones] = 0.0
            target_q_v = self.rewards + self.parameter.GAMMA ** self.parameter.N_STEP * next_q_v

        q_v = self.critic_model.q(self.observations, self.actions)

        critic_loss = F.huber_loss(q_v, target_q_v.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params)
        self.critic_optimizer.step()
        ######################
        # train critic - end #
        ######################

        # TAU: 0.0001
        self.soft_synchronize_models(
            source_model=self.actor_model, target_model=self.target_actor_model, tau=self.parameter.TAU
        )

        # TAU: 0.0001
        self.soft_synchronize_models(
            source_model=self.critic_model, target_model=self.target_critic_model, tau=self.parameter.TAU
        )

        self.last_critic_loss.value = critic_loss.item()
        self.last_actor_loss.value = actor_loss.item()
