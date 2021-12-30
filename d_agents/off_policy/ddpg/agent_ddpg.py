import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box
from torch.distributions import normal

from c_models.e_ddpg_models import DiscreteDdpgModel, ContinuousDdpgModel
from d_agents.agent import Agent
from g_utils.types import AgentMode


class AgentDdpg(Agent):
    def __init__(self, observation_space, action_space, device, parameter):
        super(AgentDdpg, self).__init__(observation_space, action_space, device, parameter)

        if isinstance(self.action_space, Discrete):
            self.n_actions = self.n_discrete_actions
            self.ddpg_model = DiscreteDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
            )

            self.target_ddpg_model = DiscreteDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
            )
        elif isinstance(self.action_space, Box):
            self.n_actions = self.n_out_actions
            self.action_bound_low = torch.tensor(np.expand_dims(self.action_space.low, axis=0), device=device)
            self.action_bound_high = torch.tensor(np.expand_dims(self.action_space.high, axis=0), device=device)

            self.action_scale_factor = torch.max(torch.maximum(
                torch.absolute(self.action_bound_low), torch.absolute(self.action_bound_high)
            ), dim=-1)[0]

            self.ddpg_model = ContinuousDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            )

            self.target_ddpg_model = ContinuousDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            )
        else:
            raise ValueError()

        self.model = self.ddpg_model.actor_model
        self.actor_model = self.ddpg_model.actor_model
        self.critic_model = self.ddpg_model.critic_model
        self.target_actor_model = self.target_ddpg_model.actor_model
        self.target_critic_model = self.target_ddpg_model.critic_model

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.synchronize_models(
            source_model=self.actor_model, target_model=self.target_actor_model
        )
        self.synchronize_models(
            source_model=self.critic_model, target_model=self.target_critic_model
        )

        self.actor_optimizer = optim.Adam(self.actor_model.actor_params, lr=self.parameter.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.critic_params, lr=self.parameter.LEARNING_RATE)

        self.training_steps = 0

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_loss = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        mu = self.actor_model.pi(obs)
        if mode == AgentMode.TRAIN:
            noise_dist = normal.Normal(loc=0.0, scale=1.0)
            noises = noise_dist.sample(sample_shape=mu.size())
            action = (mu + noises) * self.action_scale_factor
        else:
            action = mu * self.action_scale_factor

        action = action.clamp(self.action_bound_low, self.action_bound_high).detach().numpy()
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
        mu_v = self.actor_model.pi(observations)
        q_v = self.critic_model.q(observations, mu_v)
        actor_loss = -1.0 * q_v.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor_model.actor_params, self.parameter.CLIP_GRADIENT_VALUE)
        self.actor_optimizer.step()
        #####################
        # train actor - END #
        #####################

        ########################
        # train critic - BEGIN #
        ########################
        with torch.no_grad():
            next_mu_v = self.target_actor_model.pi(next_observations)
            next_q_v = self.target_critic_model.q(next_observations, next_mu_v)
            next_q_v[dones] = 0.0
            target_q_v = rewards + self.parameter.GAMMA ** self.parameter.N_STEP * next_q_v

        q_v = self.critic_model.q(observations, actions)

        critic_loss = F.mse_loss(q_v, target_q_v.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic_model.critic_params, self.parameter.CLIP_GRADIENT_VALUE)
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
