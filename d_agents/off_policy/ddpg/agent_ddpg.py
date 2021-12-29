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
            self.ddpg_model = DiscreteDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
            ).to(device)

            self.target_ddpg_model = DiscreteDdpgModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
            ).to(device)
        elif isinstance(self.action_space, Box):
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

    def get_action(self, obs, mode=AgentMode.TRAIN):
        mu = self.ddpg_model.mu(obs)

        if mode == AgentMode.TRAIN:
            noises = np.random.normal(size=self.action_space, loc=0, scale=1.0)
            action = mu + noises

        else:
            action = mu

        action = np.clip(action.cpu().numpy(), self.action_bound_low, self.action_bound_high)
        return action

    def train_dqn(self, buffer, training_steps_v):
        batch = buffer.sample(self.parameter.BATCH_SIZE, device=self.device)

        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, next_observations, rewards, dones = batch

        # state_action_values.shape: torch.Size([32, 1])
        state_action_values = self.q_net(observations).gather(
            dim=1, index=actions
        )

        with torch.no_grad():
            # next_state_values.shape: torch.Size([32, 1])
            next_state_values = self.target_q_net(next_observations).max(
                dim=1, keepdim=True
            ).values
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

            # target_state_action_values.shape: torch.Size([32, 1])
            target_state_action_values = rewards + self.parameter.GAMMA ** self.parameter.N_STEP * next_state_values

        # loss is just scalar torch value
        q_net_loss = F.mse_loss(state_action_values, target_state_action_values)

        # print("observations.shape: {0}, actions.shape: {1}, "
        #       "next_observations.shape: {2}, rewards.shape: {3}, dones.shape: {4}".format(
        #     observations.shape, actions.shape,
        #     next_observations.shape, rewards.shape, dones.shape
        # ))
        # print("state_action_values.shape: {0}".format(state_action_values.shape))
        # print("next_state_values.shape: {0}".format(next_state_values.shape))
        # print("target_state_action_values.shape: {0}".format(
        #     target_state_action_values.shape
        # ))
        # print("loss.shape: {0}".format(loss.shape))

        self.optimizer.zero_grad()
        q_net_loss.backward()
        self.optimizer.step()

        # sync
        if training_steps_v % self.parameter.TARGET_SYNC_INTERVAL_TRAINING_STEPS == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon.value = self.epsilon_tracker.epsilon(training_steps_v)

        self.last_q_net_loss.value = q_net_loss.item()
