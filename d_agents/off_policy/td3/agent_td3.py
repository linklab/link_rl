import torch.optim as optim
import numpy as np
import torch
import torch.multiprocessing as mp

from c_models.g_td3_models import ContinuousTd3Model
from d_agents.agent import Agent
from g_utils.types import AgentMode


class AgentTd3(Agent):
    def __init__(self, observation_space, action_space, config):
        super(AgentTd3, self).__init__(observation_space, action_space, config)

        self.td3_model = ContinuousTd3Model(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, config=config
        )

        self.target_td3_model = ContinuousTd3Model(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, config=config
        )

        self.model = self.td3_model.actor_model

        self.actor_model = self.td3_model.actor_model
        self.critic_model = self.td3_model.critic_model

        self.target_actor_model = self.target_td3_model.actor_model
        self.target_critic_model = self.target_td3_model.critic_model

        self.synchronize_models(source_model=self.actor_model, target_model=self.target_actor_model)
        self.synchronize_models(source_model=self.critic_model, target_model=self.target_critic_model)

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.config.LEARNING_RATE)

        self.training_step = 0

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_loss = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        mu = self.actor_model.pi(obs)
        mu = mu.detach().cpu().numpy()

        if mode == AgentMode.TRAIN:
            noises = np.random.normal(size=self.n_out_actions, loc=0, scale=1.0)
            noises = np.clip(a=noises, a_min=self.np_minus_ones, a_max=self.np_plus_ones)
            action = mu + noises
        else:
            action = mu

        action = np.clip(a=action, a_min=self.np_minus_ones, a_max=self.np_plus_ones)
        return action

    def train_td3(self, training_steps_v):
        count_training_steps = 0

        ########################
        # train critic - BEGIN #
        ########################
        with torch.no_grad():
            next_mu_v = self.target_actor_model.pi(self.next_observations)
            next_noises = torch.normal(
                mean=torch.zeros_like(next_mu_v), std=torch.ones_like(next_mu_v)
            ).to(self.config.DEVICE)
            next_action = next_mu_v + torch.clip(input=next_noises, min=self.torch_minus_ones, max=self.torch_plus_ones)
            next_action = torch.clip(input=next_action, min=self.torch_minus_ones, max=self.torch_plus_ones)

            next_q1_value, next_q2_value = self.target_critic_model.q(self.next_observations, next_action)
            min_next_q_value = torch.min(next_q1_value, next_q2_value)
            min_next_q_value[self.dones] = 0.0
            target_q_v = self.rewards + self.config.GAMMA ** self.config.N_STEP * min_next_q_value

        q1_value, q2_value = self.critic_model.q(self.observations, self.actions)

        critic_loss = self.config.LOSS_FUNCTION(q1_value, target_q_v.detach()) + self.config.LOSS_FUNCTION(q2_value, target_q_v.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params_list)
        self.critic_optimizer.step()

        self.last_critic_loss.value = critic_loss.item()

        # TAU: 0.005
        self.soft_synchronize_models(
            source_model=self.critic_model, target_model=self.target_critic_model, tau=self.config.TAU
        )
        ######################
        # train critic - end #
        ######################

        #######################
        # train actor - BEGIN #
        #######################
        if training_steps_v % self.config.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP == 0:
            mu_v = self.actor_model.pi(self.observations)
            q1_value, q2_value = self.critic_model.q(self.observations, mu_v)
            q_value = (q1_value + q2_value) / 2.0
            actor_loss = -1.0 * q_value.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params_list)
            self.actor_optimizer.step()

            self.last_actor_loss.value = actor_loss.item()

            # TAU: 0.005
            self.soft_synchronize_models(
                source_model=self.actor_model, target_model=self.target_actor_model, tau=self.config.TAU
            )
        #####################
        # train actor - END #
        #####################

        count_training_steps += 1

        return count_training_steps
