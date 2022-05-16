import torch.optim as optim
import numpy as np
import torch
import torch.multiprocessing as mp

from c_models.f_ddpg_models import ContinuousDdpgModel
from d_agents.off_policy.off_policy_agent import OffPolicyAgent
from g_utils.types import AgentMode


class AgentDdpg(OffPolicyAgent):
    def __init__(self, observation_space, action_space, config):
        super(AgentDdpg, self).__init__(observation_space, action_space, config)

        self.n_actions = self.n_out_actions

        self.ddpg_model = ContinuousDdpgModel(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, config=config
        )

        self.target_ddpg_model = ContinuousDdpgModel(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, config=config
        )

        self.model = self.ddpg_model.actor_model

        self.actor_model = self.ddpg_model.actor_model
        self.critic_model = self.ddpg_model.critic_model

        self.target_actor_model = self.target_ddpg_model.actor_model
        self.target_critic_model = self.target_ddpg_model.critic_model

        self.synchronize_models(source_model=self.actor_model, target_model=self.target_actor_model)
        self.synchronize_models(source_model=self.critic_model, target_model=self.target_critic_model)

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.config.LEARNING_RATE)

        self.training_step = 0

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        mu = self.actor_model.pi(obs, save_hidden=True)
        mu = mu.detach().cpu().numpy()

        if mode == AgentMode.TRAIN:
            noises = np.random.normal(size=self.n_actions, loc=0, scale=1.0)
            action = mu + noises
        else:
            action = mu

        action = np.clip(a=action, a_min=self.np_minus_ones, a_max=self.np_plus_ones)
        return action

    def train_ddpg(self):
        count_training_steps = 0

        ########################
        # train critic - BEGIN #
        ########################
        with torch.no_grad():
            next_mu_v = self.target_actor_model.pi(self.next_observations)
            next_q_v = self.target_critic_model.q(self.next_observations, next_mu_v)
            next_q_v[self.dones] = 0.0
            target_q_v = self.rewards + self.config.GAMMA ** self.config.N_STEP * next_q_v
            if self.config.TARGET_VALUE_NORMALIZE:
                target_q_v = (target_q_v - torch.mean(target_q_v)) / (torch.std(target_q_v) + 1e-7)

        q_v = self.critic_model.q(self.observations, self.actions)

        critic_loss_each = self.config.LOSS_FUNCTION(q_v, target_q_v.detach(), reduction="none")

        if self.config.USE_PER:
            critic_loss_each *= torch.FloatTensor(self.important_sampling_weights).to(self.config.DEVICE)[:, None]

        critic_loss = critic_loss_each.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params_list)
        self.critic_optimizer.step()
        ######################
        # train critic - end #
        ######################

        #######################
        # train actor - BEGIN #
        #######################
        mu_v = self.actor_model.pi(self.observations)
        q_v = self.critic_model.q(self.observations, mu_v)
        actor_objective = q_v.mean()
        actor_loss = -1.0 * actor_objective

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params_list)
        self.actor_optimizer.step()
        #####################
        # train actor - END #
        #####################

        # TAU: 0.005
        self.soft_synchronize_models(
            source_model=self.actor_model, target_model=self.target_actor_model, tau=self.config.TAU
        )

        # TAU: 0.005
        self.soft_synchronize_models(
            source_model=self.critic_model, target_model=self.target_critic_model, tau=self.config.TAU
        )

        self.last_critic_loss.value = critic_loss.item()
        self.last_actor_objective.value = actor_objective.item()

        count_training_steps += 1

        return count_training_steps, critic_loss_each
