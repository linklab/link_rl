import torch.optim as optim
import numpy as np
import torch
import torch.multiprocessing as mp

from link_rl.c_models.f_ddpg_models import ContinuousDdpgModel
from link_rl.c_models_v2.e_ddpg_model_creator import ContinuousDdpgModelCreator
from link_rl.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from link_rl.g_utils.types import AgentMode


class AgentDdpg(OffPolicyAgent):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentDdpg, self).__init__(observation_space, action_space, config, need_train)

        # self._model_creator = ContinuousDdpgModelCreator(
        #     n_input=self.observation_shape[0],
        #     n_out_actions=self.n_out_actions,
        #     n_discrete_actions=self.n_discrete_actions
        # )

        model = self._model_creator.create_model()
        target_model = self._model_creator.create_model()

        self.actor_model, self.critic_model = model
        self.target_actor_model, self.target_critic_model = target_model

        self.actor_model.to(self.config.DEVICE)
        self.critic_model.to(self.config.DEVICE)
        self.target_actor_model.to(self.config.DEVICE)
        self.target_critic_model.to(self.config.DEVICE)

        self.model = self.actor_model
        self.model.eval()

        self.synchronize_models(source_model=self.actor_model, target_model=self.target_actor_model)
        self.synchronize_models(source_model=self.critic_model, target_model=self.target_critic_model)

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.config.LEARNING_RATE)

        self.training_step = 0

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)

    @torch.no_grad()
    def get_action(self, obs, mode=AgentMode.TRAIN):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.config.DEVICE)

        mu = self.actor_model(obs)
        mu = mu.detach().cpu().numpy()

        if mode == AgentMode.TRAIN:
            noises = np.random.normal(size=self.n_out_actions, loc=0, scale=1.0)
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
            next_mu_v = self.target_actor_model(self.next_observations)
            next_q_v = self.target_critic_model(self.next_observations, next_mu_v)
            next_q_v[self.dones] = 0.0
            target_q_v = self.rewards + self.config.GAMMA ** self.config.N_STEP * next_q_v
            if self.config.TARGET_VALUE_NORMALIZE:
                target_q_v = (target_q_v - torch.mean(target_q_v)) / (torch.std(target_q_v) + 1e-7)

        q_v = self.critic_model(self.observations, self.actions)

        critic_loss_each = self.config.LOSS_FUNCTION(q_v, target_q_v.detach(), reduction="none")

        if self.config.USE_PER:
            critic_loss_each *= torch.FloatTensor(self.important_sampling_weights).to(self.config.DEVICE)[:, None]

        critic_loss = critic_loss_each.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.parameters())
        self.critic_optimizer.step()
        ######################
        # train critic - end #
        ######################

        #######################
        # train actor - BEGIN #
        #######################
        mu_v = self.actor_model(self.observations)
        q_v = self.critic_model(self.observations, mu_v)
        actor_objective = q_v.mean()
        actor_loss = -1.0 * actor_objective

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.clip_actor_model_parameter_grad_value(self.actor_model.parameters())
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
