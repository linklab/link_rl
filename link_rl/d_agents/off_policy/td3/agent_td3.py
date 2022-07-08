import torch.optim as optim
import numpy as np
import torch
import torch.multiprocessing as mp

from link_rl.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from link_rl.g_utils.types import AgentMode


class AgentTd3(OffPolicyAgent):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentTd3, self).__init__(observation_space, action_space, config, need_train)

        # models
        self.encoder = self._encoder_creator.create_encoder()
        self.target_encoder = self._encoder_creator.create_encoder()
        self.actor_model, self.critic_model = self._model_creator.create_model()
        self.target_actor_model, self.target_critic_model = self._model_creator.create_model()

        # to(device)
        self.encoder.to(self.config.DEVICE)
        self.target_encoder.to(self.config.DEVICE)
        self.actor_model.to(self.config.DEVICE)
        self.target_actor_model.to(self.config.DEVICE)
        self.critic_model.to(self.config.DEVICE)
        self.target_critic_model.to(self.config.DEVICE)

        # Access
        self.model = self.actor_model
        self.model.eval()

        # sync models
        self.synchronize_models(source_model=self.encoder, target_model=self.target_encoder)
        self.synchronize_models(source_model=self.actor_model, target_model=self.target_actor_model)
        self.synchronize_models(source_model=self.critic_model, target_model=self.target_critic_model)

        # share memory
        self.encoder.share_memory()
        self.actor_model.share_memory()
        self.critic_model.share_memory()

        # optimizers
        self.encoder_is_not_identity = type(self.encoder).__name__ != "Identity"
        if self.encoder_is_not_identity:
            self.encoder_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.LEARNING_RATE)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.config.LEARNING_RATE)

        # training step
        self.training_step = 0

        # loss
        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)

    def actor_forward(self, obs):
        x = self.encoder(obs)
        mu = self.actor_model(x)
        return mu

    def critic_forward(self, obs, action):
        x = self.encoder(obs)
        q1, q2 = self.critic_model(x, action)
        return q1, q2

    def target_actor_forward(self, obs):
        x = self.target_encoder(obs)
        mu = self.target_actor_model(x)
        return mu

    def target_critic_forward(self, obs, action):
        x = self.target_encoder(obs)
        q1, q2 = self.target_critic_model(x, action)
        return q1, q2

    @torch.no_grad()
    def get_action(self, obs, mode=AgentMode.TRAIN):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.config.DEVICE)

        mu = self.actor_forward(obs)
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
            next_mu_v = self.target_actor_forward(self.next_observations)
            next_noises = torch.normal(
                mean=torch.zeros_like(next_mu_v), std=torch.ones_like(next_mu_v)
            ).to(self.config.DEVICE)
            next_action = next_mu_v + torch.clip(input=next_noises, min=self.torch_minus_ones, max=self.torch_plus_ones)
            next_action = torch.clip(input=next_action, min=self.torch_minus_ones, max=self.torch_plus_ones)

            next_q1_value, next_q2_value = self.target_critic_forward(self.next_observations, next_action)
            min_next_q_value = torch.min(next_q1_value, next_q2_value)
            min_next_q_value[self.dones] = 0.0
            target_q_v = self.rewards + self.config.GAMMA ** self.config.N_STEP * min_next_q_value
            if self.config.TARGET_VALUE_NORMALIZE:
                target_q_v = (target_q_v - torch.mean(target_q_v)) / (torch.std(target_q_v) + 1e-7)

        q1_value, q2_value = self.critic_forward(self.observations, self.actions)

        critic_loss_each = (self.config.LOSS_FUNCTION(q1_value, target_q_v.detach(), reduction="none")
                            + self.config.LOSS_FUNCTION(q2_value, target_q_v.detach(), reduction="none")) / 2.0

        if self.config.USE_PER:
            critic_loss_each *= torch.FloatTensor(self.important_sampling_weights).to(self.config.DEVICE)[:, None]

        critic_loss = critic_loss_each.mean()

        if self.encoder_is_not_identity:
            self.encoder_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.encoder.parameters())
        self.clip_critic_model_parameter_grad_value(self.critic_model.parameters())
        if self.encoder_is_not_identity:
            self.encoder_optimizer.step()
        self.critic_optimizer.step()

        self.last_critic_loss.value = critic_loss.item()

        # TAU: 0.005
        self.soft_synchronize_models(
            source_model=self.encoder, target_model=self.target_encoder, tau=self.config.TAU
        )
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
            mu_v = self.actor_forward(self.observations)
            q1_value, q2_value = self.critic_forward(self.observations, mu_v)
            actor_objective = torch.min(q1_value, q2_value).mean()
            actor_loss = -1.0 * actor_objective

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.clip_actor_model_parameter_grad_value(self.actor_model.parameters())
            self.actor_optimizer.step()

            self.last_actor_objective.value = actor_objective.item()

            # TAU: 0.005
            self.soft_synchronize_models(
                source_model=self.actor_model, target_model=self.target_actor_model, tau=self.config.TAU
            )
        #####################
        # train actor - END #
        #####################

        count_training_steps += 1

        return count_training_steps, critic_loss_each
