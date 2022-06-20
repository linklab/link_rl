import numpy as np
import torch
from gym import spaces

from link_rl.d_agents.off_policy.td3 import drqv2_utils
from link_rl.d_agents.off_policy.td3.drqv2 import RandomShiftsAug, Encoder
import torch.multiprocessing as mp

from link_rl.c_models.g_td3_models import ContinuousTd3Model
from link_rl.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from link_rl.g_utils.types import AgentMode
import torch.optim as optim

class AgentTd3DrqV2(OffPolicyAgent):
    def __init__(self, observation_space, action_space, config):
        super(AgentTd3DrqV2, self).__init__(observation_space, action_space, config)
        self.td3_model = ContinuousTd3Model(
            observation_shape=(39200, ), n_out_actions=self.n_out_actions, config=config
        )

        self.target_td3_model = ContinuousTd3Model(
            observation_shape=(39200, ), n_out_actions=self.n_out_actions, config=config
        )

        self.model = self.td3_model.actor_model
        self.model.eval()

        # models
        self.encoder = Encoder(self.observation_shape).to(self.config.DEVICE)
        self.actor_model = self.td3_model.actor_model
        self.critic_model = self.td3_model.critic_model

        self.target_actor_model = self.target_td3_model.actor_model
        self.target_critic_model = self.target_td3_model.critic_model

        self.synchronize_models(source_model=self.actor_model, target_model=self.target_actor_model)
        self.synchronize_models(source_model=self.critic_model, target_model=self.target_critic_model)

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        # optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.config.LEARNING_RATE)
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.config.LEARNING_RATE)

        self.training_step = 0

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)
        self.global_step = 0

    @torch.no_grad()
    def get_action(self, obs, mode=AgentMode.TRAIN):
        self.global_step += 1

        obs = torch.as_tensor(obs, device=self.config.DEVICE)

        obs = self.encoder(obs)
        obs = torch.squeeze(input=obs, dim=0)
        stddev = 1.0
        mu = self.actor_model.pi(obs, save_hidden=True)
        std = torch.ones_like(mu) * stddev

        dist = drqv2_utils.TruncatedNormal(mu, std)

        if mode == AgentMode.TRAIN:
            action = dist.sample().detach().cpu().numpy()
            if self.global_step < self.config.num_expl_steps:
                action = np.clip(a=action, a_min=self.np_minus_ones, a_max=self.np_plus_ones)
        else:
            action = dist.mean.detach().cpu().numpy()

        return action

    def train_td3_drqv2(self, training_steps_v):
        count_training_steps = 0

        with torch.no_grad():
            stddev = drqv2_utils.schedule(self.config.stddev_schedule, training_steps_v)
            mu = self.actor_model.pi(self.next_observations)
            mu = torch.tanh(mu)
            std = torch.ones_like(mu) * stddev
            dist = drqv2_utils.TruncatedNormal(mu, std)
            next_action = dist.sample(clip=1.0)

            next_q1_value, next_q2_value = self.target_critic_model.q(self.next_observations, next_action)
            min_next_q_value = torch.min(next_q1_value, next_q2_value)
            min_next_q_value[self.dones] = 0.0
            target_q_v = self.rewards + self.config.GAMMA ** self.config.N_STEP * min_next_q_value
            if self.config.TARGET_VALUE_NORMALIZE:
                target_q_v = (target_q_v - torch.mean(target_q_v)) / (torch.std(target_q_v) + 1e-7)

        q1_value, q2_value = self.critic_model.q(self.observations, self.actions)

        critic_loss_each = (self.config.LOSS_FUNCTION(q1_value, target_q_v.detach(),
                                                      reduction="none") + self.config.LOSS_FUNCTION(q2_value,
                                                                                                    target_q_v.detach(),
                                                                                                    reduction="none")) / 2.0
        if self.config.USE_PER:
            critic_loss_each *= torch.FloatTensor(self.important_sampling_weights).to(self.config.DEVICE)[:, None]

        critic_loss = critic_loss_each.mean()

        self.encoder_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params_list)
        self.critic_optimizer.step()
        self.encoder_optimizer.step()

        self.last_critic_loss.value = critic_loss.item()

        # TAU: 0.005
        self.soft_synchronize_models(
            source_model=self.critic_model, target_model=self.target_critic_model, tau=self.config.TAU
        )
        count_training_steps += 1

        if training_steps_v % self.config.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP == 0:
            stddev = 1.0
            mu = self.actor_model.pi(self.next_observations)
            mu = torch.tanh(mu)
            std = torch.ones_like(mu) * stddev
            dist = drqv2_utils.TruncatedNormal(mu, std)
            action = dist.sample(clip=1.0)
            q1_value, q2_value = self.critic_model.q(self.observations, action)
            actor_objective = torch.min(q1_value, q2_value).mean()
            actor_loss = -1.0 * actor_objective

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params_list)
            self.actor_optimizer.step()

            self.last_actor_objective.value = actor_objective.item()

            # TAU: 0.005
            self.soft_synchronize_models(
                source_model=self.actor_model, target_model=self.target_actor_model, tau=self.config.TAU
            )
        return count_training_steps, critic_loss_each

