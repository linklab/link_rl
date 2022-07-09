# https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
# https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC.py
# PAPER: https://arxiv.org/abs/1812.05905
# https://www.pair.toronto.edu/csc2621-w20/assets/slides/lec4_sac.pdf
# https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/
import torch.optim as optim
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.distributions import Normal, TransformedDistribution, TanhTransform

from link_rl.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from link_rl.g_utils.types import AgentMode


class AgentSac(OffPolicyAgent):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentSac, self).__init__(observation_space, action_space, config, need_train)

        # models
        self.actor_model, self.critic_model = self._model_creator.create_model()
        _, self.target_critic_model = self._model_creator.create_model()

        # to(device)
        self.actor_model.to(self.config.DEVICE)
        self.critic_model.to(self.config.DEVICE)
        self.target_critic_model.to(self.config.DEVICE)

        # Access
        self.model = self.actor_model

        # sync models
        self.synchronize_models(source_model=self.critic_model, target_model=self.target_critic_model)

        # share memory
        self.actor_model.share_memory()
        self.critic_model.share_memory()
        self.actor_model.eval()
        self.critic_model.eval()

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.config.LEARNING_RATE)

        # training step
        self.training_step = 0

        # loss
        self.alpha = mp.Value('d', 0.0)
        self.min_alpha = torch.tensor(self.config.MIN_ALPHA, device=self.config.DEVICE)

        if self.config.AUTOMATIC_ENTROPY_TEMPERATURE_TUNING:
            # self.minimum_expected_entropy = -8 for ant_bullet env.
            # it is the desired minimum expected entropy
            self.minimum_expected_entropy = -1.0 * torch.prod(torch.Tensor(action_space.shape).to(self.config.DEVICE)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.config.DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.ALPHA_LEARNING_RATE)
            self.alpha.value = self.log_alpha.exp()  # 초기에는 무조건 1.0으로 시작함.
        else:
            self.alpha.value = self.config.DEFAULT_ALPHA

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)
        self.last_entropy = mp.Value('d', 0.0)

    def actor_forward(self, obs):
        x = self.encoder(obs)
        mu, var = self.actor_model(x)
        return mu, var

    def critic_forward(self, obs, action):
        x = self.encoder(obs)
        q1, q2 = self.critic_model(x, action)
        return q1, q2

    def target_critic_forward(self, obs, action):
        x = self.target_encoder(obs)
        q1, q2 = self.target_critic_model(x, action)
        return q1, q2

    @torch.no_grad()
    def get_action(self, obs, mode=AgentMode.TRAIN):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.config.DEVICE)

        mu_v, var_v = self.actor_forward(obs)

        if mode == AgentMode.TRAIN:
            # actions = np.random.normal(
            #     loc=mu_v.detach().cpu().numpy(), scale=torch.sqrt(var_v).detach().cpu().numpy()
            # )

            dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))
            actions = dist.sample().detach().cpu().numpy()
        else:
            actions = mu_v.detach().cpu().numpy()

        actions = np.clip(a=actions, a_min=self.np_minus_ones, a_max=self.np_plus_ones)

        return actions

    def train_sac(self, training_steps_v):
        count_training_steps = 0

        ############################
        #  Critic Training - BEGIN #
        ############################
        next_mu_v, next_var_v = self.actor_forward(self.next_observations)

        # next_actions_v = torch.normal(mean=next_mu_v, std=torch.sqrt(next_var_v))
        # next_actions_v = torch.clamp(next_actions_v, min=self.torch_minus_ones, max=self.torch_plus_ones)
        # next_log_prob_v = self.calc_log_prob(next_mu_v, next_var_v, next_actions_v)

        next_dist = Normal(loc=next_mu_v, scale=torch.sqrt(next_var_v))
        next_actions_v = next_dist.sample()
        next_actions_v = torch.clamp(next_actions_v, min=self.torch_minus_ones, max=self.torch_plus_ones)
        next_log_prob_v = next_dist.log_prob(value=next_actions_v).sum(dim=-1, keepdim=True)

        with torch.no_grad():
            next_q1_values, next_q2_values = self.target_critic_forward(self.next_observations, next_actions_v)
            next_q_values = torch.min(next_q1_values, next_q2_values)
            next_q_values = next_q_values - self.alpha.value * next_log_prob_v  # ALPHA!!!

            next_q_values[self.dones] = 0.0
            # target_values.shape: (32, 1)

            target_values = self.rewards + (self.config.GAMMA ** self.config.N_STEP) * next_q_values
            # normalize target_values
            if self.config.TARGET_VALUE_NORMALIZE:
                target_values = (target_values - torch.mean(target_values)) / (torch.std(target_values) + 1e-7)

        # values.shape: (32, 1)
        q1_values, q2_values = self.critic_forward(self.observations, self.actions)

        # critic_loss.shape: ()
        critic_loss_each = (self.config.LOSS_FUNCTION(q1_values, target_values.detach(), reduction="none") + self.config.LOSS_FUNCTION(q2_values, target_values.detach(), reduction="none")) / 2.0

        if self.config.USE_PER:
            critic_loss_each *= torch.FloatTensor(self.important_sampling_weights).to(self.config.DEVICE)[:, None]
            self.last_loss_for_per = critic_loss_each

        critic_loss = critic_loss_each.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.parameters())
        self.critic_optimizer.step()

        if self.encoder_is_not_identity:
            self.last_loss_for_encoder = critic_loss

        self.last_critic_loss.value = critic_loss.item()

        # TAU: 0.005
        self.soft_synchronize_models(
            source_model=self.critic_model, target_model=self.target_critic_model, tau=self.config.TAU
        )
        ##########################
        #  Critic Training - END #
        ##########################

        ###########################
        #  Actor Training - BEGIN #
        ###########################
        if training_steps_v % self.config.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP == 0:
            action_v, log_prob_v, entropy_v = self._re_parameterization_trick_sample(self.observations)
            q1_value, q2_value = self.critic_forward(self.observations, action_v)
            actor_objectives = torch.min(q1_value, q2_value) - self.alpha.value * log_prob_v
            actor_objectives = actor_objectives.mean()
            loss_actor_v = -1.0 * actor_objectives

            self.actor_optimizer.zero_grad()
            loss_actor_v.backward()
            self.clip_actor_model_parameter_grad_value(self.actor_model.parameters())
            self.actor_optimizer.step()

            self.last_actor_objective.value = actor_objectives.item()

            #  Alpha Training - BEGIN
            if self.config.AUTOMATIC_ENTROPY_TEMPERATURE_TUNING:
                alpha_loss = -1.0 * (self.log_alpha.exp() * (log_prob_v + self.minimum_expected_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha.value = torch.max(self.log_alpha.exp(), self.min_alpha)
            # Alpha Training - END

            self.last_entropy.value = entropy_v.mean().item()
        #########################
        #  Actor Training - END #
        #########################

        count_training_steps += 1

        return count_training_steps

    def _re_parameterization_trick_sample(self, obs):
        mu, var = self.actor_forward(obs)

        dist = Normal(loc=mu, scale=torch.sqrt(var))
        dist = TransformedDistribution(
            base_distribution=dist,
            transforms=TanhTransform(cache_size=1)
        )

        action_v = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))

        log_probs = dist.log_prob(action_v).sum(dim=-1, keepdim=True)

        # action_v.shape: (128, 8)
        # log_prob.shape: (128, 1)
        # entropy.shape: ()
        entropy = 0.5 * (torch.log(2.0 * np.pi * var) + 1.0).sum(dim=-1)

        return action_v, log_probs, entropy
