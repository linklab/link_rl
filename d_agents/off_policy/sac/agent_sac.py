# https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
# https://github.com/BY571/Soft-Actor-Critic-and-Extensions/blob/master/SAC.py
# PAPER: https://arxiv.org/abs/1812.05905
# https://www.pair.toronto.edu/csc2621-w20/assets/slides/lec4_sac.pdf
# https://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal

from c_models.g_sac_models import ContinuousSacModel, DiscreteSacModel
from d_agents.agent import Agent
from g_utils.types import AgentMode


class AgentSac(Agent):
    def __init__(self, observation_space, action_space, parameter):
        super(AgentSac, self).__init__(observation_space, action_space, parameter)

        if isinstance(self.action_space, Discrete):
            self.sac_model = DiscreteSacModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, parameter=parameter
            )

            self.target_sac_model = DiscreteSacModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, parameter=parameter, is_target_model=True
            )
        elif isinstance(self.action_space, Box):
            self.sac_model = ContinuousSacModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, parameter=parameter
            )

            self.target_sac_model = ContinuousSacModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, parameter=parameter,
                is_target_model=True
            )
        else:
            raise ValueError()

        self.model = self.sac_model.actor_model

        self.actor_model = self.sac_model.actor_model
        self.critic_model = self.sac_model.critic_model

        self.target_critic_model = self.target_sac_model.critic_model
        self.synchronize_models(source_model=self.critic_model, target_model=self.target_critic_model)

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=self.parameter.ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=self.parameter.LEARNING_RATE)

        self.training_steps = 0

        self.alpha = mp.Value('d', 0.0)
        self.min_alpha = torch.tensor(self.parameter.MIN_ALPHA, device=self.parameter.DEVICE)

        if self.parameter.AUTOMATIC_ENTROPY_TEMPERATURE_TUNING:
            # self.minimum_expected_entropy = -8 for ant_bullet env.
            # it is the desired minimum expected entropy
            self.minimum_expected_entropy = -1.0 * torch.prod(torch.Tensor(action_space.shape).to(self.parameter.DEVICE)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.parameter.DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.parameter.ALPHA_LEARNING_RATE)
            self.alpha.value = self.log_alpha.exp()  # 초기에는 무조건 1.0으로 시작함.
        else:
            self.alpha.value = self.parameter.DEFAULT_ALPHA

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        if isinstance(self.action_space, Discrete):
            action_prob = self.actor_model.pi(obs)
            m = Categorical(probs=action_prob)
            if mode == AgentMode.TRAIN:
                action = m.sample()
            else:
                action = torch.argmax(m.probs, dim=-1)
            return action.cpu().numpy()
        elif isinstance(self.action_space, Box):
            mu_v, var_v = self.actor_model.pi(obs)

            if mode == AgentMode.TRAIN:
                actions = np.random.normal(
                    loc=mu_v.detach().cpu().numpy(), scale=torch.sqrt(var_v).detach().cpu().numpy()
                )

                # dist = Normal(loc=mu_v, scale=std_v)
                # actions = dist.sample().detach().cpu().numpy()
            else:
                actions = mu_v.detach().cpu().numpy()

            actions = np.clip(a=actions, a_min=self.np_minus_ones, a_max=self.np_plus_ones)

            return actions
        else:
            raise ValueError()

    def train_sac(self, training_steps_v):
        ############################
        #  Critic Training - BEGIN #
        ############################
        if isinstance(self.action_space, Discrete):
            next_actions_v = None
            next_log_prob_v = None
        elif isinstance(self.action_space, Box):
            next_mu_v, next_var_v = self.actor_model.pi(self.next_observations)

            next_actions_v = torch.normal(mean=next_mu_v, std=torch.sqrt(next_var_v))
            next_actions_v = torch.clamp(next_actions_v, min=self.torch_minus_ones, max=self.torch_plus_ones)
            next_log_prob_v = self.calc_log_prob(next_mu_v, next_var_v, next_actions_v)
            # dist = Normal(loc=next_mu_v, scale=torch.sqrt(var_v))
            # next_actions_v = dist.sample()
            # next_log_prob_v = dist.log_prob(next_actions_v)
        else:
            raise ValueError()

        next_q1_v, next_q2_v = self.target_critic_model.q(self.next_observations, next_actions_v)
        next_values = torch.min(next_q1_v, next_q2_v)
        next_values = next_values - self.alpha.value * next_log_prob_v  # ALPHA!!!
        next_values[self.dones] = 0.0
        # td_target_values.shape: (32, 1)
        td_target_values = self.rewards + self.parameter.GAMMA ** self.parameter.N_STEP * next_values

        # values.shape: (32, 1)
        q1_v, q2_v = self.critic_model.q(self.observations, self.actions)
        # critic_loss.shape: ()
        critic_loss = F.mse_loss(q1_v, td_target_values.detach()) + F.mse_loss(q2_v, td_target_values.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.clip_critic_model_parameter_grad_value(self.critic_model.critic_params)
        self.critic_optimizer.step()

        self.last_critic_loss.value = critic_loss.item()
        ##########################
        #  Critic Training - END #
        ##########################

        ###########################
        #  Actor Training - BEGIN #
        ###########################
        if training_steps_v % self.parameter.POLICY_UPDATE_FREQUENCY_PER_TRAINING_STEP == 0:
            re_parameterized_action_v, re_parameterized_log_prob_v = self.sac_model.re_parameterization_trick_sample(
                self.observations
            )
            q1_v, q2_v = self.critic_model.q(self.observations, re_parameterized_action_v)
            objectives_v = torch.min(q1_v, q2_v) - self.alpha.value * re_parameterized_log_prob_v
            objectives_v = objectives_v.mean()
            loss_actor_v = -1.0 * objectives_v

            self.actor_optimizer.zero_grad()
            loss_actor_v.backward()
            self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params)
            self.actor_optimizer.step()

            self.last_actor_objective.value = objectives_v.item()

            #  Alpha Training - BEGIN
            if self.parameter.AUTOMATIC_ENTROPY_TEMPERATURE_TUNING:
                alpha_loss = -1.0 * (self.log_alpha.exp() * (re_parameterized_log_prob_v + self.minimum_expected_entropy).detach()).mean()

                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha.value = torch.max(self.log_alpha.exp(), self.min_alpha)
            # Alpha Training - END
        #########################
        #  Actor Training - END #
        #########################

        self.soft_synchronize_models(
            source_model=self.critic_model, target_model=self.target_critic_model, tau=self.parameter.TAU
        )  # TAU: 0.005

