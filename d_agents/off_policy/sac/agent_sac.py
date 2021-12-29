import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal

from c_models.e_ddpg_models import DiscreteDdpgModel, ContinuousDdpgModel
from c_models.g_sac_models import ContinuousSacModel, DiscreteSacModel
from d_agents.agent import Agent
from g_utils.commons import EpsilonTracker
from g_utils.types import AgentMode, ModelType


class AgentSac(Agent):
    def __init__(self, observation_space, action_space, device, parameter):
        super(AgentSac, self).__init__(observation_space, action_space, device, parameter)

        if isinstance(self.action_space, Discrete):
            self.sac_model = DiscreteSacModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
            )

            self.target_sac_model = DiscreteSacModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
            )
        elif isinstance(self.action_space, Box):
            self.action_bound_low = np.expand_dims(self.action_space.low, axis=0)
            self.action_bound_high = np.expand_dims(self.action_space.high, axis=0)

            self.action_scale_factor = np.max(np.maximum(
                np.absolute(self.action_bound_low), np.absolute(self.action_bound_high)
            ), axis=-1)[0]

            self.sac_model = ContinuousSacModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            )

            self.target_sac_model = ContinuousSacModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                device=device, parameter=parameter
            )
        else:
            raise ValueError()

        self.actor_model = self.sac_model.actor_model
        self.critic_model = self.sac_model.critic_model

        self.model = self.actor_model

        self.target_critic_model = self.target_sac_model.critic_model

        self.actor_model.share_memory()
        self.critic_model.share_memory()

        self.synchronize_models(source_model=self.critic_model, target_model=self.target_critic_model)

        self.actor_optimizer = optim.Adam(self.actor_model.actor_params, lr=self.parameter.LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_model.critic_params, lr=self.parameter.LEARNING_RATE)

        self.training_steps = 0

        self.last_critic_loss = mp.Value('d', 0.0)
        self.last_actor_objective = mp.Value('d', 0.0)
        self.alpha = self.parameter.ALPHA

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
            mu_v, std_v = self.actor_model.pi(obs)
            mu_v = mu_v * self.action_scale_factor

            if mode == AgentMode.TRAIN:
                dist = Normal(loc=mu_v, scale=std_v + 1.0e-7)
                actions = dist.sample()
            else:
                actions = mu_v.detach()

            actions = np.clip(actions.cpu().numpy(), self.action_bound_low, self.action_bound_high)
            return actions
        else:
            raise ValueError()

    def train_sac(self, training_steps_v):
        # observations.shape: torch.Size([32, 4, 84, 84]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4, 84, 84]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])

        observations, actions, next_observations, rewards, dones = self.buffer.sample(
            batch_size=self.parameter.BATCH_SIZE, device=self.device
        )

        ###################################
        #  Critic (Value) 손실 산출 - BEGIN #
        ###################################
        if isinstance(self.action_space, Discrete):
            pass
        elif isinstance(self.action_space, Box):
            next_mu_v, std_v = self.actor_model.pi(next_observations)
            dist = Normal(loc=next_mu_v, scale=std_v + 1.0e-7)
            next_actions_v = dist.sample()
            next_log_prob_v = dist.log_prob(next_actions_v).sum(dim=-1, keepdim=True)

        next_q1_v, next_q2_v = self.target_critic_model.q(next_observations, next_actions_v)
        next_values = torch.min(next_q1_v, next_q2_v).detach().cpu().numpy()[:, 0]
        next_log_prob_v = self.alpha * next_log_prob_v
        next_values -= next_log_prob_v.squeeze(-1).detach().cpu().numpy()

        td_target_value_lst = []

        for reward, next_value, done in zip(rewards, next_values, dones):
            td_target = reward + self.parameter.GAMMA ** self.parameter.N_STEP * next_value * (0.0 if done else 1.0)
            td_target_value_lst.append(td_target.detach())

        # td_target_values.shape: (32, 1)
        td_target_values = torch.tensor(td_target_value_lst, dtype=torch.float32, device=self.device).unsqueeze(dim=-1)

        # values.shape: (32, 1)
        q1_v, q2_v = self.critic_model.q(observations, actions)

        # critic_loss.shape: ()
        critic_loss = F.mse_loss(q1_v.squeeze(dim=-1), td_target_values) + \
                      F.mse_loss(q2_v.squeeze(dim=-1), td_target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic_model.critic_params, self.parameter.CLIP_GRADIENT_VALUE)
        self.critic_optimizer.step()
        ###################################
        #  Critic (Value)  Loss 산출 - END #
        ###################################

        ################################
        #  Actor Objective 산출 - BEGIN #
        ################################
        re_parameterization_trick_action_v, log_prob_v = self.sac_model.re_parameterization_trick_sample((observations))
        q1_v, q2_v = self.critic_model.q(observations, actions)
        objectives_v = torch.div(torch.add(q1_v, q2_v), 2.0) - self.alpha * log_prob_v

        loss_actor_v = -1.0 * objectives_v.mean()

        self.actor_optimizer.zero_grad()
        loss_actor_v.backward()
        torch.nn.utils.clip_grad_value_(self.actor_model.actor_params, self.parameter.CLIP_GRADIENT_VALUE)
        self.actor_optimizer.step()
        ##############################
        #  Actor Objective 산출 - END #
        ##############################

        # sync
        # if training_steps_v % self.parameter.TARGET_SYNC_INTERVAL_TRAINING_STEPS == 0:
        #     self.synchronize_models(source_model=self.sac_model, target_model=self.target_sac_model)
        self.soft_synchronize_models(
            source_model=self.critic_model, target_model=self.target_critic_model,
            tau=self.parameter.TAU
        )  # TAU: 0.0001

        self.last_critic_loss.value = critic_loss.item()
        self.last_actor_objective.value = -loss_actor_v.item()
