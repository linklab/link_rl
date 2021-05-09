# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
from torch.distributions import Normal

from codes.c_models.base_model import BaseModel
from codes.e_utils.common_utils import weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -2


class SoftActorCriticModel(BaseModel):
    def __init__(self, worker_id, input_shape, num_outputs, params, device):
        super(SoftActorCriticModel, self).__init__(worker_id, params, device)
        self.__name__ = "SoftActorCriticModel"

        num_inputs = input_shape[0]

        self.base = SoftActorCriticMLPBase(
            num_inputs=num_inputs, num_outputs=num_outputs, params=self.params
        )

        self.reset_average_gradients()

    def forward(self, inputs):
        if not (type(inputs) is torch.Tensor):
            inputs = torch.tensor([inputs], dtype=torch.float).to(self.device)

        mu, value = self.base.forward(inputs)

        return mu, value

    def sample(self, state):
        mu_v, logstd_v = self.base.actor.forward(state)
        dist = Normal(loc=mu_v, scale=torch.exp(logstd_v))
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action_v = torch.tanh(x_t)

        log_prob = dist.log_prob(x_t)
        log_prob = log_prob.sum(1, keepdim=True)
        # action_v.shape: [128, 1]
        # log_prob.shape: [128, 1]
        return action_v, log_prob


class SoftActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(SoftActorCriticMLPBase, self).__init__()
        self.__name__ = "SoftActorCriticMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.actor = GaussianActorMLPBase(num_inputs, num_outputs, params)

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        self.critic.apply(weights_init_)

        self.twinq = TwinMLPBase(num_inputs, num_outputs, params)

        self.layers_info = {'actor': self.actor, 'critic': self.critic, 'twinq': self.twinq}

        self.actor_params = list(self.actor.parameters())
        self.critic_params = list(self.critic.parameters())
        self.twinq_params = list(self.twinq.parameters())

        self.train()

    def forward_critic(self, inputs):
        return self.critic(inputs)


class GaussianActorMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(GaussianActorMLPBase, self).__init__()

        self.__name__ = "ActorMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.common = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.GELU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.GELU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.GELU()
        )

        self.mu = nn.Sequential(
            nn.Linear(self.hidden_3_size, num_outputs),
            nn.Tanh()
        )

        # SoftPlus is a smooth approximation to the ReLU function and can be used
        # to constrain the output of a machine to always be positive.
        self.logstd = nn.Sequential(
            nn.Linear(self.hidden_3_size, num_outputs),
            nn.Softplus()
        )

        self.apply(weights_init_)

    def forward(self, inputs):
        x = self.common(inputs)
        mu_v = self.mu(x)
        logstd_v = self.logstd(x)
        logstd_v = torch.clamp(logstd_v, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mu_v, logstd_v


class TwinMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(TwinMLPBase, self).__init__()
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, self.hidden_1_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        self.apply(weights_init_)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)
