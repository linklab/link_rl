# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
from torch.distributions import Normal

from codes.c_models.base_model import BaseModel
from codes.e_utils.common_utils import weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

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

        return action_v, log_prob,


class SoftActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, num_outputs, params):
        super(SoftActorCriticMLPBase, self).__init__()
        self.__name__ = "SoftActorCriticMLPBase"
        self.params = params

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.actor = GaussianActorMLPBase(num_inputs, num_outputs, params)
        self.twinq = TwinMLPBase(num_inputs, num_outputs, params)

        self.layers_info = {'actor': self.actor, 'twinq': self.twinq}

        self.actor_params = list(self.actor.parameters())
        self.twinq_params = list(self.twinq.parameters())

        self.train()


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


# class ActorMLPBase(nn.Module):
#     def __init__(self, num_inputs, num_outputs, params):
#         super(ActorMLPBase, self).__init__()
#         self.__name__ = "ActorMLPBase"
#         self.params = params
#
#         self.hidden_1_size = params.HIDDEN_1_SIZE
#         self.hidden_2_size = params.HIDDEN_2_SIZE
#         self.hidden_3_size = params.HIDDEN_3_SIZE
#
#         self.common = nn.Sequential(
#             nn.Linear(num_inputs, self.hidden_1_size),
#             nn.GELU(),
#             nn.Linear(self.hidden_1_size, self.hidden_2_size),
#             nn.GELU(),
#             nn.Linear(self.hidden_2_size, self.hidden_3_size),
#             nn.GELU()
#         )
#
#         self.mu = nn.Sequential(
#             nn.Linear(self.hidden_3_size, num_outputs),
#             nn.Tanh()
#         )
#
#         self.logstd = nn.Sequential(
#             nn.Linear(self.hidden_3_size, num_outputs),
#             nn.Softplus()
#         )
#
#     @staticmethod
#     def init_weights(m):
#         if type(m) == nn.Linear:
#             torch.nn.init.kaiming_normal_(m.weight)
#
#     def forward(self, inputs):
#         mu_v = self.mu(self.common(inputs))
#         logstd_v = self.logstd(self.common(inputs))
#         return mu_v, logstd_v


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