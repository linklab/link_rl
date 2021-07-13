# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal, TanhTransform, TransformedDistribution, Categorical

from codes.c_models.discrete_action.discrete_action_model import DiscreteActionModel
from codes.e_utils.common_utils import weights_init_


class DiscreteSACModel(DiscreteActionModel):
    def __init__(self, worker_id, observation_shape, action_n, params, device):
        super(DiscreteSACModel, self).__init__(worker_id, params, device)
        self.__name__ = "DiscreteSACModel"

        num_inputs = observation_shape[0]

        self.base = DiscreteSoftActorCriticMLPBase(num_inputs=num_inputs, action_n=action_n, params=params)

    def forward(self, inputs, agent_state):
        probs, _ = self.base.forward_actor(inputs, agent_state)
        return probs

    def re_parameterization_trick_sample(self, state):
        probs, _ = self.base.forward_actor(state)
        dist = Categorical(probs=probs)
        action_v = dist.sample()  # for reparameterization trick (mean + std * N(0,1))
        log_probs = dist.log_prob(action_v).sum(dim=-1, keepdim=True)

        # action_v.shape: [128, 1]
        # log_prob.shape: [128, 1]
        return action_v, log_probs


class DiscreteSoftActorCriticMLPBase(nn.Module):
    def __init__(self, num_inputs, action_n, params):
        super(DiscreteSoftActorCriticMLPBase, self).__init__()
        self.__name__ = "DiscreteSoftActorCriticMLPBase"

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_1_size),
            nn.GELU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.GELU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.GELU(),
            nn.Linear(self.hidden_3_size, action_n)
        )

        self.twinq = TwinQMLPBase(num_inputs, action_n, params)

        self.layers_info = {'actor': self.actor, 'twinq': self.twinq}

        self.actor_params = list(self.actor.parameters())
        self.twinq_params = list(self.twinq.parameters())

        self.train()

    def forward_actor(self, states, agent_states=None):
        x = self.actor(states)
        probs = F.softmax(x, dim=-1)

        return probs, agent_states

    def forward_critic(self, states, action, agent_states=None):
        q1_v, q2_v = self.twinq.forward(states, action)

        return q1_v, q2_v, agent_states


class TwinQMLPBase(nn.Module):
    def __init__(self, num_inputs, action_n, params):
        super(TwinQMLPBase, self).__init__()

        self.hidden_1_size = params.HIDDEN_1_SIZE
        self.hidden_2_size = params.HIDDEN_2_SIZE
        self.hidden_3_size = params.HIDDEN_3_SIZE

        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + 1, self.hidden_1_size),
            nn.GELU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.GELU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.GELU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + 1, self.hidden_1_size),
            nn.GELU(),
            nn.Linear(self.hidden_1_size, self.hidden_2_size),
            nn.GELU(),
            nn.Linear(self.hidden_2_size, self.hidden_3_size),
            nn.GELU(),
            nn.Linear(self.hidden_3_size, 1),
        )

        self.apply(weights_init_)

    def forward(self, obs, act):
        act = act.unsqueeze(dim=-1)
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)
