import numpy as np
import torch.optim as optim
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp

from c_models.c_actor_models import DiscreteActorModel, ContinuousStochasticActorModel
from d_agents.on_policy.on_policy_agent import OnPolicyAgent


class AgentReinforce(OnPolicyAgent):
    def __init__(self, observation_space, action_space, config):
        super(AgentReinforce, self).__init__(observation_space, action_space, config)

        if isinstance(self.action_space, Discrete):
            self.actor_model = DiscreteActorModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
                n_discrete_actions=self.n_discrete_actions, config=config
            ).to(self.config.DEVICE)
        elif isinstance(self.action_space, Box):
            self.actor_model = ContinuousStochasticActorModel(
                observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, config=config
            ).to(self.config.DEVICE)
        else:
            raise ValueError()

        self.actor_model.share_memory()

        self.optimizer = optim.Adam(self.actor_model.parameters(), lr=self.config.LEARNING_RATE)

        self.model = self.actor_model  # 에이전트 밖에서는 model이라는 이름으로 제어 모델 접근
        self.model.eval()

        self.last_log_policy_objective = mp.Value('d', 0.0)

    def train_reinforce(self):
        count_training_steps = 0

        # The episodes of low number of steps is ignored and not used for train
        if len(self.observations) < 10:
            return count_training_steps

        returns = self.get_returns()

        if isinstance(self.action_space, Discrete):
            action_probs = self.actor_model.pi(self.observations)
            dist = Categorical(probs=action_probs)

            # dist.log_prob(value=self.actions.squeeze(dim=-1)).shape: (32,)
            # return_lst.shape: (32,)
            log_pi_returns = dist.log_prob(value=self.actions.squeeze(dim=-1)) * returns
            #print(log_pi_returns, "!!!")

        elif isinstance(self.action_space, Box):
            mu_v, var_v = self.actor_model.pi(self.observations)
            dist = Normal(loc=mu_v, scale=torch.sqrt(var_v))

            # dist.log_prob(value=self.actions).sum(dim=-1).shape: (32,)
            # return_lst.shape: (32,)
            log_pi_returns = dist.log_prob(value=self.actions).sum(dim=-1) * returns

        else:
            raise ValueError()

        log_policy_objective = torch.mean(log_pi_returns)

        loss = -1.0 * log_policy_objective

        self.optimizer.zero_grad()
        loss.backward()
        #self.clip_actor_model_parameter_grad_value(self.actor_model.actor_params_list)
        self.optimizer.step()

        self.last_log_policy_objective.value = log_policy_objective.item()

        count_training_steps += 1

        return count_training_steps
