import numpy as np
import torch.optim as optim
import torch
from gym.spaces import Discrete, Box
from torch.distributions import Categorical, Normal
import torch.multiprocessing as mp

from c_models.c_policy_models import DiscretePolicyModel, ContinuousPolicyModel
from d_agents.agent import Agent
from g_utils.types import AgentMode


class AgentReinforce(Agent):
    def __init__(self, observation_space, action_space, parameter):
        super(AgentReinforce, self).__init__(observation_space, action_space, parameter)

        assert self.parameter.N_STEP == 1
        assert isinstance(self.action_space, Discrete)

        self.policy = DiscretePolicyModel(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions, parameter=parameter
        ).to(self.parameter.DEVICE)

        self.policy.share_memory()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.parameter.LEARNING_RATE)

        self.model = self.policy  # 에이전트 밖에서는 model이라는 이름으로 제어 모델 접근

        self.last_log_policy_objective = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        action_prob = self.policy.pi(obs)
        m = Categorical(probs=action_prob)

        if mode == AgentMode.TRAIN:
            action = m.sample()
        else:
            action = torch.argmax(m.probs, dim=-1)
        return action.cpu().numpy()

    def train_reinforce(self):
        count_training_steps = 0

        G = 0
        return_lst = []
        for reward in reversed(self.rewards):
            G = reward + self.parameter.GAMMA * G
            return_lst.append(G)
        return_lst = torch.tensor(return_lst[::-1], dtype=torch.float32, device=self.parameter.DEVICE)

        action_probs = self.policy.pi(self.observations)
        action_probs_selected = action_probs.gather(dim=-1, index=self.actions).squeeze(dim=-1)

        # action_probs_selected.shape: (32,)
        # return_lst.shape: (32,)
        # print(action_probs_selected.shape, return_lst.shape, "!!!!!!1")
        log_pi_returns = torch.log(action_probs_selected) * return_lst
        log_policy_objective = torch.sum(log_pi_returns)
        loss = -1.0 * log_policy_objective

        loss.backward()
        self.clip_model_parameter_grad_value(self.policy.parameters())
        self.optimizer.step()

        self.last_log_policy_objective.value = log_policy_objective.item()

        count_training_steps = 1

        return count_training_steps
