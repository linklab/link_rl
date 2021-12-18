import torch.optim as optim
import torch
from torch.distributions import Categorical
import torch.multiprocessing as mp

from c_models.c_policy_models import Policy
from d_agents.agent import Agent
from g_utils.buffers import Buffer
from g_utils.types import AgentMode


class AgentReinforce(Agent):
    def __init__(self, observation_space, action_space, device, parameter):
        super(AgentReinforce, self).__init__(observation_space, action_space, device, parameter)

        self.policy = Policy(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions, device=device
        ).to(device)
        self.policy.share_memory()

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.parameter.LEARNING_RATE)

        self.model = self.policy

        self.last_log_policy_objective = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        action_prob = self.policy.forward(obs)
        m = Categorical(probs=action_prob)

        if mode == AgentMode.TRAIN:
            action = m.sample()
        else:
            action = torch.argmax(m.probs, dim=-1)
        return action.cpu().numpy()

    def train_reinforce(self, buffer):
        observations, actions, _, rewards, _ = buffer.sample(
            batch_size=None, device=self.device
        )

        G = 0
        return_lst = []
        for reward in reversed(rewards):
            G = reward + self.parameter.GAMMA * G
            return_lst.append(G)
        return_lst = torch.tensor(
            return_lst[::-1], dtype=torch.float32, device=self.device
        )

        action_probs = self.policy.forward(observations)
        action_probs_selected = action_probs.gather(dim=-1, index=actions)

        # action_probs_selected.shape: (32, 1)
        # return_lst.shape: (32, 1)
        log_pi_returns = torch.multiply(
            torch.log(action_probs_selected), return_lst
        )
        log_policy_objective = torch.sum(log_pi_returns)
        loss = torch.multiply(log_policy_objective, -1.0)

        loss.backward()
        self.optimizer.step()

        self.last_log_policy_objective.value = log_policy_objective.item()
