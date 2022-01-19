import torch.optim as optim

from c_models.b_qnet_models import DuelingQNet
from d_agents.off_policy.dqn.agent_double_dqn import AgentDoubleDqn


class AgentDoubleDuelingDqn(AgentDoubleDqn):
    def __init__(self, observation_space, action_space, parameter):
        super(AgentDoubleDuelingDqn, self).__init__(observation_space, action_space, parameter)

        self.q_net = DuelingQNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions, parameter=parameter
        ).to(self.parameter.DEVICE)

        self.target_q_net = DuelingQNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions, parameter=parameter
        ).to(self.parameter.DEVICE)

        self.q_net.share_memory()
        self.synchronize_models(source_model=self.q_net, target_model=self.target_q_net)

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=self.parameter.LEARNING_RATE
        )

        self.model = self.q_net
