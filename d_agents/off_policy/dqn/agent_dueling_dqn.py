from c_models.b_qnet_models import DuelingQNet
from d_agents.off_policy.dqn.agent_dqn import AgentDqn


class AgentDuelingDqn(AgentDqn):
    def __init__(self, observation_space, action_space, parameter):
        super(AgentDuelingDqn, self).__init__(observation_space, action_space, parameter)

        self.q_net = DuelingQNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions, parameter=parameter
        ).to(self.parameter.DEVICE)

        self.target_q_net = DuelingQNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions, parameter=parameter
        ).to(self.parameter.DEVICE)

        self.model = self.q_net
