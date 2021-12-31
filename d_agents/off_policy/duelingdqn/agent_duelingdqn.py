from c_models.b_qnet_models import DuelingQNet
from d_agents.off_policy.dqn.agent_dqn import AgentDqn


class AgentDuelingdqn(AgentDqn):
    def __init__(self, observation_space, action_space, device, parameter):
        super(AgentDuelingdqn, self).__init__(observation_space, action_space, device, parameter)

        self.q_net = DuelingQNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
        ).to(device)

        self.target_q_net = DuelingQNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions, device=device, parameter=parameter
        ).to(device)

        self.model = self.q_net