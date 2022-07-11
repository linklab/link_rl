from link_rl.e_agents.off_policy.dqn.agent_double_dqn import AgentDoubleDqn
from link_rl.e_agents.off_policy.dqn.agent_dueling_dqn import AgentDuelingDqn


class AgentDoubleDuelingDqn(AgentDoubleDqn, AgentDuelingDqn):
    def __init__(self, observation_space, action_space, config, need_train):
        AgentDoubleDqn.__init__(self, observation_space, action_space, config, need_train)
        AgentDuelingDqn.__init__(self, observation_space, action_space, config, need_train)

        # assert isinstance(self.q_net, DuelingQNet)
        # assert isinstance(self.target_q_net, DuelingQNet)
