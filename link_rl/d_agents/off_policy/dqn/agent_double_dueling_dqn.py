from link_rl.c_models.b_qnet_models import DuelingQNet
from link_rl.d_agents.off_policy.dqn.agent_double_dqn import AgentDoubleDqn
from link_rl.d_agents.off_policy.dqn.agent_dueling_dqn import AgentDuelingDqn


class AgentDoubleDuelingDqn(AgentDoubleDqn, AgentDuelingDqn):
    def __init__(self, observation_space, action_space, config):
        AgentDoubleDqn.__init__(self, observation_space, action_space, config)
        AgentDuelingDqn.__init__(self, observation_space, action_space, config)

        assert isinstance(self.q_net, DuelingQNet)
        assert isinstance(self.target_q_net, DuelingQNet)
