import torch.optim as optim

from link_rl.c_models.b_qnet_models import DuelingQNet
from link_rl.c_models_v2.b_q_model_creator import DuelingQModelCreator
from link_rl.d_agents.off_policy.dqn.agent_dqn import AgentDqn


class AgentDuelingDqn(AgentDqn):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentDuelingDqn, self).__init__(observation_space, action_space, config, need_train)

        self._model_creator = DuelingQModelCreator(
            n_input=self.observation_shape[0],
            n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions
        )
        self.q_net = self._model_creator.create_model()
        self.target_q_net = self._model_creator.create_model()

        self.q_net.to(self.config.DEVICE)
        self.target_q_net.to(self.config.DEVICE)

        self.q_net.share_memory()
        self.synchronize_models(source_model=self.q_net, target_model=self.target_q_net)

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=self.config.LEARNING_RATE
        )

        self.model = self.q_net
        self.model.eval()
