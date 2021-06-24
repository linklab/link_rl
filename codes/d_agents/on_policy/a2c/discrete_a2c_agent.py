import torch
import torch.nn.functional as F

from codes.c_models.discrete_action.discrete_actor_critic_model import DiscreteActorCriticModel
from codes.d_agents.on_policy.a2c.a2c_agent import AgentA2C
from codes.d_agents.on_policy.on_policy_action_selector import DiscreteCategoricalActionSelector
from codes.e_utils import rl_utils
from codes.e_utils.names import DeepLearningModelName


class AgentDiscreteA2C(AgentA2C):
    """
    """
    def __init__(
            self, worker_id, input_shape, action_shape, num_outputs, params, device
    ):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_MLP,
            DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_CNN
        ]
        super(AgentDiscreteA2C, self).__init__(
            worker_id, input_shape, action_shape, num_outputs, params, device
        )

        self.__name__ = "AgentDiscreteA2C"
        self.train_action_selector = DiscreteCategoricalActionSelector()
        self.test_and_play_action_selector = DiscreteCategoricalActionSelector()

        self.model = DiscreteActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.test_model = DiscreteActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

    def __call__(self, states, critics=None):
        return self.discrete_call(states, critics)

    def on_train(self, step_idx, expected_model_version):
        batch = self.buffer.sample_all_for_on_policy(expected_model_version)

        # states_v.shape: (32, 3)
        # actions_v.shape: (32, 1)
        # target_action_values_v.shape: (32,)
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(batch, self.model, self.params)

        probs_v, value_v = self.model.base.forward(states_v)

        # Critic Optimization
        loss_critic_v = F.mse_loss(input=value_v.squeeze(-1), target=target_action_values_v.detach())

        # advantage_v.shape: (32,)
        advantage_v = target_action_values_v - value_v.squeeze(-1)
        #print(target_action_values_v, value_v.squeeze(-1), advantage_v)

        log_pi_action_v = torch.log(probs_v.gather(dim=1, index=actions_v.unsqueeze(-1)) + 1e-5).squeeze(-1)
        reinforced_log_pi_action_v = advantage_v.detach() * log_pi_action_v

        loss_actor_v = -1.0 * reinforced_log_pi_action_v.mean()

        log_pi_v = torch.log(probs_v + 1e-5)
        loss_entropy_v = (probs_v * log_pi_v).sum(dim=1).mean()
        # loss_actor_v를 작아지도록 만듦 --> log_pi_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦

        return self.backward_and_step(loss_critic_v, loss_entropy_v, loss_actor_v)
