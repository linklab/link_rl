import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.actions import ProbabilityActionSelector
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentDiscreteA2C(OnPolicyAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, num_outputs,
            train_action_selector, test_and_play_action_selector, params, device
    ):
        assert isinstance(train_action_selector, ProbabilityActionSelector)
        assert isinstance(test_and_play_action_selector, ProbabilityActionSelector)
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_MLP,
            DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_CNN
        ]

        super(AgentDiscreteA2C, self).__init__(train_action_selector, test_and_play_action_selector, params, device)

        self.__name__ = "AgentDiscreteA2C"
        self.worker_id = worker_id

        self.model = rl_utils.get_rl_model(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs, params=params, device=self.device
        )

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.BATCH_SIZE
        )

    def __call__(self, states, critics=None):
        if not isinstance(states, torch.FloatTensor):
            states = float32_preprocessor(states).to(self.device)

        logits_v = self.model.base.forward_actor(states)

        probs_v = F.softmax(logits_v, dim=1)

        probs = probs_v.data.cpu().numpy()

        if self.agent_mode == AgentMode.TRAIN:
            actions = np.array(self.train_action_selector(probs))
        else:
            actions = np.array(self.test_and_play_action_selector(probs))

        critics = torch.zeros(size=probs_v.size())
        return actions, critics

    def train(self, step_idx):
        # Lucky Episode에서 얻어낸 batch를 통해 학습할 때와, Unlucky Episode에서 얻어낸 batch를 통해 학습할 때마다 NN의 파라미터들이
        # 서로 다른 방향으로 반복적으로 휩쓸려가듯이 학습이 됨 --> Gradients의 Variance가 매우 큼
        batch = self.buffer.sample(batch_size=None)

        # states_v.shape: (32, 3)
        # actions_v.shape: (32, 1)
        # target_action_values_v.shape: (32,)
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(batch, self.model, self.params)

        logits_v, value_v = self.model(states_v)

        # Critic Optimization
        self.optimizer.zero_grad()
        loss_critic_v = F.mse_loss(input=value_v.squeeze(-1), target=target_action_values_v)

        #nn_utils.clip_grad_norm_(self.model.base.critic.parameters(), self.params.CLIP_GRAD)

        # advantage_v.shape: (32,)
        advantage_v = target_action_values_v - value_v.squeeze(-1).detach()
        log_pi_v = F.log_softmax(logits_v, dim=1)
        log_pi_action_v = log_pi_v.gather(dim=1, index=actions_v.unsqueeze(-1)).squeeze(-1)
        reinforced_log_pi_action_v = advantage_v * log_pi_action_v

        #print(actions_v.size(), advantage_v.size(), log_pi_v.size(), log_pi_action_v.size(), reinforced_log_pi_action_v.size())

        loss_actor_v = -1.0 * reinforced_log_pi_action_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_v = -1.0 * (prob_v * log_pi_v).sum(dim=1).mean()
        loss_entropy_v = -1.0 * self.params.ENTROPY_LOSS_WEIGHT * entropy_v

        # loss_actor_v를 작아지도록 만듦 --> log_pi_v.mean()가 커지도록 만듦
        # loss_entropy_v를 작아지도록 만듦 --> entropy_v가 커지도록 만듦
        loss_v = loss_critic_v + loss_actor_v + loss_entropy_v

        loss_v.backward()
        #nn_utils.clip_grad_norm_(self.model.base.actor.parameters(), self.params.CLIP_GRAD)
        self.optimizer.step()

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0