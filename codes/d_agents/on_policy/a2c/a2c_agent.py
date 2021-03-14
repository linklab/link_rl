from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import torch.nn.utils as nn_utils

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.actions import ContinuousNormalActionSelector
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentA2C(OnPolicyAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, num_outputs, params, device="cpu"
    ):
        super(AgentA2C, self).__init__(worker_id, params, device)
        self.train_action_selector = None
        self.test_and_play_action_selector = None
        self.model = None
        self.optimizer = None
        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.BATCH_SIZE
        )

    def __call__(self, states, critics=None):
        raise NotImplementedError

    # Lucky Episode에서 얻어낸 batch를 통해 학습할 때와, Unlucky Episode에서 얻어낸 batch를 통해 학습할 때마다 NN의 파라미터들이
    # 서로 다른 방향으로 반복적으로 휩쓸려가듯이 학습이 됨 --> Gradients의 Variance가 매우 큼
    def train(self, step_idx):
        raise NotImplementedError

    def backward_and_step(self, loss_critic_v, loss_entropy_v, loss_actor_v):
        self.optimizer.zero_grad()
        loss_actor_v.backward(retain_graph=True)
        (loss_critic_v + self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v).backward()
        nn_utils.clip_grad_norm_(self.model.base.parameters(), self.params.CLIP_GRAD)
        self.optimizer.step()

        gradients = self.model.get_gradients_for_current_parameters()
        self.model.check_gradient_nan_or_zero(gradients)
        self.buffer.clear()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0