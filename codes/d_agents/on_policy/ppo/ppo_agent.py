import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal
import torch.nn.utils as nn_utils

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.actions import ContinuousNormalActionSelector
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentPPO(OnPolicyAgent):
    """
    """
    def __init__(
            self, worker_id, params, device
    ):
        assert params.N_STEP == 1  # GAE will consider various N_STEPs

        super(AgentPPO, self).__init__(worker_id=worker_id, params=params, device=device)

        self.train_action_selector = None
        self.test_and_play_action_selector = None
        self.model = None
        self.optimizer = None
        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.PPO_TRAJECTORY_SIZE
        )

    def __call__(self, states, critics=None):
        raise NotImplementedError

    def train(self, step_idx):
        raise NotImplementedError

    def backward_and_step_in_trajectory(self, loss_critic_v, loss_entropy_v, loss_actor_v):
        self.optimizer.zero_grad()
        # loss_v = loss_actor_v + \
        #          self.params.CRITIC_LOSS_WEIGHT * loss_critic_v + \
        #          self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v
        # loss_v.backward()

        loss_actor_v.backward(retain_graph=True)
        (loss_critic_v + self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v).backward()
        #nn_utils.clip_grad_norm_(self.model.base.parameters(), self.params.CLIP_GRAD)
        self.optimizer.step()