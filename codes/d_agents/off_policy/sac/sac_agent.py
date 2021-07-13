# https://spinningup.openai.com/en/latest/algorithms/sac.html
# https://github.com/pranz24/pytorch-soft-actor-critic
#https://github.com/ku2482/soft-actor-critic.pytorch/blob/master/code/agent.py
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.distributions import Normal

from codes.a_config._rl_parameters.off_policy.parameter_sac import SACActionSelectorType
from codes.a_config._rl_parameters.off_policy.parameter_td3 import TD3ActionSelectorType
from codes.c_models.continuous_action.continuous_sac_model import ContinuousSACModel
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.d_agents.off_policy.sac.sac_action_selector import ContinuousNormalSACActionSelector, \
    SomeTimesBlowSACActionSelector
from codes.d_agents.off_policy.td3.td3_action_selector import TD3ActionSelector
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import grad_false, show_info
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentSAC(OffPolicyAgent):
    """
    """
    def __init__(self, worker_id, action_shape, params, device):
        super(AgentSAC, self).__init__(
            worker_id=worker_id, action_shape=action_shape, params=params, device=device
        )

        self.alpha = torch.tensor(self.params.ALPHA).to(self.device)

    def reset_alpha(self):
        # if self.params.ENTROPY_TUNING:
        # Target entropy is -|A|.
        self.target_entropy = -torch.prod(torch.Tensor(self.action_shape).to(self.device)).item()

        #print(self.target_entropy, "!")

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = rl_utils.get_optimizer(
            parameters=[self.log_alpha],
            learning_rate=self.params.LEARNING_RATE,
            params=self.params
        )

    def __call__(self, state, agent_states=None):
        raise NotImplementedError()

    def adjust_alpha(self):
        self.alpha_optimizer.zero_grad()
        # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.
        entropy_loss = -1.0 * self.log_alpha * (self.target_entropy - sampled_entropies_v).mean().detach()
        entropy_loss.backward()
        nn_utils.clip_grad_norm_([self.log_alpha], self.params.CLIP_GRAD)
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()