# https://spinningup.openai.com/en/latest/algorithms/sac.html
# https://github.com/pranz24/pytorch-soft-actor-critic
# https://github.com/ku2482/soft-actor-critic.pytorch/blob/master/code/agent.py
# https://github.com/cyoon1729/Policy-Gradient-Methods/blob/master/sac/sac2019.py
import torch
import torch.nn.utils as nn_utils

from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import show_info


class AgentSAC(OffPolicyAgent):
    """
    """
    def __init__(self, worker_id, action_shape, params, device):
        super(AgentSAC, self).__init__(
            worker_id=worker_id, action_shape=action_shape, params=params, device=device
        )

        self.alpha = torch.tensor(self.params.ALPHA).to(self.device)

        self.target_entropy = None
        self.log_alpha = None
        self.alpha_optimizer = None

    def reset_alpha(self):
        # if self.params.ENTROPY_TUNING:
        # Target entropy is -|A|.
        self.target_entropy = -torch.prod(torch.Tensor(self.action_shape))

        #print(self.target_entropy, "!")

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.tensor([self.params.ALPHA], requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha)
        self.alpha_optimizer = rl_utils.get_optimizer(
            parameters=[self.log_alpha],
            learning_rate=self.params.ALPHA_LEARNING_RATE,
            params=self.params
        )

    def __call__(self, state, agent_states=None):
        raise NotImplementedError()

    def adjust_alpha(self, log_prob_v):
        self.alpha_optimizer.zero_grad()
        # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.

        entropy_loss = -1.0 * (self.log_alpha * (self.target_entropy + log_prob_v).detach()).mean()
        entropy_loss.backward()
        nn_utils.clip_grad_norm_([self.log_alpha], self.params.CLIP_GRAD)
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha)

    def adjust_alpha_for_discrete_action(self, probs, log_prob_v):
        self.alpha_optimizer.zero_grad()
        # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.

        entropy_loss = -1.0 * (probs * self.log_alpha * (self.target_entropy - log_prob_v)).mean().detach()
        entropy_loss.backward()
        nn_utils.clip_grad_norm_([self.log_alpha], self.params.CLIP_GRAD)
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha)