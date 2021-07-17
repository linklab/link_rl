# https://spinningup.openai.com/en/latest/algorithms/sac.html
# https://github.com/pranz24/pytorch-soft-actor-critic
# https://github.com/ku2482/soft-actor-critic.pytorch/blob/master/code/agent.py
# https://github.com/cyoon1729/Policy-Gradient-Methods/blob/master/sac/sac2019.py
# https://github.com/twni2016/Meta-SAC/blob/d60c0fc8c4446bb416ed2da168acef1ce2636f92/meta_sac/mainmeta.py
# https://github.com/twni2016/Meta-SAC/blob/d60c0fc8c4446bb416ed2da168acef1ce2636f92/meta_sac/sacmeta.py
from collections import deque

import torch
import torch.nn.utils as nn_utils
from torch.distributions import Normal

from codes.c_models.continuous_action.continuous_sac_model import ContinuousSACModel
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.e_utils import rl_utils
from codes.e_utils.common_utils import show_info, float32_preprocessor
import numpy as np

from codes.e_utils.names import RLAlgorithmName


class AgentSAC(OffPolicyAgent):
    """
    """
    def __init__(self, worker_id, action_shape, params, device):
        super(AgentSAC, self).__init__(
            worker_id=worker_id, action_shape=action_shape, params=params, device=device
        )

        self.alpha = torch.tensor(self.params.ALPHA).to(self.device)
        self.min_alpha = torch.tensor(self.params.MIN_ALPHA).to(self.device)

        self.observation_shape = None
        self.num_outputs = None

        self.target_entropy = None
        self.alpha_optimizer = None
        self.log_alpha = None

        self.initial_obs_deque = deque(maxlen=params.BATCH_SIZE)

    def reset_alpha(self):
        # if self.params.ENTROPY_TUNING:
        # Target entropy is -|A|.
        self.target_entropy = -torch.tensor(self.params.ENTROPY_TUNING_TARGET_ENTROPY)
        # self.target_entropy = -torch.zeros(self.action_shape, device=self.device).detach()

        #print(self.target_entropy, "!")

        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.tensor(np.log(self.params.ALPHA), requires_grad=True, device=self.device)
        self.alpha = torch.exp(self.log_alpha)

        self.alpha_optimizer = rl_utils.get_optimizer(
            parameters=[self.log_alpha],
            learning_rate=self.params.ALPHA_LEARNING_RATE,
            params=self.params
        )

    def __call__(self, state, agent_states=None):
        raise NotImplementedError()

    def meta_learning_alpha(self):
        if len(self.initial_obs_deque) >= self.params.BATCH_SIZE:
            self.alpha_optimizer.zero_grad()
            self.model.eval()

            initial_obs_batch = torch.FloatTensor(list(self.initial_obs_deque)).to(self.device)

            mu_v, logstd_v = self.model.base.actor(initial_obs_batch)
            dist = Normal(loc=mu_v, scale=torch.exp(logstd_v))
            log_probs_v = dist.log_prob(mu_v).sum(dim=-1, keepdim=True)
            q1_v, q2_v = self.model.base.twinq(initial_obs_batch, mu_v)

            meta_objective = torch.div(torch.add(q1_v, q2_v), 2.0) - self.alpha * log_probs_v
            meta_loss = meta_objective.mean()
            meta_loss.backward()
            nn_utils.clip_grad_norm_([self.log_alpha], self.params.CLIP_GRAD)
            self.alpha_optimizer.step()

            self.alpha = torch.max(torch.exp(self.log_alpha), torch.exp(self.min_alpha))

            self.model.train()

            #print(meta_objective.mean(), self.log_alpha, self.alpha, "!!!!!!!!! - 1")

    def adjust_alpha(self, log_prob_v):
        self.alpha_optimizer.zero_grad()
        # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.

        # print(self.target_entropy)
        # print(log_prob_v, "!!!!!!!!!!!1")

        entropy_loss = (self.log_alpha * (-self.target_entropy - log_prob_v).detach()).mean()
        entropy_loss.backward()
        nn_utils.clip_grad_norm_([self.log_alpha], self.params.CLIP_GRAD)
        self.alpha_optimizer.step()

        self.alpha = torch.max(torch.exp(self.log_alpha), torch.exp(self.min_alpha))
        #print(self.alpha, "!!!!!!!!!")

    def adjust_alpha_for_discrete_action(self, probs, log_prob_v):
        self.alpha_optimizer.zero_grad()
        # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.

        entropy_loss = -1.0 * (probs * self.log_alpha * (self.target_entropy - log_prob_v)).mean().detach()
        entropy_loss.backward()
        nn_utils.clip_grad_norm_([self.log_alpha], self.params.CLIP_GRAD)
        self.alpha_optimizer.step()
        self.alpha = torch.exp(self.log_alpha)

    def unpack_batch_for_sac(
            self, batch, target_model=None, sac_base_model=None, alpha=None, params=None
    ):
        states, actions, rewards, not_done_idx, last_states, last_steps = [], [], [], [], [], []

        for idx, exp in enumerate(batch):
            states.append(np.array(exp.state, copy=False))

            if hasattr(exp, "is_reset"):
                if exp.is_reset:
                    self.initial_obs_deque.append(exp.state)

            actions.append(exp.action)
            rewards.append(exp.reward)

            if exp.last_state is not None:
                not_done_idx.append(idx)
                last_states.append(np.array(exp.last_state, copy=False))
                last_steps.append(exp.last_step)

        states_v = float32_preprocessor(states).to(self.device)
        actions_v = self.convert_action_to_torch_tensor(actions, self.device)

        # handle rewards
        target_action_values_np = np.array(rewards, dtype=np.float32)

        if not_done_idx:
            last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(self.device)

            if self.params.RL_ALGORITHM in [RLAlgorithmName.CONTINUOUS_SAC_V0]:
                last_steps_v = np.asarray(last_steps)
                last_mu_v, last_logstd_v, _ = sac_base_model.forward_actor(last_states_v)
                dist = Normal(loc=last_mu_v, scale=torch.exp(last_logstd_v))

                last_actions_v = dist.sample()
                last_log_prob_v = dist.log_prob(last_actions_v).sum(dim=-1, keepdim=True)

                last_q_1_v, last_q_2_v = target_model.base.twinq(last_states_v, last_actions_v)
                last_q_np = torch.min(last_q_1_v, last_q_2_v).detach().cpu().numpy()[:, 0] * (params.GAMMA ** last_steps_v)
                last_log_prob_v = alpha * last_log_prob_v

                # last_q_np.shape: (128,)
                # entropy_v.squeeze(-1).detach().numpy().shape: (128,)
                last_q_np -= last_log_prob_v.squeeze(-1).detach().cpu().numpy()

            else:
                # probs.shape: torch.Size([32, 2])
                probs, _ = sac_base_model.forward_actor(last_states_v)
                z = (probs == 0.0).float() * 1e-8
                last_log_prob_v = torch.log(probs + z)

                last_q_1_v, last_q_2_v = target_model.base.twinq(last_states_v)

                # torch.min(last_q_1_v, last_q_2_v).shape: (32, 2)
                # torch.unsqueeze(params.GAMMA ** torch.as_tensor(last_steps), dim=1).shape: (32, 1)
                last_q_np = torch.min(last_q_1_v, last_q_2_v) * torch.unsqueeze(params.GAMMA ** torch.as_tensor(last_steps), dim=1)
                last_log_prob_v = alpha * last_log_prob_v

                last_q_np = probs * (last_q_np - last_log_prob_v)
                last_q_np = last_q_np.sum(dim=-1).detach().numpy()

            target_action_values_np[not_done_idx] += last_q_np

        target_action_values_v = float32_preprocessor(target_action_values_np).to(self.device)

        # states_v.shape: [128, 3]
        # actions_v.shape: [128, 1]
        # target_action_values_v.shape: [128]

        return states_v, actions_v, target_action_values_v
