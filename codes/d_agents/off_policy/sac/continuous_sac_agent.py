# https://spinningup.openai.com/en/latest/algorithms/sac.html
# https://github.com/pranz24/pytorch-soft-actor-critic
#https://github.com/ku2482/soft-actor-critic.pytorch/blob/master/code/agent.py
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.distributions import Normal

from codes.a_config._rl_parameters.off_policy.parameter_sac import SACActionSelectorType
from codes.a_config._rl_parameters.off_policy.parameter_td3 import TD3ActionSelectorType
from codes.c_models.continuous_action.soft_actor_critic_model import SoftActorCriticModel
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
    def __init__(self, worker_id, input_shape, action_shape, num_outputs, action_min, action_max, params, device):
        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.SOFT_ACTOR_CRITIC_MLP

        super(AgentSAC, self).__init__(
            worker_id=worker_id, params=params, action_shape=action_shape,
            action_min=action_min, action_max=action_max, device=device
        )

        self.__name__ = "AgentSAC"

        if params.TYPE_OF_SAC_ACTION_SELECTOR == SACActionSelectorType.BASIC_ACTION_SELECTOR:
            self.train_action_selector = ContinuousNormalSACActionSelector(params=params)
        elif params.TYPE_OF_SAC_ACTION_SELECTOR == SACActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR:
            self.train_action_selector = SomeTimesBlowSACActionSelector(
                min_blowing_action=-5.0, max_blowing_action=5.0, params=self.params,
            )
        else:
            raise ValueError()

        self.test_and_play_action_selector = ContinuousNormalSACActionSelector(params=params)

        self.model = SoftActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.target_model = SoftActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        # grad_false(self.target_model)

        self.test_model = SoftActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.actor_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.actor_params,
            learning_rate=self.params.ACTOR_LEARNING_RATE,
            params=params
        )

        self.twinq_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.twinq_params,
            learning_rate=self.params.LEARNING_RATE,
            params=params
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
        return self.continuous_sac_call(state, agent_states)

    def continuous_sac_call(self, state, agent_states=None):
        state = self.preprocess(state)

        if len(state) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.agent_mode == AgentMode.TRAIN:
            with torch.no_grad():
                mu_v, logstd_v = self.model.base.actor(state)
                actions = self.train_action_selector(mu_v=mu_v, logstd_v=logstd_v)
        else:
            with torch.no_grad():
                mu_v, _ = self.test_model.base.actor(state)
                actions = self.test_and_play_action_selector(mu_v=mu_v, logstd_v=None)

        return actions, agent_states

    def on_train(self, step_idx):
        if self.params.PER:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(
            batch, self.target_model, self.params, alpha=self.alpha
        )

        # train twinq
        self.twinq_optimizer.zero_grad()
        q1_v, q2_v = self.model.base.twinq(states_v, actions_v)

        q_loss_v = F.mse_loss(q1_v.squeeze(dim=-1), target_action_values_v.detach(), reduction="none") + \
                   F.mse_loss(q2_v.squeeze(dim=-1), target_action_values_v.detach(), reduction="none")
        q_loss_v = q_loss_v.mean()

        q_loss_v.backward(retain_graph=True)
        nn_utils.clip_grad_norm_(self.model.base.twinq_params, self.params.CLIP_GRAD)
        self.twinq_optimizer.step()

        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            self.buffer.update_priorities(batch_indices, q_loss_v.abs().detach().cpu().numpy())
            self.buffer.update_beta(step_idx)

        # train actor
        self.actor_optimizer.zero_grad()

        mu_v, logstd_v, _ = self.model.base.forward_actor(states_v)
        re_parameterized_action_v, entropy_v = self.train_action_selector.select_reparameterization_trick_action(
            mu_v=mu_v, logstd_v=logstd_v
        )

        # states_v.shape: torch.Size([128, 3])
        # re_parameterized_action_v.shape: torch.Size([128, 1])
        # entropies_v.shape: torch.Size([128, 1])

        q1_v, q2_v = self.model.base.twinq(states_v, re_parameterized_action_v)

        # q1_v.shape: torch.Size([128, 1])
        # q2_v.shape: torch.Size([128, 1])
        # torch.min(q1_v, q2_v).shape: torch.Size([128, 1])
        # entropies_v.shape: torch.Size([128, 1])
        entropy_v = self.alpha * entropy_v
        objectives_v = torch.min(q1_v, q2_v) + entropy_v

        loss_actor_v = -1.0 * objectives_v.mean()
        loss_actor_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.actor_params, self.params.CLIP_GRAD)
        self.actor_optimizer.step()

        self.target_model.twinq_alpha_sync(self.model, alpha=1 - self.params.TAU)

        if self.params.ENTROPY_TUNING:
            self.alpha_optimizer.zero_grad()
            # Intuitively, we increase alpha when entropy is less than target entropy, vice versa.
            entropy_loss = -1.0 * self.log_alpha * (self.target_entropy - sampled_entropies_v).mean().detach()
            entropy_loss.backward()
            nn_utils.clip_grad_norm_([self.log_alpha], self.params.CLIP_GRAD)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # gradients = self.model.get_gradients_for_current_parameters()
        gradients = None

        return gradients, q_loss_v.item(), loss_actor_v.item() * -1.0
