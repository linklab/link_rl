# https://spinningup.openai.com/en/latest/algorithms/sac.html
# https://github.com/pranz24/pytorch-soft-actor-critic
#https://github.com/ku2482/soft-actor-critic.pytorch/blob/master/code/agent.py
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from codes.c_models.discrete_action.discrete_sac_model import DiscreteSACModel
from codes.d_agents.off_policy.sac.sac_agent import AgentSAC
from codes.d_agents.on_policy.stochastic_policy_action_selector import DiscreteCategoricalActionSelector
from codes.e_utils import rl_utils
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentDiscreteSAC(AgentSAC):
    """
    """
    def __init__(self, worker_id, observation_shape, action_shape, action_n, params, device):
        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.DISCRETE_SAC_MLP

        super(AgentDiscreteSAC, self).__init__(worker_id=worker_id, action_shape=action_shape, params=params, device=device)

        self.__name__ = "AgentDiscreteSAC"

        self.train_action_selector = DiscreteCategoricalActionSelector(params=params)
        self.test_and_play_action_selector = DiscreteCategoricalActionSelector(params=params)

        self.model = DiscreteSACModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            action_n=action_n,
            params=params,
            device=device
        ).to(device)

        self.target_model = DiscreteSACModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            action_n=action_n,
            params=params,
            device=device
        ).to(device)

        # grad_false(self.target_model)

        self.test_model = DiscreteSACModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            action_n=action_n,
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

        self.cache_loss_actor_v = torch.tensor(0.0)


    def __call__(self, state, agent_states=None):
        return self.discrete_sac_call(state, agent_states)

    def discrete_sac_call(self, state, agent_states=None):
        state = self.preprocess(state)

        if len(state) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.agent_mode == AgentMode.TRAIN:
            with torch.no_grad():
                probs, _ = self.model.base.forward_actor(state)
                actions = self.train_action_selector(probs=probs, agent_mode=AgentMode.TRAIN)
        else:
            with torch.no_grad():
                probs, _ = self.test_model.base.forward_actor(state)
                actions = self.test_and_play_action_selector(probs=probs, agent_mode=AgentMode.TEST)

        return actions, agent_states

    def on_train(self, step_idx):
        if self.params.ENTROPY_TUNING and self.target_entropy is None:
            self.reset_alpha()

        if self.params.PER:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        states_v, actions_v, target_action_values_v = self.unpack_batch_for_actor_critic(
            batch=batch, target_model=self.target_model, sac_base_model=self.model.base,
            alpha=self.alpha, params=self.params
        )

        # train twinq
        self.twinq_optimizer.zero_grad()
        q1_v, q2_v = self.model.base.twinq(states_v)

        curr_q1 = q1_v.gather(-1, index=actions_v.unsqueeze(1))
        curr_q2 = q2_v.gather(-1, index=actions_v.unsqueeze(1))

        # target_action_values_v.shape: torch.Size([32])
        q_loss_v = F.mse_loss(curr_q1.squeeze(-1), target_action_values_v.detach(), reduction="none") + \
                   F.mse_loss(curr_q2.squeeze(-1), target_action_values_v.detach(), reduction="none")
        q_loss_v = q_loss_v.mean()

        q_loss_v.backward(retain_graph=True)
        nn_utils.clip_grad_norm_(self.model.base.twinq_params, self.params.CLIP_GRAD)
        self.twinq_optimizer.step()

        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            self.buffer.update_priorities(batch_indices, q_loss_v.abs().detach().cpu().numpy())
            self.buffer.update_beta(step_idx)

        # train actor
        probs, action_v, log_prob_v = self.model.sample(states_v)
        # Delayed policy updates
        if step_idx % self.params.POLICY_UPDATE_FREQUENCY == 0:
            self.actor_optimizer.zero_grad()

            # states_v.shape: torch.Size([128, 3])
            # re_parameterization_trick_action_v.shape: torch.Size([128, 1])

            q1_v, q2_v = self.model.base.twinq(states_v)

            # q1_v.shape: torch.Size([128, 1])
            # q2_v.shape: torch.Size([128, 1])
            # torch.min(q1_v, q2_v).shape: torch.Size([128, 1])
            # log_prob_v.shape: torch.Size([128, 1])

            objectives_v = probs * (torch.min(q1_v, q2_v) - self.alpha * log_prob_v)
            loss_actor_v = -1.0 * objectives_v.mean()
            self.cache_loss_actor_v = loss_actor_v

            loss_actor_v.backward()
            nn_utils.clip_grad_norm_(self.model.base.actor_params, self.params.CLIP_GRAD)
            self.actor_optimizer.step()

            self.target_model.twinq_alpha_sync(self.model, alpha=1 - self.params.TAU)
        else:
            loss_actor_v = self.cache_loss_actor_v

        if self.params.ENTROPY_TUNING:
            self.adjust_alpha(log_prob_v)

        # gradients = self.model.get_gradients_for_current_parameters()
        gradients = None

        return gradients, q_loss_v.item(), loss_actor_v.item() * -1.0
