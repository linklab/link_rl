import torch
import torch.nn.utils as nn_utils
import numpy as np

from codes.c_models.base_model import RNNModel
from codes.d_agents.on_policy.on_policy_agent import OnPolicyAgent
from codes.e_utils import replay_buffer
from codes.e_utils.common_utils import float32_preprocessor


class AgentA2C(OnPolicyAgent):
    """
    """
    def __init__(self, worker_id, action_shape, params, device):
        super(AgentA2C, self).__init__(worker_id, action_shape, params, device)
        self.train_action_selector = None
        self.test_and_play_action_selector = None
        self.model = None
        self.optimizer = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.buffer = replay_buffer.ExperienceReplayBuffer(
            experience_source=None, buffer_size=self.params.BATCH_SIZE
        )

    def __call__(self, state, critics=None):
        raise NotImplementedError

    # Lucky Episode에서 얻어낸 batch를 통해 학습할 때와, Unlucky Episode에서 얻어낸 batch를 통해 학습할 때마다 NN의 파라미터들이
    # 서로 다른 방향으로 반복적으로 휩쓸려가듯이 학습이 됨 --> Gradients의 Variance가 매우 큼
    def on_train(self, step_idx, expected_model_version):
        raise NotImplementedError

    def unpack_batch_for_actor_critic(
            self, batch, target_model=None, sac_base_model=None, alpha=None, params=None
    ):
        """
        Convert batch into training tensors
        :param batch:
        :param model:
        :return: state variable, actions tensor, target values variable
        """
        states, actions, rewards, not_done_idx, last_states, last_steps = [], [], [], [], [], []

        if isinstance(self.model, RNNModel):
            actor_hidden_states = []
            critic_hidden_states = []
            critic_1_hidden_states = None
            critic_2_hidden_states = None
        else:
            actor_hidden_states = critic_hidden_states = critic_1_hidden_states = critic_2_hidden_states = None

        for idx, exp in enumerate(batch):
            states.append(np.array(exp.state, copy=False))
            actions.append(exp.action)
            rewards.append(exp.reward)

            if exp.last_state is not None:
                not_done_idx.append(idx)
                last_states.append(np.array(exp.last_state, copy=False))
                last_steps.append(exp.last_step)

            if isinstance(self.model, RNNModel):
                actor_hidden_states.append(exp.agent_state.actor_hidden_state)
                critic_hidden_states.append(exp.agent_state.critic_hidden_state)

        states_v = float32_preprocessor(states).to(self.device)
        actions_v = self.convert_action_to_torch_tensor(actions, self.device)

        if isinstance(self.model, RNNModel):
            actor_hidden_states_v = float32_preprocessor(actor_hidden_states).to(self.device)
            critic_hidden_states_v = float32_preprocessor(critic_hidden_states).to(self.device)
            critic_1_hidden_states_v = None
            critic_2_hidden_states_v = None
        else:
            actor_hidden_states_v = critic_hidden_states_v = critic_1_hidden_states_v = critic_2_hidden_states_v = None

        # handle rewards
        target_action_values_np = np.array(rewards, dtype=np.float32)

        if not_done_idx:
            last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(self.device)
            last_steps_v = np.asarray(last_steps)
            last_values_v, _ = target_model.forward_critic(last_states_v, critic_hidden_states_v)
            last_values_np = last_values_v.detach().numpy()[:, 0] * (params.GAMMA ** last_steps_v)
            target_action_values_np[not_done_idx] += last_values_np

        target_action_values_v = float32_preprocessor(target_action_values_np).to(self.device)

        # states_v.shape: [128, 3]
        # actions_v.shape: [128, 1]
        # target_action_values_v.shape: [128]

        if isinstance(self.model, RNNModel):
            return states_v, actions_v, target_action_values_v, actor_hidden_states_v, critic_hidden_states_v
        else:
            return states_v, actions_v, target_action_values_v

    # def backward_and_step(self, loss_critic_v, loss_entropy_v, loss_actor_v):
    #     self.optimizer.zero_grad()
    #     loss_actor_v.backward(retain_graph=True)
    #     (loss_critic_v + self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v).backward()
    #     nn_utils.clip_grad_norm_(self.model.base.parameters(), self.params.CLIP_GRAD)
    #     self.optimizer.step()
    #
    #     gradients = self.model.get_gradients_for_current_parameters()
    #
    #     # try:
    #     #     self.model.check_gradient_nan_or_zero(gradients)
    #     # except ValueError as e:
    #     #     print(loss_critic_v, loss_entropy_v, loss_actor_v)
    #     #     exit(-1)
    #
    #     self.buffer.clear()
    #
    #     return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0

    # def backward_and_step(self, loss_critic_v, loss_entropy_v, loss_actor_v):
    #     self.optimizer.zero_grad()
    #     loss_actor_v.backward(retain_graph=True)
    #     (loss_critic_v + self.params.ENTROPY_LOSS_WEIGHT * loss_entropy_v).backward()
    #     nn_utils.clip_grad_norm_(self.model.base.parameters(), self.params.CLIP_GRAD)
    #     self.optimizer.step()
    #
    #     gradients = self.model.get_gradients_for_current_parameters()
    #
    #     # try:
    #     #     self.model.check_gradient_nan_or_zero(gradients)
    #     # except ValueError as e:
    #     #     print(loss_critic_v, loss_entropy_v, loss_actor_v)
    #     #     exit(-1)
    #
    #     self.buffer.clear()
    #
    #     return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0
