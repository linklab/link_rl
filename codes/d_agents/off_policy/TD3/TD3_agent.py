import numpy as np
import torch
import torch.nn.functional as F
from icecream import ic

from codes.c_models.continuous_action.deterministic_continuous_actor_critic_model import \
    DeterministicContinuousActorCriticModel
from codes.d_agents.a0_base_agent import TargetNet, float32_preprocessor
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.actions import TD3ActionSelector, EpsilonTracker
from codes.e_utils.names import DeepLearningModelName, AgentMode, EnvironmentName
import copy


class AgentTD3(OffPolicyAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """

    def __init__(self, worker_id, input_shape, num_outputs, action_min, action_max, params, device):
        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.TD3_MLP

        super(AgentTD3, self).__init__(worker_id=worker_id, params=params, device=device)

        self.__name__ = "AgentTD3"
        self.action_min = action_min
        self.action_max = action_max
        self.policy_noise = 0.2
        self.noise_clip = 0.5

        self.train_action_selector = TD3ActionSelector(
            epsilon=params.EPSILON_INIT, scale_factor=params.ACTION_SCALE
        )
        self.test_and_play_action_selector = TD3ActionSelector(
            epsilon=0.0, scale_factor=params.ACTION_SCALE
        )

        self.epsilon_tracker = EpsilonTracker(
            action_selector=self.train_action_selector,
            eps_start=params.EPSILON_INIT,
            eps_final=params.EPSILON_MIN,
            eps_frames=params.EPSILON_MIN_STEP
        )

        self.model = DeterministicContinuousActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.target_agent = TargetNet(self.model.base)

        # self.base_optimizer = rl_utils.get_optimizer(
        #     parameters=self.model.base.parameters(),
        #     learning_rate=self.params.LEARNING_RATE,
        #     params=params
        # )

        self.actor_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.actor.parameters(),
            learning_rate=self.params.ACTOR_LEARNING_RATE,
            params=params
        )

        self.critic_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.critic_params,
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

    def __call__(self, states, agent_states=None):
        if not agent_states:
            agent_states = [None] * len(states)

        if not isinstance(states, torch.FloatTensor):
            states = float32_preprocessor(states).to(self.device)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        mu_v = self.model(states)
        mu = mu_v.detach().cpu().numpy()

        if self.agent_mode == AgentMode.TRAIN:
            actions, new_agent_states = self.train_action_selector(mu, agent_states)
        else:
            actions, new_agent_states = self.test_and_play_action_selector(mu, agent_states)

        actions = np.clip(actions, self.action_min, self.action_max)
        #####################################

        return actions, new_agent_states

    def train(self, step_idx):
        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        states_v, actions_v, rewards_v, dones_mask, last_states_v = self.unpack_batch_for_ddpg(batch)

        with torch.no_grad():
            noise = (torch.randn_like(actions_v) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            current_actions_v = (self.target_agent.target_model.forward_actor(states_v) + noise)

            target_q_v_1, target_q_v_2 = self.target_agent.target_model.forward_critic(states_v, current_actions_v)
            target_q_v = torch.min(target_q_v_1, target_q_v_2)
            next_target_q_v = self.params.GAMMA * target_q_v
            next_target_q_v[dones_mask] = 0.0
            target_q_v = rewards_v + next_target_q_v

            # for i in range(len(dones_mask)):
            #     if not dones_mask[i]:
            #         target_q_v = rewards_v + (self.params.GAMMA * target_q_v)

        current_q_v_1, current_q_v_2 = self.model.base.forward_critic(last_states_v, actions_v)

        critic_loss_v = F.mse_loss(current_q_v_1, target_q_v) + F.mse_loss(current_q_v_2, target_q_v)

        self.critic_optimizer.zero_grad()
        critic_loss_v.backward()
        self.critic_optimizer.step()

        actions_for_actor_v = self.model.base.forward_actor(last_states_v)
        critic_for_actor_v = self.model.base.forward_only_critic_1(last_states_v, actions_for_actor_v)
        actor_loss_v = -1.0 * critic_for_actor_v.mean()

        self.actor_optimizer.zero_grad()
        actor_loss_v.backward()
        self.actor_optimizer.step()

        self.target_agent.alpha_sync(alpha=1 - 0.0001)  # (1 - 0.001)

        gradients = self.model.get_gradients_for_current_parameters()

        # self.model.check_gradient_nan(gradients)

        return gradients, critic_loss_v.item(), actor_loss_v.item() * -1.0

    def unpack_batch_for_ddpg(self, batch):
        states, actions, rewards, dones, last_states = [], [], [], [], []

        for exp in batch:
            states.append(np.array(exp.state, copy=False))
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.last_state is None)
            if exp.last_state is None:
                last_states.append(exp.state)  # the result will be masked anyway
            else:
                last_states.append(np.array(exp.last_state, copy=False))

        states_v = float32_preprocessor(states).to(self.device)
        actions_v = float32_preprocessor(actions).to(self.device)
        rewards_v = float32_preprocessor(rewards).to(self.device)
        last_states_v = float32_preprocessor(last_states).to(self.device)
        dones_t = torch.BoolTensor(dones).to(self.device)

        return states_v, actions_v, rewards_v, dones_t, last_states_v