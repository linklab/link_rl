import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from codes.a_config._rl_parameters.off_policy.parameter_td3 import TD3ActionType, TD3ActionSelectorType
from codes.c_models.continuous_action.continuous_deterministic_actor_critic_model import \
    DeterministicContinuousActorCriticModel
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.d_agents.off_policy.td3.td3_action_selector import SomeTimesBlowTD3ActionSelector, TD3ActionSelector
from codes.e_utils import rl_utils
from codes.d_agents.actions import EpsilonTracker
from codes.e_utils.common_utils import float32_preprocessor
from codes.e_utils.names import DeepLearningModelName, AgentMode
from torch.distributions import normal

# https://github.com/sfujim/TD3
# https://spinningup.openai.com/en/latest/algorithms/td3.html

class AgentTD3(OffPolicyAgent):
    def __init__(self, worker_id, observation_shape, action_shape, num_outputs, action_min, action_max, params, device):
        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.TD3_MLP

        super(AgentTD3, self).__init__(worker_id=worker_id, action_shape=action_shape, params=params, device=device)
        self.__name__ = "AgentTD3"
        self.num_outputs = num_outputs
        self.action_min = action_min
        self.action_max = action_max

        if params.TYPE_OF_TD3_ACTION_SELECTOR == TD3ActionSelectorType.BASIC_ACTION_SELECTOR:
            self.train_action_selector = TD3ActionSelector(
                epsilon=params.EPSILON_INIT, noise_std=params.TRAIN_ACTION_NOISE_STD, params=self.params
            )
        elif params.TYPE_OF_TD3_ACTION_SELECTOR == TD3ActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR:
            self.train_action_selector = SomeTimesBlowTD3ActionSelector(
                epsilon=params.EPSILON_INIT, noise_std=params.TRAIN_ACTION_NOISE_STD,
                min_blowing_action=-5.0, max_blowing_action=5.0,
                params=self.params
            )
        elif params.TYPE_OF_TD3_ACTION_SELECTOR == TD3ActionSelectorType.NOISY_NET_ACTION_SELECTOR:
            self.train_action_selector = TD3ActionSelector(
                epsilon=0.0, noise_std=0.0, params=self.params
            )
        else:
            raise ValueError()

        self.test_and_play_action_selector = TD3ActionSelector(epsilon=0.0, noise_std=0.0, params=self.params)

        self.model = DeterministicContinuousActorCriticModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        self.target_model = DeterministicContinuousActorCriticModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)
        self.target_model.sync(self.model)

        self.test_model = DeterministicContinuousActorCriticModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)
        self.test_model.sync(self.model)


        # self.base_optimizer = rl_utils.get_optimizer(
        #     parameters=self.model.base.parameters(),
        #     learning_rate=self.params.LEARNING_RATE,
        #     params=params
        # )

        self.actor_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.actor_params,
            learning_rate=self.params.ACTOR_LEARNING_RATE,
            params=params
        )

        self.critic_optimizer = rl_utils.get_optimizer(
            parameters=self.model.base.critic_params,
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

        if self.params.TYPE_OF_TD3_ACTION == TD3ActionType.GAUSSIAN_NOISE_WITH_EPSILON:
            self.epsilon_tracker = EpsilonTracker(
                action_selector=self.train_action_selector,
                eps_start=params.EPSILON_INIT,
                eps_final=params.EPSILON_MIN,
                eps_last_frames=params.EPSILON_MIN_STEP
            )
        else:
            self.epsilon_tracker = None

        self.cache_loss_actor_v = torch.tensor(0.0)
        self.last_noise = 0.0

    def __call__(self, state, agent_state):
        if not isinstance(state, torch.FloatTensor):
            state = float32_preprocessor(state).to(self.device)

        if len(state) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.agent_mode == AgentMode.TRAIN:
            mu_v, agent_state = self.model(state, agent_state)
            mu = mu_v.detach().cpu().numpy()
            actions, new_noises = self.train_action_selector(mu)
        else:
            mu_v, agent_state = self.test_model(state, agent_state)
            mu = mu_v.detach().cpu().numpy()
            actions, new_noises = self.test_and_play_action_selector(mu)

        self.last_noise = new_noises[0][0]
        #####################################

        return actions, agent_state

    # @profile
    def on_train(self, step_idx):
        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        states_v, actions_v, rewards_v, dones_mask, last_states_v, agent_states = self.unpack_batch(batch)

        ########################
        # train critic - start #
        ########################
        self.critic_optimizer.zero_grad()

        # noise: [128, 1]
        m = normal.Normal(loc=0.0, scale=self.params.TRAIN_ACTION_NOISE_STD)
        clipped_noise = m.sample(actions_v.size()).clamp(
            -self.params.TRAIN_ACTION_NOISE_CLIP, self.params.TRAIN_ACTION_NOISE_CLIP
        ).to(self.device)

        # print(actions_v.size(), noise.size(), "!!!!")

        # last_actions_v: [128, 1]
        last_smoothed_actions_v = (
            self.target_model.base.forward_actor(last_states_v, agent_states)[0] + clipped_noise
        ).clamp(-1.0, 1.0)

        # target_q_v_1, target_q_v_2: [128, 1]
        target_q_v_1, target_q_v_2, _ = self.target_model.base.forward_critic(
            last_states_v, last_smoothed_actions_v, agent_states
        )

        # target_min_q_v_1, next_target_q_v, target_q_v: [128, 1]
        target_min_q_v = torch.min(target_q_v_1, target_q_v_2)
        next_target_q_v = (self.params.GAMMA ** self.params.N_STEP) * target_min_q_v
        next_target_q_v[dones_mask] = 0.0
        target_q_v = rewards_v.unsqueeze(dim=-1) + next_target_q_v

        # current_q_v_1, current_q_v_2: [128, 1]
        current_q_v_1, current_q_v_2, _ = self.model.base.forward_critic(states_v, actions_v, agent_states)

        # loss_critic_v: [128, 1]
        loss_critic_v = F.mse_loss(current_q_v_1, target_q_v.detach(), reduction='none') + \
                        F.mse_loss(current_q_v_2, target_q_v.detach(), reduction='none')

        loss_critic_v = loss_critic_v.mean()
        loss_critic_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.critic_params, self.params.CLIP_GRAD)
        self.critic_optimizer.step()
        ######################
        # train critic - end #
        ######################


        #######################
        # train actor - start #
        #######################
        # Delayed policy updates
        if step_idx % self.params.POLICY_UPDATE_FREQUENCY == 0:
            self.actor_optimizer.zero_grad()

            current_actions_v, _ = self.model.base.forward_actor(states_v, agent_states)
            q_v_for_actor, _ = self.model.base.forward_only_critic_1(states_v, current_actions_v, agent_states)
            loss_actor_v = -1.0 * q_v_for_actor.mean()
            self.cache_loss_actor_v = loss_actor_v

            loss_actor_v.backward()
            nn_utils.clip_grad_norm_(self.model.base.actor_params, self.params.CLIP_GRAD)
            self.actor_optimizer.step()
            #print(step_idx, "ACTOR")

            self.target_model.soft_update(self.model, tau=self.params.TAU)
        else:
            loss_actor_v = self.cache_loss_actor_v
        #####################
        # train actor - end #
        #####################

        # gradients = self.model.get_gradients_for_current_parameters()
        # self.model.check_gradient_nan_or_zero(gradients)

        gradients = None

        if self.params.TYPE_OF_TD3_ACTION_SELECTOR == TD3ActionSelectorType.NOISY_NET_ACTION_SELECTOR:
            self.model.base.reset_noise()  # Pick a new noise vector (until next optimisation step)
            self.target_model.base.reset_noise()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0
