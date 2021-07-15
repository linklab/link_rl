import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from codes.a_config._rl_parameters.off_policy.parameter_ddpg import PARAMETERS_DDPG, DDPGActionSelectorType, \
    DDPGActionType
from codes.c_models.continuous_action.continuous_deterministic_actor_critic_model import DeterministicContinuousActorCriticModel
from codes.d_agents.off_policy.ddpg.ddpg_action_selector import DDPGActionSelector, SomeTimesBlowDDPGActionSelector
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.e_utils import rl_utils
from codes.d_agents.actions import EpsilonTracker
from codes.e_utils.common_utils import float32_preprocessor
from codes.e_utils.names import DeepLearningModelName, AgentMode


class AgentDDPG(OffPolicyAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, worker_id, observation_shape, action_shape, num_outputs, action_min, action_max, params, device):
        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.CONTINUOUS_DETERMINISTIC_ACTOR_CRITIC_MLP
        assert issubclass(params, PARAMETERS_DDPG)

        super(AgentDDPG, self).__init__(
            worker_id=worker_id, action_shape=action_shape, params=params, device=device
        )
        self.__name__ = "AgentDDPG"

        self.num_outputs = num_outputs
        self.action_min = action_min
        self.action_max = action_max

        # if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0]:
        #     self.train_action_selector = SomeTimesBlowDDPGActionSelector(
        #         noise_enabled=params.NOISE_ENABLED, ou_mu=np.zeros(self.action_shape), ou_sigma=self.params.OU_SIGMA
        #         min_blowing_action=-10.0 * params.ACTION_SCALE, max_blowing_action=10.0 * params.ACTION_SCALE,
        #     )
        #     self.test_and_play_action_selector = SomeTimesBlowDDPGActionSelector(
        #         noise_enabled=False,
        #         min_blowing_action=-10.0 * params.ACTION_SCALE, max_blowing_action=10.0 * params.ACTION_SCALE
        #     )
        # else:
        #     self.train_action_selector = DDPGActionSelector(noise_enabled=params.NOISE_ENABLED, ou_sigma=self.params.OU_SIGMA)
        #     self.test_and_play_action_selector = DDPGActionSelector(noise_enabled=False)

        if params.TYPE_OF_DDPG_ACTION_SELECTOR == DDPGActionSelectorType.BASIC_ACTION_SELECTOR:
            self.train_action_selector = DDPGActionSelector(
                noise_enabled=params.NOISE_ENABLED, ou_mu=np.zeros(self.action_shape), ou_sigma=self.params.OU_SIGMA,
                epsilon=params.EPSILON_INIT, params=params
            )
        elif params.TYPE_OF_DDPG_ACTION_SELECTOR == DDPGActionSelectorType.SOMETIMES_BLOW_ACTION_SELECTOR:
            self.train_action_selector = SomeTimesBlowDDPGActionSelector(
                noise_enabled=params.NOISE_ENABLED, ou_mu=np.zeros(self.action_shape), ou_sigma=self.params.OU_SIGMA,
                min_blowing_action=-5.0, max_blowing_action=5.0,
                epsilon=params.EPSILON_INIT, params=params
            )
        elif params.TYPE_OF_DDPG_ACTION_SELECTOR == DDPGActionSelectorType.NOISY_NET_ACTION_SELECTOR:
            self.train_action_selector = DDPGActionSelector(noise_enabled=False, params=params)
        else:
            raise ValueError()

        self.test_and_play_action_selector = DDPGActionSelector(noise_enabled=False, params=params)

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

        self.test_model = DeterministicContinuousActorCriticModel(
            worker_id=worker_id,
            observation_shape=observation_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

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

        self.last_noise = 0.0
        self.global_uncertainty = 1.0

        if self.params.TYPE_OF_DDPG_ACTION in [
            DDPGActionType.OU_NOISE_WITH_EPSILON,
            DDPGActionType.GAUSSIAN_NOISE_WITH_EPSILON
        ]:
            self.epsilon_tracker = EpsilonTracker(
                action_selector=self.train_action_selector,
                eps_start=params.EPSILON_INIT,
                eps_final=params.EPSILON_MIN,
                eps_last_frames=params.EPSILON_MIN_STEP
            )
        else:
            self.epsilon_tracker = None

        self.num_trains = 0

    def __call__(self, state, agent_state=None):
        noises = [None] * len(state)

        if not isinstance(state, torch.FloatTensor):
            state = float32_preprocessor(state).to(self.device)

        if len(state) == 1:
            self.model.eval()
        else:
            self.model.train()

        if self.agent_mode == AgentMode.TRAIN:
            mu_v = self.model(state)
            mu = mu_v.detach().cpu().numpy()
            actions, new_noises = self.train_action_selector(mu, noises, self.global_uncertainty)
            self.last_noise = new_noises[0][0]
        else:
            mu_v = self.test_model(state)
            mu = mu_v.detach().cpu().numpy()
            actions, new_noises = self.test_and_play_action_selector(mu, noises)

        # print("actions: {0:7.4f}, noises: {1:7.4f}".format(
        #     actions[0][0], self.last_noise
        # ))

        return actions, new_noises

    def on_train(self, step_idx):
        self.num_trains += 1
        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        states_v, actions_v, rewards_v, dones_mask, last_states_v, agent_states = self.unpack_batch(batch)
        self.actor_optimizer.zero_grad()

        #######################
        # train actor - start #
        #######################
        current_actions_v = self.model.base.forward_actor(states_v)
        q_v_for_actor = self.model.base.forward_critic(states_v, current_actions_v)
        loss_actor_v = -1.0 * q_v_for_actor.mean()

        loss_actor_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.actor_params, self.params.CLIP_GRAD)
        self.actor_optimizer.step()
        #####################
        # train actor - end #
        #####################

        ########################
        # train critic - start #
        ########################
        self.critic_optimizer.zero_grad()
        # critic_parameters = self.model.base.critic.parameters()
        # for p in critic_parameters:
        #     p.requires_grad = True

        q_v = self.model.base.forward_critic(states_v, actions_v)
        last_act_v = self.target_model.forward_actor(last_states_v)
        q_last_v = self.target_model.forward_critic(last_states_v, last_act_v)
        q_last_v[dones_mask] = 0.0
        target_q_v = rewards_v.unsqueeze(dim=-1) + q_last_v * self.params.GAMMA ** self.params.N_STEP

        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            batch_l1_loss = F.smooth_l1_loss(q_v, target_q_v.detach(), reduction='none')
            batch_mse1_loss = F.mse_loss(q_v, target_q_v.detach(), reduction='none')
            batch_weights_v = torch.tensor(batch_weights)
            # critic_loss_v = batch_weights_v * batch_l1_loss
            critic_loss_v = batch_weights_v * batch_mse1_loss

            self.buffer.update_priorities(batch_indices, batch_l1_loss.detach().cpu().numpy() + 1e-5)
            self.buffer.update_beta(step_idx)
        else:
            critic_loss_v = F.mse_loss(q_v, target_q_v.detach(), reduction='none')

        loss_critic_v = critic_loss_v.mean()

        loss_critic_v.backward()
        nn_utils.clip_grad_norm_(self.model.base.critic_params, self.params.CLIP_GRAD)
        self.critic_optimizer.step()
        ######################
        # train critic - end #
        ######################

        self.target_model.soft_update(self.model, tau=self.params.TAU) #(1 - 0.001)

        # gradients = self.model.get_gradients_for_current_parameters()
        # self.model.check_gradient_nan_or_zero(gradients)

        gradients = None

        if self.params.TYPE_OF_DDPG_ACTION_SELECTOR == DDPGActionSelectorType.NOISY_NET_ACTION_SELECTOR:
            self.model.base.reset_noise()  # Pick a new noise vector (until next optimisation step)
            self.target_model.base.reset_noise()

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0
