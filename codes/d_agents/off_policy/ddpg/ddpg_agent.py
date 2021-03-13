import numpy as np
import torch
import torch.nn.functional as F
from icecream import ic

from codes.c_models.continuous_action.deterministic_continuous_actor_critic_model import DeterministicContinuousActorCriticModel
from codes.d_agents.a0_base_agent import TargetNet, float32_preprocessor
from codes.d_agents.off_policy.off_policy_agent import OffPolicyAgent
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.actions import EpsilonGreedySomeTimesBlowDDPGActionSelector, EpsilonGreedyDDPGActionSelector, \
    EpsilonTracker
from codes.e_utils.names import DeepLearningModelName, AgentMode, EnvironmentName


class AgentDDPG(OffPolicyAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process
    """
    def __init__(self, worker_id, input_shape, num_outputs, action_min, action_max, params, device):
        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.DETERMINISTIC_CONTINUOUS_ACTOR_CRITIC_MLP

        super(AgentDDPG, self).__init__(worker_id=worker_id, params=params, device=device)

        self.__name__ = "AgentDDPG"
        self.action_min = action_min
        self.action_max = action_max

        if params.ENVIRONMENT_ID in [EnvironmentName.PENDULUM_MATLAB_V0, EnvironmentName.PENDULUM_MATLAB_DOUBLE_RIP_V0]:
            self.train_action_selector = EpsilonGreedySomeTimesBlowDDPGActionSelector(
                epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=params.ACTION_SCALE,
                min_blowing_action=-10.0 * params.ACTION_SCALE, max_blowing_action=10.0 * params.ACTION_SCALE
            )
            self.test_and_play_action_selector = EpsilonGreedySomeTimesBlowDDPGActionSelector(
                epsilon=0.0, ou_enabled=False, scale_factor=params.ACTION_SCALE,
                min_blowing_action=-10.0 * params.ACTION_SCALE, max_blowing_action=10.0 * params.ACTION_SCALE
            )
        else:
            self.train_action_selector = EpsilonGreedyDDPGActionSelector(
                epsilon=params.EPSILON_INIT, ou_enabled=True, scale_factor=params.ACTION_SCALE
            )
            self.test_and_play_action_selector = EpsilonGreedyDDPGActionSelector(
                epsilon=0.0, ou_enabled=False, scale_factor=params.ACTION_SCALE
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
            parameters=self.model.base.critic.parameters(),
            learning_rate=self.params.LEARNING_RATE,
            params=params
        )

        self.last_noise = 0.0

    def __call__(self, states, noises=None):
        if not noises:
            noises = [None] * len(states)

        if not isinstance(states, torch.FloatTensor):
            states = float32_preprocessor(states).to(self.device)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        mu_v = self.model(states)
        mu = mu_v.detach().cpu().numpy()

        if self.agent_mode == AgentMode.TRAIN:
            actions, new_noises = self.train_action_selector(mu, noises)
        else:
            actions, new_noises = self.test_and_play_action_selector(mu, noises)

        self.last_noise = new_noises[0][0]
        #print(actions, self.action_min, self.action_max, "!!!!!!!!!!!!!!!!")
        #actions = np.clip(actions, self.action_min, self.action_max)
        #####################################

        # print("actions: {0:7.4f}, noises: {1:7.4f}".format(
        #     actions[0][0], self.last_noise
        # ))

        return actions, new_noises

    def train(self, step_idx):
        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            batch, batch_indices, batch_weights = self.buffer.sample(self.params.BATCH_SIZE)
        else:
            batch = self.buffer.sample(self.params.BATCH_SIZE)
            batch_indices, batch_weights = None, None

        # print(batch)
        states_v, actions_v, rewards_v, dones_mask, last_states_v = self.unpack_batch(batch)
        self.actor_optimizer.zero_grad()

        current_actions_v = self.model.base.forward_actor(states_v)
        q_v_for_actor = self.model.base.forward_critic(states_v, current_actions_v)
        loss_actor_v = -1.0 * q_v_for_actor.mean()

        loss_actor_v.backward()
        self.actor_optimizer.step()

        # train critic
        self.critic_optimizer.zero_grad()
        # critic_parameters = self.model.base.critic.parameters()
        # for p in critic_parameters:
        #     p.requires_grad = True

        q_v = self.model.base.forward_critic(states_v, actions_v)
        last_act_v = self.target_agent.target_model.forward_actor(last_states_v)
        q_last_v = self.target_agent.target_model.forward_critic(last_states_v, last_act_v)
        q_last_v[dones_mask] = 0.0
        target_q_v = rewards_v.unsqueeze(dim=-1) + q_last_v * self.params.GAMMA ** self.params.N_STEP

        if self.params.PER_PROPORTIONAL or self.params.PER_RANK_BASED:
            batch_l1_loss = F.smooth_l1_loss(q_v, target_q_v.detach(), reduction='none')  # for PER
            batch_mse1_loss = F.mse_loss(q_v, target_q_v.detach(), reduction='none')  # for PER
            batch_weights_v = torch.tensor(batch_weights)
            # critic_loss_v = batch_weights_v * batch_l1_loss
            critic_loss_v = batch_weights_v * batch_mse1_loss

            self.buffer.update_priorities(batch_indices, batch_l1_loss.detach().cpu().numpy() + 1e-5)
            self.buffer.update_beta(step_idx)
        else:
            # critic_loss_v = F.smooth_l1_loss(q_v, target_q_v.detach(), reduction='none')
            critic_loss_v = F.mse_loss(q_v, target_q_v.detach(), reduction='none')

        loss_critic_v = critic_loss_v.mean()

        loss_critic_v.backward()
        self.critic_optimizer.step()

        # train actor
        # self.actor_optimizer.zero_grad()
        # critic_parameters = self.model.base.critic.parameters()
        # for p in critic_parameters:
        #     p.requires_grad = False

        # self.base_optimizer.zero_grad()
        # loss_actor_v.backward(retain_graph=True)
        # loss_critic_v.backward()
        # self.base_optimizer.step()


        self.target_agent.alpha_sync(alpha=1 - self.params.TAU) #(1 - 0.001)

        gradients = self.model.get_gradients_for_current_parameters()

        self.model.check_gradient_nan(gradients)

        return gradients, loss_critic_v.item(), loss_actor_v.item() * -1.0
