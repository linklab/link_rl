import math

import torch
import torch.nn.functional as F

from codes.d_agents.a0_base_agent import BaseAgent, float32_preprocessor
from codes.e_utils import rl_utils, replay_buffer
from codes.e_utils.names import DeepLearningModelName


class AgentContinuousPPO(BaseAgent):
    """
    """
    def __init__(
            self, worker_id, input_shape, num_outputs, action_selector, action_min, action_max, params,
            preprocessor=float32_preprocessor, device="cpu"
    ):
        super(AgentContinuousPPO, self).__init__()
        self.__name__ = "AgentContinuousPPO"
        self.device = device
        self.preprocessor = preprocessor
        self.action_selector = action_selector
        self.worker_id = worker_id
        self.params = params
        self.device = device
        self.action_min = action_min
        self.action_max = action_max

        assert params.DEEP_LEARNING_MODEL == DeepLearningModelName.STOCHASTIC_CONTINUOUS_ACTOR_CRITIC_MLP
        self.model = rl_utils.get_rl_model(
            worker_id=worker_id, input_shape=input_shape, num_outputs=num_outputs, params=params, device=self.device
        )

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

        self.trajectory = []

        self.buffer = replay_buffer.ExperienceReplayBuffer(experience_source=None, buffer_size=self.params.BATCH_SIZE)

    def __call__(self, states, critics=None):
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        if len(states) == 1:
            self.model.eval()
        else:
            self.model.train()

        mu_v, values_v = self.model(states)
        actions = self.action_selector(mu_v, self.model.base.actor.logstd, self.action_min, self.action_max)
        critics = [values_v.data.squeeze().cpu().numpy()]
        return actions, critics

    def train_net(self, trajectory):
        trajectory_states = [experience.state for experience in trajectory]
        trajectory_states_v = torch.FloatTensor(trajectory_states).to(self.device)

        trajectory_actions = [experience.action for experience in trajectory]
        trajectory_actions_v = torch.FloatTensor(trajectory_actions).to(self.device)

        trajectory_mu_v, _ = self.model(trajectory_states_v)
        trajectory_old_log_pi_v = self.calc_log_pi(
            trajectory_mu_v, self.model.base.actor.logstd, trajectory_actions_v
        )

        # 아래 변수는 전체 trajectory의 원소보다 1 적음
        trajectory_advantage_v, trajectory_target_action_value_v = self.get_advantage_and_target_action_values(
            trajectory, trajectory_states_v, device=self.device
        )

        # normalize advantages
        trajectory_advantage_v = trajectory_advantage_v - torch.mean(trajectory_advantage_v)
        trajectory_advantage_v /= torch.std(trajectory_advantage_v)

        # drop last entry from the trajectory, an our adv and target action value calculated without it
        trajectory = trajectory[:-1]
        trajectory_states_v = trajectory_states_v[:-1]
        trajectory_actions_v = trajectory_actions_v[:-1]
        trajectory_old_log_pi_v = trajectory_old_log_pi_v[:-1].detach()

        sum_loss_critic = 0.0
        sum_loss_actor = 0.0
        count_steps = 0

        for epoch in range(self.params.PPO_K_EPOCHS):
            for batch_offset in range(0, len(trajectory), self.params.PPO_TRAJECTORY_BATCH_SIZE):
                batch_l = batch_offset + self.params.PPO_TRAJECTORY_BATCH_SIZE

                batch_states_v = trajectory_states_v[batch_offset:batch_l]
                batch_actions_v = trajectory_actions_v[batch_offset:batch_l]
                batch_advantage_v = trajectory_advantage_v[batch_offset:batch_l].unsqueeze(-1)
                batch_target_action_value_v = trajectory_target_action_value_v[batch_offset:batch_l]
                batch_old_log_pi_v = trajectory_old_log_pi_v[batch_offset:batch_l]

                # actor training
                self.actor_optimizer.zero_grad()
                batch_mu_v, _ = self.model(batch_states_v)
                batch_log_pi_v = self.calc_log_pi(
                    batch_mu_v, self.model.base.actor.logstd, batch_actions_v
                )
                batch_ratio_v = torch.exp(batch_log_pi_v - batch_old_log_pi_v).to(self.device)
                batch_surrogate_v = batch_advantage_v * batch_ratio_v
                batch_clipped_ratio_v = torch.clamp(
                    batch_ratio_v, min=1.0 - self.params.PPO_EPSILON_CLIP, max=1.0 + self.params.PPO_EPSILON_CLIP
                )
                batch_clipped_surrogate_v = batch_advantage_v * batch_clipped_ratio_v
                loss_actor_v = -1.0 * torch.min(batch_surrogate_v, batch_clipped_surrogate_v).mean()

                batch_entropy_v = -1.0 * (torch.log(2.0 * math.pi * torch.exp(self.model.base.actor.logstd)) + 1) / 2
                loss_entropy_v = self.params.PPO_ENTROPY_WEIGHT * batch_entropy_v.mean()

                loss_actor_and_entropy_v = loss_actor_v + loss_entropy_v
                loss_actor_and_entropy_v.backward(retain_graph=True)
                self.actor_optimizer.step()

                # critic training
                self.critic_optimizer.zero_grad()
                batch_values_v = self.model.base.forward_critic(batch_states_v)
                loss_critic_v = F.mse_loss(batch_values_v.squeeze(-1), batch_target_action_value_v)
                loss_critic_v.backward()
                self.critic_optimizer.step()

                sum_loss_critic += loss_critic_v.item()
                sum_loss_actor += loss_actor_v.item()
                count_steps += 1

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, sum_loss_critic / count_steps, (sum_loss_actor / count_steps) * -1.0

    # @staticmethod
    # def calc_log_pi(mu_v, var_v, actions_v):
    #     # https://pytorch.org/docs/stable/generated/torch.clamp.html, clamp: 단단히 고정시키다.
    #     p1 = - ((mu_v - actions_v) ** 2) / (2 * var_v.clamp(min=1e-3))
    #     p2 = - torch.log(torch.sqrt(2 * math.pi * var_v))
    #     return p1 + p2

    @staticmethod
    def calc_log_pi(mu_v, logstd_v, actions_v):
        p1 = - ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
        return p1 + p2

    def get_advantage_and_target_action_values(self, trajectory, states_v, device="cpu"):
        """
        By trajectory calculate advantage and 1-step target action value
        :param trajectory: trajectory list
        :param critic_model: critic deep learning network
        :param states_v: states tensor
        :return: tuple with advantage numpy array and reference values
        """
        values_v = self.model.base.forward_critic(states_v)
        values = values_v.squeeze().data.cpu().numpy()

        # generalized advantage estimator: smoothed version of the advantage
        last_gae = 0.0
        result_advantages = []
        result_target_action_values = []
        for value, next_value, exp in zip(reversed(values[:-1]), reversed(values[1:]), reversed(trajectory[:-1])):
            if exp.done:
                delta = exp.reward - value
                last_gae = delta
            else:
                delta = exp.reward + self.params.GAMMA * next_value - value
                last_gae = delta + self.params.GAMMA * self.params.PPO_GAE_LAMBDA * last_gae
            result_advantages.append(last_gae)
            result_target_action_values.append(last_gae + value)

        advantage_v = torch.FloatTensor(list(reversed(result_advantages)))
        target_action_value_v = torch.FloatTensor(list(reversed(result_target_action_values)))
        return advantage_v.to(device), target_action_value_v.to(device)