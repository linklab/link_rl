import torch

from codes.c_models.discrete_action.discrete_actor_critic_model import DiscreteActorCriticModel
from codes.d_agents.on_policy.ppo.ppo_agent import AgentPPO
from codes.e_utils import rl_utils
from codes.d_agents.actions import DiscreteCategoricalActionSelector
from codes.e_utils.names import DeepLearningModelName


class AgentDiscretePPO(AgentPPO):
    """
    """
    def __init__(
            self, worker_id, input_shape, action_shape, num_outputs, params, device
    ):
        assert params.DEEP_LEARNING_MODEL in [
            DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_MLP,
            DeepLearningModelName.STOCHASTIC_DISCRETE_ACTOR_CRITIC_CNN,
        ]

        super(AgentDiscretePPO, self).__init__(
            worker_id=worker_id, params=params, action_shape=action_shape, device=device
        )
        self.__name__ = "AgentDiscretePPO"

        self.train_action_selector = DiscreteCategoricalActionSelector()
        self.test_and_play_action_selector = DiscreteCategoricalActionSelector()

        self.model = DiscreteActorCriticModel(
            worker_id=worker_id,
            input_shape=input_shape,
            num_outputs=num_outputs,
            params=params,
            device=device
        ).to(device)

        # self.optimizer = rl_utils.get_optimizer(
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

    def __call__(self, states, critics=None):
        return self.discrete_call(states, critics)

    def train(self, step_idx):
        trajectory = self.buffer.sample(batch_size=None)

        # trajectory_states_v: (2049, 4)
        trajectory_states = [experience.state for experience in trajectory]
        trajectory_states_v = torch.FloatTensor(trajectory_states).to(self.device)

        # trajectory_actions_v: (2049,)
        trajectory_actions = [experience.action for experience in trajectory]
        trajectory_actions_v = torch.LongTensor(trajectory_actions).to(self.device)

        # trajectory_probs_v: (2049, 2)
        # trajectory_values_v: (2849, 1)
        trajectory_probs_v, trajectory_values_v = self.model.base.forward(trajectory_states_v)

        # trajectory_old_log_pi_action_v: (2849, 1)
        trajectory_old_log_pi_action_v = torch.log(
            trajectory_probs_v.gather(dim=1, index=trajectory_actions_v.unsqueeze(-1)) + 1e-5
        )

        # trajectory_logits_v, trajectory_values_v = self.model(trajectory_states_v)
        # trajectory_log_pi_v = F.log_softmax(trajectory_logits_v, dim=1)
        # trajectory_old_log_pi_action_v = trajectory_log_pi_v.gather(
        #     dim=1, index=trajectory_actions_v.unsqueeze(-1)
        # ).squeeze(-1)

        # 아래 변수는 전체 trajectory의 원소보다 1 적음
        # trajectory_advantage_v: (2048,)
        # trajectory_target_action_value_v: (2048,)
        with torch.no_grad():
            trajectory_advantage_v, trajectory_target_action_value_v = self.get_advantage_and_target_action_values(
                trajectory, trajectory_values_v, device=self.device
            )
            # normalize advantages
            trajectory_advantage_v = trajectory_advantage_v - torch.mean(trajectory_advantage_v)
            trajectory_advantage_v /= torch.std(trajectory_advantage_v) + 1e-5

        # drop last entry from the trajectory, an our adv and target action value calculated without it
        trajectory = trajectory[:-1]
        trajectory_states_v = trajectory_states_v[:-1]
        trajectory_actions_v = trajectory_actions_v[:-1]
        trajectory_old_log_pi_action_v = trajectory_old_log_pi_action_v[:-1].detach()

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
                batch_old_log_pi_action_v = trajectory_old_log_pi_action_v[batch_offset:batch_l]

                # batch_probs_v: (64, 2)
                # batch_values_v: (64, 1)
                batch_probs_v, batch_values_v = self.model.base.forward(batch_states_v)

                batch_loss_critic_v = self.backward_and_step_for_critic(batch_values_v, batch_target_action_value_v)

                # batch_log_pi_v = F.log_softmax(batch_logits_v, dim=1)
                # print(batch_logits_v.size())
                # batch_log_pi_action_v = batch_log_pi_v.gather(dim=1, index=batch_actions_v.unsqueeze(-1)).squeeze(-1)

                # batch_log_pi_action_v: (64,)
                batch_log_pi_action_v = torch.log(
                    batch_probs_v.gather(dim=1, index=batch_actions_v.unsqueeze(-1)) + 1e-5
                )

                # batch_probs_v: (64, 2)
                # batch_log_pi_v: (64, 2)
                batch_log_pi_v = torch.log(batch_probs_v + 1e-5)
                batch_loss_entropy_v = (batch_probs_v * batch_log_pi_v).sum(dim=1).mean()

                batch_loss_actor_v = self.backward_and_step_for_actor(
                    batch_log_pi_action_v, batch_old_log_pi_action_v, batch_advantage_v, batch_loss_entropy_v
                )

                sum_loss_critic += batch_loss_critic_v.item()
                sum_loss_actor += batch_loss_actor_v.item()
                count_steps += 1

        gradients = self.model.get_gradients_for_current_parameters()

        return gradients, sum_loss_critic / count_steps, (sum_loss_actor / count_steps) * -1.0