# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/

import torch
from link_rl.e_agents.off_policy.dqn.agent_dqn import AgentDqn


class AgentDoubleDqn(AgentDqn):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentDoubleDqn, self).__init__(observation_space, action_space, config, need_train)

    def train_double_dqn(self, training_steps_v):
        count_training_steps = 0

        # q_values.shape: torch.Size([32, 1])
        q_values = self.q_net_forward(self.observations).gather(dim=-1, index=self.actions)

        with torch.no_grad():
            target_argmax_action = torch.argmax(self.q_net_forward(self.next_observations), dim=-1, keepdim=True)
            next_q_values = self.target_q_net_forward(self.next_observations).gather(dim=-1, index=target_argmax_action)
            next_q_values[self.dones] = 0.0
            next_q_values = next_q_values.detach()

            # target_q_values.shape: torch.Size([32, 1])
            target_q_values = self.rewards + self.config.GAMMA ** self.config.N_STEP * next_q_values
            if self.config.TARGET_VALUE_NORMALIZE:
                target_q_values = (target_q_values - torch.mean(target_q_values)) / (torch.std(target_q_values) + 1e-7)

        q_net_loss_each = self.config.LOSS_FUNCTION(q_values, target_q_values.detach(), reduction="none")

        if self.config.USE_PER:
            q_net_loss_each *= torch.FloatTensor(self.important_sampling_weights).to(self.config.DEVICE)[:, None]
            self.last_loss_for_per = q_net_loss_each

        q_net_loss = q_net_loss_each.mean()

        # print("observations.shape: {0}, actions.shape: {1}, "
        #       "next_observations.shape: {2}, rewards.shape: {3}, dones.shape: {4}".format(
        #     observations.shape, actions.shape,
        #     next_observations.shape, rewards.shape, dones.shape
        # ))
        # print("q_values.shape: {0}".format(q_values.shape))
        # print("next_state_values.shape: {0}".format(next_state_values.shape))
        # print("target_q_values.shape: {0}".format(
        #     target_q_values.shape
        # ))
        # print("loss.shape: {0}".format(loss.shape))

        self.optimizer.zero_grad()
        q_net_loss.backward()
        self.clip_model_config_grad_value(self.q_net.parameters())
        self.optimizer.step()

        if self.encoder_is_not_identity:
            self.train_encoder()

        # soft-sync
        self.soft_synchronize_models(source_model=self.q_net, target_model=self.target_q_net, tau=self.config.TAU)

        self.epsilon.value = self.epsilon_tracker.epsilon(training_steps_v)

        self.last_q_net_loss.value = q_net_loss.item()

        count_training_steps += 1

        return count_training_steps