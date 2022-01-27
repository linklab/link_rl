# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
import torch
import torch.nn.functional as F
from d_agents.off_policy.dqn.agent_dqn import AgentDqn


class AgentDoubleDqn(AgentDqn):
    def __init__(self, observation_space, action_space, config):
        super(AgentDoubleDqn, self).__init__(observation_space, action_space, config)

    def train_double_dqn(self, training_steps_v):
        count_training_steps = 0

        # state_action_values.shape: torch.Size([32, 1])
        state_action_values = self.q_net(self.observations).gather(dim=-1, index=self.actions)

        with torch.no_grad():
            target_argmax_action = torch.argmax(self.q_net(self.next_observations), dim=-1, keepdim=True)
            next_q_values = self.target_q_net(self.next_observations).gather(dim=-1, index=target_argmax_action)
            next_q_values[self.dones] = 0.0
            next_q_values = next_q_values.detach()

            # target_state_action_values.shape: torch.Size([32, 1])
            target_q_values = self.rewards + self.config.GAMMA ** self.config.N_STEP * next_q_values

        # loss is just scalar torch value
        q_net_loss = self.config.LOSS_FUNCTION(state_action_values, target_q_values)

        # print("observations.shape: {0}, actions.shape: {1}, "
        #       "next_observations.shape: {2}, rewards.shape: {3}, dones.shape: {4}".format(
        #     observations.shape, actions.shape,
        #     next_observations.shape, rewards.shape, dones.shape
        # ))
        # print("state_action_values.shape: {0}".format(state_action_values.shape))
        # print("next_state_values.shape: {0}".format(next_state_values.shape))
        # print("target_state_action_values.shape: {0}".format(
        #     target_state_action_values.shape
        # ))
        # print("loss.shape: {0}".format(loss.shape))

        self.optimizer.zero_grad()
        q_net_loss.backward()
        self.clip_model_config_grad_value(self.q_net.qnet_params_list)
        self.optimizer.step()

        # soft-sync
        self.soft_synchronize_models(source_model=self.q_net, target_model=self.target_q_net, tau=self.config.TAU)

        self.epsilon.value = self.epsilon_tracker.epsilon(training_steps_v)

        self.last_q_net_loss.value = q_net_loss.item()

        count_training_steps += 1

        return count_training_steps