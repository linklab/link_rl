# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
import torch
import torch.nn.functional as F
from d_agents.off_policy.dqn.agent_dqn import AgentDqn


class AgentDoubleDqn(AgentDqn):
    def __init__(self, observation_space, action_space, device, parameter):
        super(AgentDoubleDqn, self).__init__(observation_space, action_space, device, parameter)

    def train_double_dqn(self, training_steps_v):
        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, next_observations, rewards, dones = self.buffer.sample(
            batch_size=self.parameter.BATCH_SIZE, device=self.device
        )

        # state_action_values.shape: torch.Size([32, 1])
        state_action_values = self.q_net(observations).gather(dim=-1, index=actions)

        with torch.no_grad():
            target_argmax_action = torch.argmax(self.target_q_net(next_observations), dim=-1, keepdim=True)
            next_q_values = self.q_net(next_observations).gather(dim=-1, index=target_argmax_action)
            next_q_values[dones] = 0.0
            next_q_values = next_q_values.detach()

            # target_state_action_values.shape: torch.Size([32, 1])
            target_q_values = rewards + self.parameter.GAMMA ** self.parameter.N_STEP * next_q_values

        # loss is just scalar torch value
        q_net_loss = F.mse_loss(state_action_values, target_q_values)

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
        self.clip_model_parameter_grad_value(self.q_net.parameters())
        self.optimizer.step()

        # soft-sync
        self.soft_synchronize_models(source_model=self.q_net, target_model=self.target_q_net, tau=self.parameter.TAU)

        self.epsilon.value = self.epsilon_tracker.epsilon(training_steps_v)

        self.last_q_net_loss.value = q_net_loss.item()