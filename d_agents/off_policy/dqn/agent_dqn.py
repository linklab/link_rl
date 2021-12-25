# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
import torch.optim as optim
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from c_models.b_qnet_models import QNet
from d_agents.agent import Agent
from g_utils.commons import EpsilonTracker
from g_utils.types import AgentMode, ModelType


class AgentDqn(Agent):
    def __init__(self, observation_space, action_space, device, parameter, max_training_steps=None):
        super(AgentDqn, self).__init__(observation_space, action_space, device, parameter)

        self.q_net = QNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            device=device, parameter=parameter
        ).to(device)

        self.target_q_net = QNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            device=device, parameter=parameter
        ).to(device)

        self.q_net.share_memory()
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(
            self.q_net.parameters(), lr=self.parameter.LEARNING_RATE
        )

        self.epsilon_tracker = EpsilonTracker(
            epsilon_init=self.parameter.EPSILON_INIT,
            epsilon_final=self.parameter.EPSILON_FINAL,
            epsilon_final_training_step=self.parameter.EPSILON_FINAL_TRAINING_STEP_PERCENT * max_training_steps
        )
        self.epsilon = mp.Value('d', self.parameter.EPSILON_INIT)  # d: float

        self.model = self.q_net  # 에이전트 밖에서는 model이라는 이름으로 제어 모델 접근

        self.training_steps = 0

        self.last_q_net_loss = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        if mode == AgentMode.TRAIN:
            coin = np.random.random()    # 0.0과 1.0사이의 임의의 값을 반환
            if coin < self.epsilon.value:
                action = np.random.randint(low=0, high=self.n_out_actions, size=len(obs))
            else:
                out = self.q_net.forward(obs)
                action = out.argmax(dim=-1)
                action = action.cpu().numpy()  # argmax: 가장 큰 값에 대응되는 인덱스 반환
        else:
            out = self.q_net.forward(obs)
            action = out.argmax(dim=-1)
            action = action.cpu().numpy()

        return action

    def train_dqn(self, training_steps_v):
        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, next_observations, rewards, dones = self.buffer.sample(
            batch_size=self.parameter.BATCH_SIZE, device=self.device
        )

        # state_action_values.shape: torch.Size([32, 1])
        state_action_values = self.q_net(observations).gather(
            dim=1, index=actions
        )

        with torch.no_grad():
            # next_state_values.shape: torch.Size([32, 1])
            next_state_values = self.target_q_net(next_observations).max(
                dim=1, keepdim=True
            ).values
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

            # target_state_action_values.shape: torch.Size([32, 1])
            target_state_action_values = rewards + self.parameter.GAMMA ** self.parameter.N_STEP * next_state_values

        # loss is just scalar torch value
        q_net_loss = F.mse_loss(state_action_values, target_state_action_values)

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
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), self.parameter.CLIP_GRADIENT_VALUE)
        self.optimizer.step()

        # sync
        if training_steps_v % self.parameter.TARGET_SYNC_INTERVAL_TRAINING_STEPS == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.epsilon.value = self.epsilon_tracker.epsilon(training_steps_v)

        self.last_q_net_loss.value = q_net_loss.item()
