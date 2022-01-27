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
    def __init__(self, observation_space, action_space, config):
        super(AgentDqn, self).__init__(observation_space, action_space, config)

        self.q_net = QNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions, config=config
        ).to(self.config.DEVICE)

        self.target_q_net = QNet(
            observation_shape=self.observation_shape, n_out_actions=self.n_out_actions,
            n_discrete_actions=self.n_discrete_actions, config=config
        ).to(self.config.DEVICE)

        self.q_net.share_memory()
        self.synchronize_models(source_model=self.q_net, target_model=self.target_q_net)

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.LEARNING_RATE)

        self.epsilon_tracker = EpsilonTracker(
            epsilon_init=self.config.EPSILON_INIT,
            epsilon_final=self.config.EPSILON_FINAL,
            epsilon_final_training_step=(self.config.EPSILON_FINAL_TRAINING_STEP_PROPORTION
                                         * self.config.MAX_TRAINING_STEPS)
        )
        self.epsilon = mp.Value('d', self.config.EPSILON_INIT)  # d: float

        self.model = self.q_net  # 에이전트 밖에서는 model이라는 이름으로 제어 모델 접근

        self.training_step = 0

        self.last_q_net_loss = mp.Value('d', 0.0)

    def get_action(self, obs, mode=AgentMode.TRAIN):
        out = self.q_net.forward(obs, save_hidden=True)

        if mode == AgentMode.TRAIN:
            coin = np.random.random()    # 0.0과 1.0사이의 임의의 값을 반환
            if coin < self.epsilon.value:
                action = np.random.randint(low=0, high=self.n_discrete_actions, size=len(obs))
            else:
                action = out.argmax(dim=-1)
                action = action.cpu().numpy()  # argmax: 가장 큰 값에 대응되는 인덱스 반환
        else:
            action = out.argmax(dim=-1)
            action = action.cpu().numpy()

        return action

    def train_dqn(self, training_steps_v):
        count_training_steps = 0

        # state_action_values.shape: torch.Size([32, 1])
        state_action_values = self.q_net(self.observations).gather(dim=1, index=self.actions)

        with torch.no_grad():
            # next_state_values.shape: torch.Size([32, 1])
            next_q_v = self.target_q_net(self.next_observations).max(dim=1, keepdim=True).values
            next_q_v[self.dones] = 0.0

            # target_state_action_values.shape: torch.Size([32, 1])
            target_state_action_values = self.rewards + self.config.GAMMA ** self.config.N_STEP * next_q_v

        # loss is just scalar torch value
        q_net_loss = self.config.LOSS_FUNCTION(state_action_values, target_state_action_values.detach())

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

        # sync
        if training_steps_v % self.config.TARGET_SYNC_INTERVAL_TRAINING_STEPS == 0:
            self.synchronize_models(source_model=self.q_net, target_model=self.target_q_net)

        self.epsilon.value = self.epsilon_tracker.epsilon(training_steps_v)

        self.last_q_net_loss.value = q_net_loss.item()

        count_training_steps += 1

        return count_training_steps