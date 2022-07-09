# https://www.deeplearningwizard.com/deep_learning/deep_reinforcement_learning_pytorch/dynamic_programming_frozenlake/
import torch.optim as optim
import numpy as np
import torch
import torch.multiprocessing as mp

from link_rl.e_agents.off_policy.off_policy_agent import OffPolicyAgent
from link_rl.h_utils.commons import EpsilonTracker
from link_rl.h_utils.types import AgentMode


class AgentDqn(OffPolicyAgent):
    def __init__(self, observation_space, action_space, config, need_train):
        super(AgentDqn, self).__init__(observation_space, action_space, config, need_train)

        # models
        self.q_net = self._model_creator.create_model()
        self.target_q_net = self._model_creator.create_model()

        # to(device)
        self.q_net.to(self.config.DEVICE)
        self.target_q_net.to(self.config.DEVICE)

        # Access
        self.model = self.q_net
        self.model.eval()

        # sync models
        self.synchronize_models(source_model=self.q_net, target_model=self.target_q_net)

        # share memory
        self.q_net.share_memory()

        # optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.LEARNING_RATE)

        # epsilon
        self.epsilon_tracker = EpsilonTracker(
            epsilon_init=self.config.EPSILON_INIT,
            epsilon_final=self.config.EPSILON_FINAL,
            epsilon_final_training_step=(self.config.EPSILON_FINAL_TRAINING_STEP_PROPORTION
                                         * self.config.MAX_TRAINING_STEPS)
        )
        self.epsilon = mp.Value('d', self.config.EPSILON_INIT)  # d: float

        # training step
        self.training_step = 0

        # loss
        self.last_q_net_loss = mp.Value('d', 0.0)

    def q_net_forward(self, obs):
        x = self.encoder(obs)
        q = self.q_net(x)
        return q

    def target_q_net_forward(self, obs):
        x = self.target_encoder(obs)
        q = self.target_q_net(x)
        return q

    @torch.no_grad()
    def get_action(self, obs, unavailable_actions=None, mode=AgentMode.TRAIN):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.config.DEVICE)
        if mode == AgentMode.TRAIN:
            coin = np.random.random()    # 0.0과 1.0사이의 임의의 값을 반환
            if coin < self.epsilon.value:
                q_values = torch.rand(len(obs), self.n_discrete_actions)
            else:
                q_values = self.q_net_forward(obs)
        else:
            q_values = self.q_net_forward(obs)

        if unavailable_actions is not None:
            for i, unavailable_action in enumerate(unavailable_actions):
                q_values[i][unavailable_action] = -np.inf
            q_values = torch.softmax(q_values, dim=-1)

        action = q_values.argmax(dim=-1)
        action = action.cpu().numpy()  # argmax: 가장 큰 값에 대응되는 인덱스 반환

        return action

    def train_dqn(self, training_steps_v):
        count_training_steps = 0

        # q_values.shape: torch.Size([32, 1])
        q_values = self.q_net_forward(self.observations).gather(dim=-1, index=self.actions)

        with torch.no_grad():  # autograd를 끔으로써 메모리 사용량을 줄이고 연산 속도를 높히기 위함
            # next_state_values.shape: torch.Size([32, 1])
            next_q_v = self.target_q_net_forward(self.next_observations).max(dim=-1, keepdim=True).values
            next_q_v[self.dones] = 0.0

            # target_state_action_values.shape: torch.Size([32, 1])
            target_q_values = self.rewards + self.config.GAMMA ** self.config.N_STEP * next_q_v
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

        # sync
        if training_steps_v % self.config.TARGET_SYNC_INTERVAL_TRAINING_STEPS == 0:
            self.synchronize_models(source_model=self.q_net, target_model=self.target_q_net)

        self.epsilon.value = self.epsilon_tracker.epsilon(training_steps_v)

        self.last_q_net_loss.value = q_net_loss.item()

        count_training_steps += 1

        return count_training_steps