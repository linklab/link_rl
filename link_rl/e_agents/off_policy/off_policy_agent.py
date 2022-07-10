import sys
from abc import abstractmethod

from link_rl.e_agents.agent import Agent
from link_rl.h_utils.buffers.buffer import Buffer
from link_rl.h_utils.buffers.prioritized_buffer import PrioritizedBuffer
from link_rl.h_utils.types import AgentType, OffPolicyAgentTypes
from link_rl.e_agents.off_policy.tdmpc.helper import ReplayBuffer


class OffPolicyAgent(Agent):
    def __init__(self, observation_space, action_space, config, need_train):
        super(OffPolicyAgent, self).__init__(observation_space, action_space, config)
        assert self.config.AGENT_TYPE in OffPolicyAgentTypes

        if need_train and not hasattr(self, "replay_buffer"):
            if self.config.AGENT_TYPE == AgentType.TDMPC:
                self.replay_buffer = ReplayBuffer(config, observation_space=observation_space, action_space=action_space)
            else:
                if self.config.USE_PER:
                    assert self.config.AGENT_TYPE in OffPolicyAgentTypes
                    self.replay_buffer = PrioritizedBuffer(
                        observation_space=observation_space, action_space=action_space, config=self.config
                    )
                    self.important_sampling_weights = None
                    self.last_loss_for_per = None
                else:
                    self.replay_buffer = Buffer(
                        observation_space=observation_space, action_space=action_space, config=self.config
                    )

            if self.config.USE_HER:
                from link_rl.h_utils.buffers.her_buffer import HerEpisodeBuffer
                self.her_buffer = HerEpisodeBuffer(
                    observation_space=observation_space, action_space=action_space, config=self.config
                )
                self.her_buffer.reset()
        else:
            self.replay_buffer = None
            self.her_buffer = None
            self.important_sampling_weights = None

        if self.config.USE_DRQ:
            from link_rl.e_agents.off_policy.tdmpc.helper import RandomShiftsAug

            self.aug = RandomShiftsAug(self.config)

    def _before_train(self):
        super(OffPolicyAgent, self)._before_train()

        # obs, next_obs, action, reward.unsqueeze(2), idxs, weights
        if self.config.AGENT_TYPE == AgentType.TDMPC:
            self.observations, self.next_observations, self.actions, self.rewards, self.idx, self.important_sampling_weights = \
            self.replay_buffer.sample()
            self.dones = None
        else:
            if self.config.USE_PER:
                self.observations, self.actions, self.next_observations, self.rewards, self.dones, \
                self.infos, self.important_sampling_weights = self.replay_buffer.sambuple(
                    batch_size=self.config.BATCH_SIZE
                )
            else:
                self.observations, self.actions, self.next_observations, self.rewards, self.dones, self.infos \
                    = self.replay_buffer.sample(batch_size=self.config.BATCH_SIZE)

            if self.config.USE_DRQ:
                self.observations = self.aug(self.observations)
                self.next_observations = self.aug(self.next_observations)

    def train(self, training_steps_v=None):
        count_training_steps = 0

        if len(self.replay_buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
            self._before_train()

            if self.config.AGENT_TYPE in (AgentType.DQN, AgentType.DUELING_DQN):
                count_training_steps = self.train_dqn(training_steps_v=training_steps_v)

            elif self.config.AGENT_TYPE in (AgentType.DOUBLE_DQN, AgentType.DOUBLE_DUELING_DQN):
                count_training_steps = self.train_double_dqn(training_steps_v=training_steps_v)

            elif self.config.AGENT_TYPE == AgentType.DDPG:
                count_training_steps = self.train_ddpg()

            elif self.config.AGENT_TYPE == AgentType.TD3:
                count_training_steps = self.train_td3(training_steps_v=training_steps_v)

            elif self.config.AGENT_TYPE == AgentType.SAC:
                count_training_steps = self.train_sac(training_steps_v=training_steps_v)

            elif self.config.AGENT_TYPE == AgentType.TDMPC:
                # print("TDMPC TRAIN - START")
                if training_steps_v == 0:
                    for train_idx in range(self.config.SEED_STEPS):
                        # print("{0}".format(train_idx), end=" ")
                        train_info = self.train_tdmpc(training_steps_v=training_steps_v)
                    count_training_steps = self.config.SEED_STEPS
                else:
                    for train_idx in range(int(self.config.FIXED_TOTAL_TIME_STEPS_PER_EPISODE/self.config.ACTION_REPEAT)):
                        # print("{0}".format(train_idx), end=" ")
                        train_info = self.train_tdmpc(training_steps_v=training_steps_v)
                    print()
                    count_training_steps = int(self.config.FIXED_TOTAL_TIME_STEPS_PER_EPISODE/self.config.ACTION_REPEAT)
                # print("\nTDMPC TRAIN - END")
            else:
                raise ValueError()

            self._after_train()
        else:
            pass

        return count_training_steps

    def _after_train(self):
        super(OffPolicyAgent, self)._after_train()

        if self.config.USE_PER and self.last_loss_for_per is not None:
            self.replay_buffer.update_priorities(self.last_loss_for_per.detach().cpu().numpy())

    # OFF POLICY
    @abstractmethod
    def train_dqn(self, training_steps_v):
        raise NotImplementedError()

    @abstractmethod
    def train_double_dqn(self, training_steps_v):
        raise NotImplementedError()

    @abstractmethod
    def train_ddpg(self):
        raise NotImplementedError()

    @abstractmethod
    def train_td3(self, training_steps_v):
        raise NotImplementedError()

    @abstractmethod
    def train_sac(self, training_steps_v):
        raise NotImplementedError()

    @abstractmethod
    def train_tdmpc(self, training_steps_v):
        raise NotImplementedError()
