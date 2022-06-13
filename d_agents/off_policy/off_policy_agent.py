from abc import abstractmethod

from d_agents.agent import Agent
from g_utils.buffers.buffer import Buffer
from g_utils.buffers.prioritized_buffer import PrioritizedBuffer
from g_utils.types import AgentType, OffPolicyAgentTypes, ActorCriticAgentTypes
from d_agents.off_policy.tdmpc.helper import ReplayBuffer


class OffPolicyAgent(Agent):
    def __init__(self, observation_space, action_space, config):
        super(OffPolicyAgent, self).__init__(observation_space, action_space, config)
        assert self.config.AGENT_TYPE in OffPolicyAgentTypes

        if self.config.AGENT_TYPE == AgentType.TDMPC:
            self.replay_buffer = ReplayBuffer(config, observation_space=observation_space, action_space=action_space)
        else:
            if self.config.USE_PER:
                assert self.config.AGENT_TYPE in OffPolicyAgentTypes
                self.replay_buffer = PrioritizedBuffer(
                    observation_space=observation_space, action_space=action_space, config=self.config
                )
                self.important_sampling_weights = None
            else:
                self.replay_buffer = Buffer(
                    observation_space=observation_space, action_space=action_space, config=self.config
                )

            if self.config.USE_HER:
                from g_utils.buffers.her_buffer import HerEpisodeBuffer
                self.her_buffer = HerEpisodeBuffer(
                    observation_space=observation_space, action_space=action_space, config=self.config
                )
                self.her_buffer.reset()

    def _before_train(self):
        if self.config.AGENT_TYPE in ActorCriticAgentTypes:
            assert self.actor_model
            assert self.critic_model
            assert self.model is self.actor_model
        # obs, next_obs, action, reward.unsqueeze(2), idxs, weights
        if self.config.AGENT_TYPE == AgentType.MUZERO:
            self.episode_idxs, self.episode_historys = self.replay_buffer.sample_muzero(batch_size=self.config.BATCH_SIZE)
        elif self.config.AGENT_TYPE == AgentType.TDMPC:
            self.observations, self.next_observations, self.actions, self.rewards, self.idx, self.weights = \
            self.replay_buffer.sample()
        else:
            if self.config.USE_PER:
                self.observations, self.actions, self.next_observations, self.rewards, self.dones, \
                self.infos, self.important_sampling_weights = self.replay_buffer.sample(
                    batch_size=self.config.BATCH_SIZE
                )
            else:
                self.observations, self.actions, self.next_observations, self.rewards, self.dones, \
                self.infos = self.replay_buffer.sample(
                    batch_size=self.config.BATCH_SIZE
                )

    def train(self, training_steps_v=None):
        count_training_steps = 0

        if self.config.AGENT_TYPE in (AgentType.DQN, AgentType.DUELING_DQN):
            if len(self.replay_buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                self._before_train()
                count_training_steps, q_net_loss_each = self.train_dqn(training_steps_v=training_steps_v)
                self._after_train(q_net_loss_each)

        elif self.config.AGENT_TYPE in (AgentType.DOUBLE_DQN, AgentType.DOUBLE_DUELING_DQN):
            if len(self.replay_buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                self._before_train()
                count_training_steps, q_net_loss_each = self.train_double_dqn(training_steps_v=training_steps_v)
                self._after_train(q_net_loss_each)

        elif self.config.AGENT_TYPE == AgentType.DDPG:
            if len(self.replay_buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                self._before_train()
                count_training_steps, critic_loss_each = self.train_ddpg()
                self._after_train(critic_loss_each)

        elif self.config.AGENT_TYPE == AgentType.TD3:
            if len(self.replay_buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                self._before_train()
                count_training_steps, critic_loss_each = self.train_td3(training_steps_v=training_steps_v)
                self._after_train(critic_loss_each)

        elif self.config.AGENT_TYPE == AgentType.SAC:
            if len(self.replay_buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                self._before_train()
                count_training_steps, critic_loss_each = self.train_sac(training_steps_v=training_steps_v)
                self._after_train(critic_loss_each)

        elif self.config.AGENT_TYPE == AgentType.MUZERO:
            assert self.config.N_VECTORIZED_ENVS == 1
            if len(self.replay_buffer) >= self.config.BATCH_SIZE:
                self._before_train()
                count_training_steps, critic_loss_each = self.train_muzero(training_steps_v=training_steps_v)
                self._after_train(critic_loss_each)

        elif self.config.AGENT_TYPE == AgentType.TDMPC:
            if len(self.replay_buffer) >= self.config.MIN_BUFFER_SIZE_FOR_TRAIN:
                self._before_train()
                if training_steps_v == 0:
                    for _ in range(self.config.SEED_STEPS):
                        train_info = self.train_tdmpc(training_steps_v=training_steps_v)
                    count_training_steps = 5000
                else:
                    for _ in range(int(1000/self.config.ACTION_REPEAT)):
                        train_info = self.train_tdmpc(training_steps_v=training_steps_v)
                    count_training_steps = int(1000/self.config.ACTION_REPEAT)

        else:
            raise ValueError()

        return count_training_steps

    def _after_train(self, loss_each=None):
        if loss_each is not None and self.config.USE_PER:
            self.replay_buffer.update_priorities(loss_each.detach().cpu().numpy())

        if self.config.AGENT_TYPE == AgentType.MUZERO:
            del self.episode_historys
            del self.episode_idxs
        else:
            del self.observations
            del self.actions
            del self.next_observations
            del self.rewards
            del self.dones

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
    def train_muzero(self, training_steps_v):
        raise NotImplementedError()

    @abstractmethod
    def train_tdmpc(self, training_steps_v):
        raise NotImplementedError()
