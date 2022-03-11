from abc import abstractmethod

from d_agents.agent import Agent
from g_utils.buffers import Buffer
from g_utils.prioritized_buffer import PrioritizedBuffer
from g_utils.types import AgentType, OffPolicyAgentTypes, ActorCriticAgentTypes


class OffPolicyAgent(Agent):
    def __init__(self, observation_space, action_space, config):
        super(OffPolicyAgent, self).__init__(observation_space, action_space, config)
        assert self.config.AGENT_TYPE in OffPolicyAgentTypes

        if self.config.USE_PER:
            assert self.config.AGENT_TYPE in OffPolicyAgentTypes
            self.replay_buffer = PrioritizedBuffer(action_space=action_space, config=self.config)
        else:
            self.replay_buffer = Buffer(action_space=action_space, config=self.config)

    def _before_train(self):
        if self.config.AGENT_TYPE in ActorCriticAgentTypes:
            assert self.actor_model
            assert self.critic_model
            assert self.model is self.actor_model

        if self.config.AGENT_TYPE == AgentType.MUZERO:
            self.episode_idxs, self.episode_historys = self.replay_buffer.sample_muzero(batch_size=sample_length)
        else:
            self.observations, self.actions, self.next_observations, self.rewards, self.dones, self.infos = self.replay_buffer.sample(
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
