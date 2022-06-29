import torch
from gym.spaces import Box, Discrete

from link_rl.g_utils.types import AgentType


class Episode(object):
    """Storage object for a single episode."""

    def __init__(self, config, init_obs, n_out_action):
        self.config = config
        episode_length = int(1000/config.ACTION_REPEAT)
        dtype = torch.float32 if not config.FROM_PIXELS else torch.uint8
        self.obs = torch.empty((episode_length + 1, *init_obs.shape), dtype=dtype, device=self.config.DEVICE)
        self.obs[0] = torch.tensor(init_obs, dtype=dtype, device=self.config.DEVICE)
        self.action = torch.empty((episode_length, n_out_action), dtype=torch.float32, device=self.config.DEVICE)
        self.reward = torch.empty((episode_length,), dtype=torch.float32, device=self.config.DEVICE)
        self.cumulative_reward = 0
        self.done = False
        self._idx = 0
        self.info = {}

    def __len__(self):
        return self._idx

    @property
    def first(self):
        return len(self) == 0

    def __add__(self, transition):
        self.add(*transition)
        return self

    def append(self, transition):
        self.add(*transition)
        return self

    def add(self, obs, action, reward, done, info):
        self.obs[self._idx + 1] = torch.tensor(obs, dtype=self.obs.dtype, device=self.obs.device)
        self.action[self._idx] = action
        self.reward[self._idx] = reward
        self.cumulative_reward += reward
        self.done = done
        self.info = info
        self._idx += 1


def get_agent(observation_space, action_space, config=None, need_train=True):
    assert isinstance(observation_space, Box)

    if config.AGENT_TYPE == AgentType.DQN:
        assert isinstance(action_space, Discrete)
        from link_rl.d_agents.off_policy.dqn.agent_dqn import AgentDqn
        agent = AgentDqn(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.DOUBLE_DQN:
        assert isinstance(action_space, Discrete)
        from link_rl.d_agents.off_policy.dqn.agent_double_dqn import AgentDoubleDqn
        agent = AgentDoubleDqn(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.DUELING_DQN:
        assert isinstance(action_space, Discrete)
        from link_rl.d_agents.off_policy.dqn.agent_dueling_dqn import AgentDuelingDqn
        agent = AgentDuelingDqn(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.DOUBLE_DUELING_DQN:
        assert isinstance(action_space, Discrete)
        from link_rl.d_agents.off_policy.dqn.agent_double_dueling_dqn import AgentDoubleDuelingDqn
        agent = AgentDoubleDuelingDqn(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.REINFORCE:
        assert config.N_ACTORS * config.N_VECTORIZED_ENVS == 1, "TOTAL NUMBERS OF ENVS should be one"
        from link_rl.d_agents.on_policy.reinforce.agent_reinforce import AgentReinforce
        agent = AgentReinforce(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.A2C:
        from link_rl.d_agents.on_policy.a2c.agent_a2c import AgentA2c
        agent = AgentA2c(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.A3C:
        from link_rl.d_agents.on_policy.a3c.agent_a3c import AgentA3c
        agent = AgentA3c(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.PPO:
        from link_rl.d_agents.on_policy.ppo.agent_ppo import AgentPpo
        agent = AgentPpo(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.ASYNCHRONOUS_PPO:
        from link_rl.d_agents.on_policy.asynchronous_ppo.agent_asynchronous_ppo import AgentAsynchronousPpo
        agent = AgentAsynchronousPpo(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.PPO_TRAJECTORY:
        from link_rl.d_agents.on_policy.ppo.agent_ppo_trajectory import AgentPpoTrajectory
        assert hasattr(config, "BATCH_SIZE")
        agent = AgentPpoTrajectory(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.DDPG:
        assert isinstance(action_space, Box)
        from link_rl.d_agents.off_policy.ddpg.agent_ddpg import AgentDdpg
        agent = AgentDdpg(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.TD3:
        assert isinstance(action_space, Box)
        from link_rl.d_agents.off_policy.td3.agent_td3 import AgentTd3
        agent = AgentTd3(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.SAC:
        assert isinstance(action_space, Box)
        from link_rl.d_agents.off_policy.sac.agent_sac import AgentSac
        agent = AgentSac(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    elif config.AGENT_TYPE == AgentType.TDMPC:
        from link_rl.d_agents.off_policy.tdmpc.agent_tdmpc import AgentTdmpc
        agent = AgentTdmpc(
            observation_space=observation_space, action_space=action_space, config=config, need_train=need_train
        )
    else:
        raise ValueError()

    return agent
